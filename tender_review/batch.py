from __future__ import annotations

import hashlib
import json
import shutil
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from .config import Settings
from .platform import PlatformClient, PlatformError
from .results import repair_mojibake


ISSUE_STATUSES = {"PENDING_CONFIRMATION", "REJECTED", "NON_COMPLIANT"}
TERMINAL_FAILURE_STATUSES = {
    "FAILED",
    "ERROR",
    "CANCELLED",
    "ABORTED",
    "AI_REVIEW_FAILED",
}


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    relative_path: Path
    expected_issue: str
    archive_directory: str

    @property
    def file_name(self) -> str:
        return self.path.name


def discover_documents(source_dir: Path) -> list[SourceDocument]:
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    paths = sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.relative_to(source_dir).as_posix(),
    )
    if not paths:
        raise ValueError(f"No PDF files found under: {source_dir}")

    normalized_names: dict[str, Path] = {}
    documents: list[SourceDocument] = []
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(source_dir)
        if len(relative.parts) < 2:
            raise ValueError(f"PDF must be inside an issue folder: {path}")
        normalized = _normalized_name(path.name)
        if normalized in normalized_names:
            raise ValueError(
                "Duplicate PDF names cannot be correlated with platform results: "
                f"{normalized_names[normalized]} and {path}"
            )
        normalized_names[normalized] = path
        documents.append(
            SourceDocument(
                path=path.resolve(),
                relative_path=relative,
                expected_issue=relative.parts[0],
                archive_directory=f"待审项目_{index:03d}",
            )
        )
    return documents


def build_batch_zip(documents: list[SourceDocument], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for document in documents:
            archive.write(
                document.path,
                arcname=f"{document.archive_directory}/{document.file_name}",
            )
    return target


def write_structured_outputs(
    *,
    report: dict[str, Any],
    documents: list[SourceDocument],
    output_dir: Path,
    run_id: str,
    product_id: str,
    task_id: str,
    review_item_ids: list[str],
    raw_report_path: Path,
    task_ids_by_file: dict[str, str] | None = None,
    product_ids_by_file: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    grouped: dict[str, list[dict[str, Any]]] = {}
    unmatched_rows: list[dict[str, Any]] = []

    source_names = {_normalized_name(document.file_name) for document in documents}
    for row in report.get("data") or []:
        if not isinstance(row, dict):
            continue
        file_name = repair_mojibake(
            row.get("fileName") or row.get("reviewFileName") or ""
        )
        normalized = _normalized_name(str(file_name))
        if normalized not in source_names:
            unmatched_rows.append(_normalize_opinion(row))
            continue
        grouped.setdefault(normalized, []).append(row)

    file_summaries: list[dict[str, Any]] = []
    total_issue_count = 0
    total_opinion_count = 0
    missing_files = 0
    incomplete_files = 0

    for document in documents:
        file_task_id = task_id
        if task_ids_by_file:
            file_task_id = task_ids_by_file.get(
                _normalized_name(document.file_name), task_id
            )
        file_product_id = product_id
        if product_ids_by_file:
            file_product_id = product_ids_by_file.get(
                _normalized_name(document.file_name), product_id
            )
        rows = grouped.get(_normalized_name(document.file_name), [])
        opinions = sorted(
            (_normalize_opinion(row) for row in rows), key=_opinion_sort_key
        )
        evaluated_items = sorted(
            {
                str(opinion["review_item"])
                for opinion in opinions
                if opinion.get("review_item") not in (None, "")
            },
            key=_review_item_sort_key,
        )
        issue_count = sum(
            1 for opinion in opinions if opinion.get("compliant") is False
        )
        failed_count = sum(
            1
            for opinion in opinions
            if str(opinion.get("status") or "").upper()
            in TERMINAL_FAILURE_STATUSES
        )
        missing_items = sorted(
            set(review_item_ids) - set(evaluated_items), key=_review_item_sort_key
        )
        if not opinions:
            status = "missing"
            missing_files += 1
        elif failed_count or missing_items:
            status = "partial"
            incomplete_files += 1
        else:
            status = "completed"

        relative_output = document.relative_path.with_suffix(".json")
        output_path = output_dir / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "source": {
                "file_name": document.file_name,
                "relative_path": document.relative_path.as_posix(),
                "expected_issue": document.expected_issue,
                "size_bytes": document.path.stat().st_size,
                "sha256": _sha256(document.path),
            },
            "review": {
                "status": status,
                "product_id": file_product_id,
                "task_id": file_task_id,
                "reviewed_at": generated_at,
                "review_item_count": len(evaluated_items),
                "opinion_count": len(opinions),
                "issue_count": issue_count,
                "failed_count": failed_count,
                "missing_review_items": missing_items,
                "opinions": opinions,
            },
        }
        _write_json(output_path, payload)

        total_issue_count += issue_count
        total_opinion_count += len(opinions)
        file_summaries.append(
            {
                "file_name": document.file_name,
                "relative_path": document.relative_path.as_posix(),
                "expected_issue": document.expected_issue,
                "task_id": file_task_id,
                "status": status,
                "review_item_count": len(evaluated_items),
                "opinion_count": len(opinions),
                "issue_count": issue_count,
                "failed_count": failed_count,
                "missing_review_items": missing_items,
                "output_file": relative_output.as_posix(),
            }
        )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "product_id": product_id,
        "task_id": task_id,
        "task_ids_by_file": task_ids_by_file or {},
        "product_ids_by_file": product_ids_by_file or {},
        "generated_at": generated_at,
        "raw_report": str(raw_report_path.resolve()),
        "review_item_ids": review_item_ids,
        "totals": {
            "source_files": len(documents),
            "completed_files": len(documents) - missing_files - incomplete_files,
            "partial_files": incomplete_files,
            "missing_files": missing_files,
            "opinion_count": total_opinion_count,
            "issue_count": total_issue_count,
            "unmatched_report_rows": len(unmatched_rows),
        },
        "files": file_summaries,
    }
    if unmatched_rows:
        summary["unmatched_rows"] = unmatched_rows
    _write_json(output_dir / "汇总.json", summary)
    return summary


class BatchReviewRunner:
    def __init__(
        self,
        settings: Settings,
        source_dir: Path,
        output_dir: Path,
        timeout_seconds: float = 14400,
        run_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.source_dir = source_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.timeout_seconds = timeout_seconds
        self.run_dir = (
            run_dir.resolve()
            if run_dir is not None
            else self._new_run_dir(settings.runs_dir)
        )
        self.run_dir.mkdir(parents=True, exist_ok=run_dir is not None)
        self.platform = PlatformClient(settings.base_url, settings.username)

    def run(self) -> dict[str, Any]:
        try:
            documents = discover_documents(self.source_dir)
            zip_path = self.run_dir / "待审文件_批量审批.zip"
            build_batch_zip(documents, zip_path)
            excel_copy = self.run_dir / "审核要点_原始.xlsx"
            shutil.copy2(self.settings.excel_path, excel_copy)
            manifest = self._manifest(documents, zip_path, excel_copy)
            self._write_run_json("00_清单.json", manifest)

            self._log("生成临时平台 token")
            self.platform.refresh_token()
            product_name = f"招标文件批量审核_{datetime.now():%Y%m%d%H%M%S}"

            self._log(f"创建批量审核产品: {product_name}")
            product = self.platform.create_product(
                product_name, self.settings.product_property
            )
            product_id = str((product["response"].get("data") or {})["id"])
            self._write_run_json("01_创建产品.json", product)

            self._log("导入完整审核要点")
            imported = self.platform.import_review_points(
                product_id, self.settings.excel_path
            )
            self._write_run_json("02_导入审核要点_响应.json", imported)

            self._log(f"上传包含 {len(documents)} 份 PDF 的批量 ZIP")
            upload = self.platform.upload_zip(product_id, zip_path)
            self._write_run_json("03_上传ZIP_响应.json", upload)
            self._validate_upload(upload, documents)

            review_item_ids = _review_item_ids(upload)
            self._log(
                f"创建任务: {len(documents)} 份文件 x {len(review_item_ids)} 个审核要点"
            )
            task = self.platform.create_task(
                product_id=product_id,
                product_name=product_name,
                applicant_name=self.settings.applicant_name,
                product_property=self.settings.product_property,
                clinical_evaluation=self.settings.clinical_evaluation,
                upload=upload,
            )
            task_id = task["task_id"]
            self._write_run_json("04_创建任务.json", task)
            started = self.platform.start_task(task_id)
            self._write_run_json("05_启动任务_响应.json", started)

            self._log(f"等待任务 {task_id} 完成")
            progress = self.platform.wait_for_task(
                task_id,
                self.settings.poll_interval_seconds,
                self.timeout_seconds,
                on_progress=self._progress,
            )
            self._write_run_json("06_最终进度.json", progress)
            report = self.platform.get_results(task_id)
            raw_report_path = self._write_run_json("07_AI审核报告.json", report)
            summary = write_structured_outputs(
                report=report,
                documents=documents,
                output_dir=self.output_dir,
                run_id=self.run_dir.name,
                product_id=product_id,
                task_id=task_id,
                review_item_ids=review_item_ids,
                raw_report_path=raw_report_path,
            )
            self._write_run_json("08_结构化汇总.json", summary)
            return summary
        except Exception as exc:
            self._write_run_json(
                "99_运行失败.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )
            raise

    def resume(self) -> dict[str, Any]:
        try:
            manifest = self._read_run_json("00_清单.json")
            documents = _documents_from_manifest(manifest)
            product = self._read_run_json("01_创建产品.json")
            upload = self._read_run_json("03_上传ZIP_响应.json")
            task = self._read_run_json("04_创建任务.json")
            product_id = str((product["response"].get("data") or {})["id"])
            task_id = str(task["task_id"])
            review_item_ids = _review_item_ids(upload)
            raw_report_path = self.run_dir / "07_AI审核报告.json"

            if raw_report_path.is_file():
                report = self._read_run_json("07_AI审核报告.json")
            else:
                self._log("刷新 token 并恢复任务轮询")
                self.platform.refresh_token()
                progress = self.platform.wait_for_task(
                    task_id,
                    self.settings.poll_interval_seconds,
                    self.timeout_seconds,
                    on_progress=self._progress,
                )
                self._write_run_json("06_最终进度.json", progress)
                report = self.platform.get_results(task_id)
                raw_report_path = self._write_run_json("07_AI审核报告.json", report)

            summary = write_structured_outputs(
                report=report,
                documents=documents,
                output_dir=self.output_dir,
                run_id=self.run_dir.name,
                product_id=product_id,
                task_id=task_id,
                review_item_ids=review_item_ids,
                raw_report_path=raw_report_path,
            )
            self._write_run_json("08_结构化汇总.json", summary)
            return summary
        except Exception as exc:
            self._write_run_json(
                "99_运行失败.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )
            raise

    def _manifest(
        self,
        documents: list[SourceDocument],
        zip_path: Path,
        excel_copy: Path,
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_dir.name,
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "excel_path": str(self.settings.excel_path),
            "excel_copy": str(excel_copy),
            "zip_path": str(zip_path),
            "documents": [
                {
                    "path": str(document.path),
                    "relative_path": document.relative_path.as_posix(),
                    "expected_issue": document.expected_issue,
                    "archive_directory": document.archive_directory,
                }
                for document in documents
            ],
        }

    def _validate_upload(
        self, upload: dict[str, Any], documents: list[SourceDocument]
    ) -> None:
        matched_names = {
            _normalized_name(
                str(
                    repair_mojibake(
                        item.get("fileName") or item.get("reviewFileName") or ""
                    )
                )
            )
            for item in self.platform._matches(upload)
            if isinstance(item, dict)
        }
        missing = [
            document.file_name
            for document in documents
            if _normalized_name(document.file_name) not in matched_names
        ]
        if missing:
            raise RuntimeError(
                "Platform did not match uploaded PDFs: " + ", ".join(missing)
            )
        if not _review_item_ids(upload):
            raise RuntimeError("Platform returned no review points for the batch")

    def _progress(self, snapshot: dict[str, Any]) -> None:
        self._write_run_json("06_当前进度.json", snapshot)
        self._log(
            "进度: "
            f"{snapshot.get('percentage', 0)}% "
            f"status={snapshot.get('status')} "
            f"detail={snapshot.get('progressStatus', '')}"
        )

    def _write_run_json(self, name: str, value: Any) -> Path:
        path = self.run_dir / name
        _write_json(path, value)
        return path

    def _read_run_json(self, name: str) -> Any:
        return json.loads((self.run_dir / name).read_text(encoding="utf-8"))

    @staticmethod
    def _new_run_dir(runs_dir: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return runs_dir.resolve() / f"batch_{timestamp}"

    @staticmethod
    def _log(message: str) -> None:
        try:
            print(f"[BATCH] {message}", flush=True)
        except OSError:
            # The desktop task can outlive the terminal pipe after an interruption.
            pass


class IndividualTaskBatchRunner(BatchReviewRunner):
    """Review every PDF in its own platform task under one shared product."""

    def run(self) -> dict[str, Any]:
        try:
            documents = discover_documents(self.source_dir)
            excel_copy = self.run_dir / "审核要点_原始.xlsx"
            shutil.copy2(self.settings.excel_path, excel_copy)
            manifest = {
                "run_id": self.run_dir.name,
                "mode": "one_task_per_pdf",
                "source_dir": str(self.source_dir),
                "output_dir": str(self.output_dir),
                "excel_path": str(self.settings.excel_path),
                "excel_copy": str(excel_copy),
                "documents": [
                    {
                        "path": str(document.path),
                        "relative_path": document.relative_path.as_posix(),
                        "expected_issue": document.expected_issue,
                        "archive_directory": document.archive_directory,
                    }
                    for document in documents
                ],
            }
            self._write_run_json("00_清单.json", manifest)

            self._log("生成临时平台 token")
            self.platform.refresh_token()
            product_name = f"招标文件逐份审核_{datetime.now():%Y%m%d%H%M%S}"
            self._log(f"创建共享审核产品: {product_name}")
            product = self.platform.create_product(
                product_name, self.settings.product_property
            )
            product_id = str((product["response"].get("data") or {})["id"])
            self._write_run_json("01_创建产品.json", product)
            imported = self.platform.import_review_points(
                product_id, self.settings.excel_path
            )
            self._write_run_json("02_导入审核要点_响应.json", imported)

            records = self._submit_missing_tasks(
                documents=documents,
                product_id=product_id,
                product_name=product_name,
                records=[],
            )
            return self._wait_and_finalize(
                documents=documents,
                product_id=product_id,
                records=records,
            )
        except Exception as exc:
            self._record_failure(exc)
            raise

    def resume(self) -> dict[str, Any]:
        try:
            manifest = self._read_run_json("00_清单.json")
            documents = _documents_from_manifest(manifest)
            product = self._read_run_json("01_创建产品.json")
            product_id = str((product["response"].get("data") or {})["id"])
            product_name = str(product["request"]["productName"])
            index_path = self.run_dir / "03_任务索引.json"
            records = (
                json.loads(index_path.read_text(encoding="utf-8"))
                if index_path.is_file()
                else []
            )
            self._log("刷新 token 并恢复逐文件任务")
            self.platform.refresh_token()
            records = self._submit_missing_tasks(
                documents=documents,
                product_id=product_id,
                product_name=product_name,
                records=records,
            )
            return self._wait_and_finalize(
                documents=documents,
                product_id=product_id,
                records=records,
            )
        except Exception as exc:
            self._record_failure(exc)
            raise

    def recover_sequentially(self) -> dict[str, Any]:
        """Harvest completed reports, then rerun unresolved PDFs one at a time."""
        try:
            manifest = self._read_run_json("00_清单.json")
            documents = _documents_from_manifest(manifest)
            product = self._read_run_json("01_创建产品.json")
            product_id = str((product["response"].get("data") or {})["id"])
            product_name = str(product["request"]["productName"])
            records = self._read_run_json("03_任务索引.json")
            if len(records) != len(documents):
                raise RuntimeError(
                    f"Expected {len(documents)} task records, found {len(records)}"
                )

            self._log("刷新 token 并收集已有有效报告")
            self.platform.refresh_token()
            for record in records:
                if self._saved_report_is_valid(record):
                    continue
                report = self._try_get_valid_report(record, str(record["task_id"]))
                if report is not None:
                    self._save_valid_report(record, report, str(record["task_id"]))
            self._write_run_json("03_任务索引.json", records)

            total = len(records)
            for position, record in enumerate(records, start=1):
                if self._saved_report_is_valid(record):
                    self._log(f"已有有效报告 {position}/{total}: {record['file_name']}")
                    continue

                # A previously queued task may have completed while earlier files ran.
                original_report = self._try_get_valid_report(
                    record, str(record["task_id"])
                )
                if original_report is not None:
                    self._save_valid_report(
                        record, original_report, str(record["task_id"])
                    )
                    self._write_run_json("03_任务索引.json", records)
                    continue

                self._log(f"顺序恢复 {position}/{total}: {record['file_name']}")
                report, successful_task_id = self._run_sequential_attempts(
                    record=record,
                    product_id=product_id,
                    product_name=product_name,
                    position=position,
                    total=total,
                )
                self._save_valid_report(record, report, successful_task_id)
                self._write_run_json("03_任务索引.json", records)

            return self._finalize_saved_reports(
                documents=documents,
                product_id=product_id,
                records=records,
            )
        except Exception as exc:
            self._record_failure(exc)
            raise

    def recover_with_isolated_products(self) -> dict[str, Any]:
        """Recover unresolved PDFs with a fresh product for each file."""
        try:
            manifest = self._read_run_json("00_清单.json")
            documents = _documents_from_manifest(manifest)
            original_product = self._read_run_json("01_创建产品.json")
            original_product_id = str(
                (original_product["response"].get("data") or {})["id"]
            )
            records = self._read_run_json("03_任务索引.json")
            if len(records) != len(documents):
                raise RuntimeError(
                    f"Expected {len(documents)} task records, found {len(records)}"
                )

            self._log("刷新 token 并检查可复用报告")
            self.platform.refresh_token()
            for record in records:
                if self._saved_report_is_valid(record):
                    continue
                report = self._try_get_valid_report(record, str(record["task_id"]))
                if report is not None:
                    self._save_valid_report(
                        record,
                        report,
                        str(record["task_id"]),
                        original_product_id,
                    )
            self._write_run_json("03_任务索引.json", records)

            documents_by_name = {
                _normalized_name(document.file_name): document
                for document in documents
            }
            total = len(records)
            for position, record in enumerate(records, start=1):
                if self._saved_report_is_valid(record):
                    self._log(f"已有有效报告 {position}/{total}: {record['file_name']}")
                    continue

                original_report = self._try_get_valid_report(
                    record, str(record["task_id"])
                )
                if original_report is not None:
                    self._save_valid_report(
                        record,
                        original_report,
                        str(record["task_id"]),
                        original_product_id,
                    )
                    self._write_run_json("03_任务索引.json", records)
                    continue

                known_report = self._harvest_isolated_attempt(record)
                if known_report is not None:
                    report, task_id, product_id = known_report
                    self._save_valid_report(record, report, task_id, product_id)
                    self._write_run_json("03_任务索引.json", records)
                    continue

                active_report = self._resume_active_isolated_attempt(
                    record=record,
                    position=position,
                    total=total,
                )
                if active_report is not None:
                    report, task_id, product_id = active_report
                    self._save_valid_report(record, report, task_id, product_id)
                    self._write_run_json("03_任务索引.json", records)
                    continue

                document = documents_by_name[_normalized_name(str(record["file_name"]))]
                self._log(f"独立产品恢复 {position}/{total}: {record['file_name']}")
                report, task_id, product_id = self._run_isolated_attempts(
                    record=record,
                    document=document,
                    position=position,
                    total=total,
                )
                self._save_valid_report(record, report, task_id, product_id)
                self._write_run_json("03_任务索引.json", records)

            return self._finalize_saved_reports(
                documents=documents,
                product_id=original_product_id,
                records=records,
            )
        except Exception as exc:
            self._record_failure(exc)
            raise

    def write_partial_outputs(self) -> dict[str, Any]:
        manifest = self._read_run_json("00_清单.json")
        documents = _documents_from_manifest(manifest)
        original_product = self._read_run_json("01_创建产品.json")
        original_product_id = str(
            (original_product["response"].get("data") or {})["id"]
        )
        records = self._read_run_json("03_任务索引.json")
        combined_rows: list[dict[str, Any]] = []
        review_item_ids: set[str] = set()
        task_ids_by_file: dict[str, str] = {}
        product_ids_by_file: dict[str, str] = {}
        for record in records:
            report_path = Path(record["task_dir"]) / "05_AI审核报告.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = [row for row in report.get("data") or [] if isinstance(row, dict)]
            try:
                self._validate_individual_report(record, rows)
            except RuntimeError:
                continue
            combined_rows.extend(rows)
            review_item_ids.update(str(item) for item in record["review_item_ids"])
            normalized = _normalized_name(str(record["file_name"]))
            task_ids_by_file[normalized] = str(
                record.get("successful_task_id") or record["task_id"]
            )
            product_ids_by_file[normalized] = str(
                record.get("successful_product_id") or original_product_id
            )
        partial_report = {
            "success": False,
            "mode": "partial_valid_reports",
            "total": len(combined_rows),
            "data": combined_rows,
        }
        raw_report_path = self._write_run_json(
            "05_部分有效报告.json", partial_report
        )
        return write_structured_outputs(
            report=partial_report,
            documents=documents,
            output_dir=self.output_dir,
            run_id=self.run_dir.name,
            product_id=original_product_id,
            task_id="pending",
            review_item_ids=sorted(review_item_ids, key=_review_item_sort_key),
            raw_report_path=raw_report_path,
            task_ids_by_file=task_ids_by_file,
            product_ids_by_file=product_ids_by_file,
        )

    def _harvest_isolated_attempt(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any], str, str] | None:
        for attempt in reversed(record.get("isolated_attempts") or []):
            task_id = str(attempt.get("task_id") or "")
            product_id = str(attempt.get("product_id") or "")
            if not task_id or not product_id:
                continue
            report = self._try_get_valid_report(record, task_id)
            if report is not None:
                attempt["status"] = "completed"
                return report, task_id, product_id
        return None

    def _resume_active_isolated_attempt(
        self,
        *,
        record: dict[str, Any],
        position: int,
        total: int,
    ) -> tuple[dict[str, Any], str, str] | None:
        attempts = record.get("isolated_attempts") or []
        if not attempts:
            return None
        attempt = attempts[-1]
        task_id = str(attempt.get("task_id") or "")
        product_id = str(attempt.get("product_id") or "")
        if not task_id or not product_id or attempt.get("status") != "started":
            return None
        rows = self.platform.get_task_progress([task_id])
        if not rows:
            attempt["status"] = "missing"
            return None
        status = str(rows[0].get("status") or "").upper()
        if status in TERMINAL_FAILURE_STATUSES:
            attempt["status"] = "failed"
            return None
        startup_started = time.monotonic()
        try:
            progress = self.platform.wait_for_task(
                task_id,
                self.settings.poll_interval_seconds,
                self.settings.poll_timeout_seconds,
                on_progress=lambda snapshot: self._guarded_isolated_progress(
                    position,
                    total,
                    record,
                    int(attempt.get("attempt") or 0),
                    snapshot,
                    startup_started,
                ),
            )
        except (PlatformError, TimeoutError) as exc:
            attempt["status"] = "failed"
            attempt["error"] = str(exc)
            self._delete_stalled_task(task_id)
            return None
        attempt_dir = Path(record["task_dir"]) / "isolated" / (
            f"{int(attempt.get('attempt') or 0):02d}"
        )
        _write_json(attempt_dir / "06_最终进度.json", progress)
        report = self.platform.get_results(task_id)
        _write_json(attempt_dir / "07_AI审核报告.json", report)
        report_rows = [
            row for row in report.get("data") or [] if isinstance(row, dict)
        ]
        self._validate_individual_report(record, report_rows)
        attempt["status"] = "completed"
        return report, task_id, product_id

    def _run_isolated_attempts(
        self,
        *,
        record: dict[str, Any],
        document: SourceDocument,
        position: int,
        total: int,
    ) -> tuple[dict[str, Any], str, str]:
        task_dir = Path(record["task_dir"])
        zip_path = task_dir / "待审文件.zip"
        attempts = list(record.get("isolated_attempts") or [])
        for attempt_number in range(len(attempts) + 1, 11):
            attempt_dir = task_dir / "isolated" / f"{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_record: dict[str, Any] | None = None
            task_id = ""
            product_id = ""
            product_name = (
                f"招标文件独立审核_{position:02d}_"
                f"{datetime.now():%Y%m%d%H%M%S}"
            )
            try:
                product = self.platform.create_product(
                    product_name, self.settings.product_property
                )
                product_id = str((product["response"].get("data") or {})["id"])
                _write_json(attempt_dir / "01_创建产品.json", product)
                imported = self.platform.import_review_points(
                    product_id, self.settings.excel_path
                )
                _write_json(attempt_dir / "02_导入审核要点_响应.json", imported)
                upload = self.platform.upload_zip(product_id, zip_path)
                _write_json(attempt_dir / "03_上传ZIP_响应.json", upload)
                self._validate_upload(upload, [document])
                record["review_item_ids"] = _review_item_ids(upload)
                task = self.platform.create_task(
                    product_id=product_id,
                    product_name=product_name,
                    applicant_name=self.settings.applicant_name,
                    product_property=self.settings.product_property,
                    clinical_evaluation=self.settings.clinical_evaluation,
                    upload=upload,
                )
                task_id = str(task["task_id"])
                _write_json(attempt_dir / "04_创建任务.json", task)
                started = self.platform.start_task(task_id)
                _write_json(attempt_dir / "05_启动任务_响应.json", started)
                attempt_record: dict[str, Any] = {
                    "attempt": attempt_number,
                    "product_id": product_id,
                    "task_id": task_id,
                    "status": "started",
                }
                attempts.append(attempt_record)
                record["isolated_attempts"] = attempts
                self._write_run_json(
                    "03_任务索引.json", self._read_task_index_with(record)
                )
                startup_started = time.monotonic()
                progress = self.platform.wait_for_task(
                    task_id,
                    self.settings.poll_interval_seconds,
                    self.settings.poll_timeout_seconds,
                    on_progress=lambda snapshot: self._guarded_isolated_progress(
                        position,
                        total,
                        record,
                        attempt_number,
                        snapshot,
                        startup_started,
                    ),
                )
                _write_json(attempt_dir / "06_最终进度.json", progress)
                report = self.platform.get_results(task_id)
                _write_json(attempt_dir / "07_AI审核报告.json", report)
                rows = [
                    row
                    for row in report.get("data") or []
                    if isinstance(row, dict)
                ]
                self._validate_individual_report(record, rows)
                attempt_record["status"] = "completed"
                return report, task_id, product_id
            except (PlatformError, RuntimeError, TimeoutError) as exc:
                if task_id:
                    self._delete_stalled_task(task_id)
                if attempt_record is None:
                    attempt_record = {
                        "attempt": attempt_number,
                        "status": "failed",
                    }
                    attempts.append(attempt_record)
                    record["isolated_attempts"] = attempts
                attempt_record["status"] = "failed"
                attempt_record["error"] = str(exc)
                self._write_run_json(
                    "03_任务索引.json", self._read_task_index_with(record)
                )
                _write_json(
                    attempt_dir / "99_失败.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
                self._log(
                    f"独立任务第 {attempt_number}/10 次失败，将更换产品重试"
                )
        raise RuntimeError(
            f"Isolated recovery failed ten times: {record['file_name']}"
        )

    def _run_sequential_attempts(
        self,
        *,
        record: dict[str, Any],
        product_id: str,
        product_name: str,
        position: int,
        total: int,
    ) -> tuple[dict[str, Any], str]:
        task_dir = Path(record["task_dir"])
        upload = json.loads(
            (task_dir / "01_上传ZIP_响应.json").read_text(encoding="utf-8")
        )
        attempts = list(record.get("sequential_attempts") or [])
        for attempt_number in range(len(attempts) + 1, 4):
            attempt_dir = task_dir / "recovery" / f"{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            task = self.platform.create_task(
                product_id=product_id,
                product_name=product_name,
                applicant_name=self.settings.applicant_name,
                product_property=self.settings.product_property,
                clinical_evaluation=self.settings.clinical_evaluation,
                upload=upload,
            )
            task_id = str(task["task_id"])
            _write_json(attempt_dir / "01_创建任务.json", task)
            started = self.platform.start_task(task_id)
            _write_json(attempt_dir / "02_启动任务_响应.json", started)
            attempt_record: dict[str, Any] = {
                "attempt": attempt_number,
                "task_id": task_id,
                "status": "started",
            }
            attempts.append(attempt_record)
            record["sequential_attempts"] = attempts
            self._write_run_json("03_任务索引.json", self._read_task_index_with(record))
            try:
                progress = self.platform.wait_for_task(
                    task_id,
                    self.settings.poll_interval_seconds,
                    self.settings.poll_timeout_seconds,
                    on_progress=lambda snapshot: self._sequential_progress(
                        position, total, record, attempt_number, snapshot
                    ),
                )
                _write_json(attempt_dir / "03_最终进度.json", progress)
                report = self.platform.get_results(task_id)
                _write_json(attempt_dir / "04_AI审核报告.json", report)
                rows = [
                    row
                    for row in report.get("data") or []
                    if isinstance(row, dict)
                ]
                self._validate_individual_report(record, rows)
                attempt_record["status"] = "completed"
                return report, task_id
            except (PlatformError, RuntimeError, TimeoutError) as exc:
                attempt_record["status"] = "failed"
                attempt_record["error"] = str(exc)
                _write_json(
                    attempt_dir / "99_失败.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
                self._log(
                    f"任务 {task_id} 第 {attempt_number}/3 次失败，将重试"
                )
        raise RuntimeError(
            f"Sequential recovery failed three times: {record['file_name']}"
        )

    def _try_get_valid_report(
        self, record: dict[str, Any], task_id: str
    ) -> dict[str, Any] | None:
        try:
            progress_rows = self.platform.get_task_progress([task_id])
            if not progress_rows:
                return None
            progress = progress_rows[0]
            status = str(progress.get("status") or "").upper()
            percentage = float(progress.get("percentage") or 0)
            if status != "COMPLETED" and percentage < 100:
                return None
            report = self.platform.get_results(task_id)
            rows = [row for row in report.get("data") or [] if isinstance(row, dict)]
            self._validate_individual_report(record, rows)
            return report
        except (PlatformError, RuntimeError):
            return None

    def _saved_report_is_valid(self, record: dict[str, Any]) -> bool:
        report_path = Path(record["task_dir"]) / "05_AI审核报告.json"
        if not report_path.is_file():
            return False
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = [row for row in report.get("data") or [] if isinstance(row, dict)]
            self._validate_individual_report(record, rows)
            return True
        except (json.JSONDecodeError, RuntimeError):
            return False

    @staticmethod
    def _save_valid_report(
        record: dict[str, Any],
        report: dict[str, Any],
        task_id: str,
        product_id: str | None = None,
    ) -> None:
        _write_json(Path(record["task_dir"]) / "05_AI审核报告.json", report)
        record["successful_task_id"] = task_id
        if product_id:
            record["successful_product_id"] = product_id

    def _finalize_saved_reports(
        self,
        *,
        documents: list[SourceDocument],
        product_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        combined_rows: list[dict[str, Any]] = []
        all_review_item_ids: set[str] = set()
        task_ids_by_file: dict[str, str] = {}
        product_ids_by_file: dict[str, str] = {}
        for record in records:
            report_path = Path(record["task_dir"]) / "05_AI审核报告.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = [row for row in report.get("data") or [] if isinstance(row, dict)]
            self._validate_individual_report(record, rows)
            combined_rows.extend(rows)
            all_review_item_ids.update(str(item) for item in record["review_item_ids"])
            task_ids_by_file[_normalized_name(str(record["file_name"]))] = str(
                record.get("successful_task_id") or record["task_id"]
            )
            product_ids_by_file[_normalized_name(str(record["file_name"]))] = str(
                record.get("successful_product_id") or product_id
            )
        combined_report = {
            "success": True,
            "mode": "one_task_per_pdf_recovery",
            "total": len(combined_rows),
            "data": combined_rows,
        }
        raw_report_path = self._write_run_json(
            "05_合并AI审核报告.json", combined_report
        )
        summary = write_structured_outputs(
            report=combined_report,
            documents=documents,
            output_dir=self.output_dir,
            run_id=self.run_dir.name,
            product_id=product_id,
            task_id="multiple",
            review_item_ids=sorted(all_review_item_ids, key=_review_item_sort_key),
            raw_report_path=raw_report_path,
            task_ids_by_file=task_ids_by_file,
            product_ids_by_file=product_ids_by_file,
        )
        self._write_run_json("06_结构化汇总.json", summary)
        return summary

    def _sequential_progress(
        self,
        position: int,
        total: int,
        record: dict[str, Any],
        attempt: int,
        snapshot: dict[str, Any],
    ) -> None:
        payload = {
            "file_position": position,
            "file_count": total,
            "file_name": record["file_name"],
            "attempt": attempt,
            "task": snapshot,
        }
        self._write_run_json("04_顺序恢复进度.json", payload)
        self._log(
            f"顺序进度 {position}/{total}: "
            f"{snapshot.get('percentage', 0)}% {snapshot.get('status', '')}"
        )

    def _isolated_progress(
        self,
        position: int,
        total: int,
        record: dict[str, Any],
        attempt: int,
        snapshot: dict[str, Any],
    ) -> None:
        payload = {
            "file_position": position,
            "file_count": total,
            "file_name": record["file_name"],
            "attempt": attempt,
            "task": snapshot,
        }
        self._write_run_json("04_独立恢复进度.json", payload)
        self._log(
            f"独立进度 {position}/{total}: "
            f"{snapshot.get('percentage', 0)}% {snapshot.get('status', '')}"
        )

    def _guarded_isolated_progress(
        self,
        position: int,
        total: int,
        record: dict[str, Any],
        attempt: int,
        snapshot: dict[str, Any],
        startup_started: float,
    ) -> None:
        self._isolated_progress(position, total, record, attempt, snapshot)
        percentage = float(snapshot.get("percentage") or 0)
        task_id = str(snapshot.get("taskId") or "")
        now = time.monotonic()
        watches = getattr(self, "_isolated_progress_watches", {})
        self._isolated_progress_watches = watches
        watch = watches.get(task_id)
        if watch is None or percentage > float(watch["percentage"]):
            watches[task_id] = {"percentage": percentage, "changed_at": now}
            watch = watches[task_id]
        stalled_for = now - float(watch["changed_at"])
        if percentage <= 0:
            stalled_for = max(stalled_for, now - startup_started)
        if stalled_for >= 900:
            raise TimeoutError(
                f"Task {task_id} stayed at {percentage}% for 15 minutes"
            )

    def _delete_stalled_task(self, task_id: str) -> None:
        try:
            self.platform.delete_task(task_id)
            self._log(f"已删除卡死任务 {task_id}")
        except (PlatformError, requests.RequestException) as exc:
            self._log(f"删除卡死任务 {task_id} 失败: {exc}")

    def _read_task_index_with(self, updated: dict[str, Any]) -> list[dict[str, Any]]:
        records = self._read_run_json("03_任务索引.json")
        target = _normalized_name(str(updated["file_name"]))
        return [
            updated
            if _normalized_name(str(record["file_name"])) == target
            else record
            for record in records
        ]

    def _submit_missing_tasks(
        self,
        *,
        documents: list[SourceDocument],
        product_id: str,
        product_name: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        submitted = {
            _normalized_name(str(record["file_name"])) for record in records
        }
        tasks_root = self.run_dir / "tasks"
        tasks_root.mkdir(parents=True, exist_ok=True)

        for index, document in enumerate(documents, start=1):
            if _normalized_name(document.file_name) in submitted:
                continue
            self._log(f"提交 {index}/{len(documents)}: {document.file_name}")
            task_dir = tasks_root / f"{index:03d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            zip_path = build_batch_zip([document], task_dir / "待审文件.zip")
            upload = self.platform.upload_zip(product_id, zip_path)
            _write_json(task_dir / "01_上传ZIP_响应.json", upload)
            self._validate_upload(upload, [document])
            review_item_ids = _review_item_ids(upload)
            task = self.platform.create_task(
                product_id=product_id,
                product_name=product_name,
                applicant_name=self.settings.applicant_name,
                product_property=self.settings.product_property,
                clinical_evaluation=self.settings.clinical_evaluation,
                upload=upload,
            )
            _write_json(task_dir / "02_创建任务.json", task)
            started = self.platform.start_task(task["task_id"])
            _write_json(task_dir / "03_启动任务_响应.json", started)
            record = {
                "index": index,
                "file_name": document.file_name,
                "relative_path": document.relative_path.as_posix(),
                "task_id": str(task["task_id"]),
                "task_dir": str(task_dir),
                "review_item_ids": review_item_ids,
            }
            records.append(record)
            submitted.add(_normalized_name(document.file_name))
            self._write_run_json("03_任务索引.json", records)
        return records

    def _wait_and_finalize(
        self,
        *,
        documents: list[SourceDocument],
        product_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(records) != len(documents):
            raise RuntimeError(
                f"Expected {len(documents)} submitted tasks, found {len(records)}"
            )
        task_ids = [str(record["task_id"]) for record in records]
        progress = self._wait_for_tasks(task_ids)
        self._write_run_json("04_最终进度.json", progress)

        combined_rows: list[dict[str, Any]] = []
        all_review_item_ids: set[str] = set()
        task_ids_by_file: dict[str, str] = {}
        for record in records:
            task_id = str(record["task_id"])
            task_dir = Path(record["task_dir"])
            self._log(f"下载报告: {record['file_name']}")
            report = self.platform.get_results(task_id)
            _write_json(task_dir / "05_AI审核报告.json", report)
            rows = [row for row in report.get("data") or [] if isinstance(row, dict)]
            self._validate_individual_report(record, rows)
            combined_rows.extend(rows)
            all_review_item_ids.update(str(item) for item in record["review_item_ids"])
            task_ids_by_file[_normalized_name(str(record["file_name"]))] = task_id

        combined_report = {
            "success": True,
            "mode": "one_task_per_pdf",
            "total": len(combined_rows),
            "data": combined_rows,
        }
        raw_report_path = self._write_run_json(
            "05_合并AI审核报告.json", combined_report
        )
        summary = write_structured_outputs(
            report=combined_report,
            documents=documents,
            output_dir=self.output_dir,
            run_id=self.run_dir.name,
            product_id=product_id,
            task_id="multiple",
            review_item_ids=sorted(all_review_item_ids, key=_review_item_sort_key),
            raw_report_path=raw_report_path,
            task_ids_by_file=task_ids_by_file,
        )
        self._write_run_json("06_结构化汇总.json", summary)
        return summary

    def _wait_for_tasks(self, task_ids: list[str]) -> dict[str, dict[str, Any]]:
        deadline = time.monotonic() + self.timeout_seconds
        last: dict[str, dict[str, Any]] = {}
        retry_path = self.run_dir / "04_任务重试.json"
        retry_counts: dict[str, int] = (
            json.loads(retry_path.read_text(encoding="utf-8"))
            if retry_path.is_file()
            else {}
        )
        while time.monotonic() < deadline:
            rows = self.platform.get_task_progress(task_ids)
            for row in rows:
                task_id = str(row.get("taskId") or row.get("id") or "")
                if task_id:
                    last[task_id] = row
            failed = {
                task_id
                for task_id, row in last.items()
                if str(row.get("status") or "").upper()
                in TERMINAL_FAILURE_STATUSES
            }
            completed = {
                task_id
                for task_id, row in last.items()
                if str(row.get("status") or "").upper() == "COMPLETED"
                or float(row.get("percentage") or 0) >= 100
            }
            completed -= failed
            active = set(task_ids) - completed - failed
            percentages = [float(row.get("percentage") or 0) for row in last.values()]
            snapshot = {
                "task_count": len(task_ids),
                "completed_count": len(completed),
                "average_percentage": (
                    round(sum(percentages) / len(task_ids), 2) if percentages else 0
                ),
                "tasks": last,
            }
            self._write_run_json("04_当前进度.json", snapshot)
            self._log(
                f"总体进度: {snapshot['average_percentage']}% "
                f"({len(completed)}/{len(task_ids)} 已完成)"
            )
            if len(completed) == len(task_ids):
                return last
            if failed and not active:
                exhausted = [
                    task_id
                    for task_id in failed
                    if int(retry_counts.get(task_id, 0)) >= 3
                ]
                if exhausted:
                    raise RuntimeError(
                        "Platform tasks failed after retries: "
                        + ", ".join(sorted(exhausted))
                    )
                for task_id in sorted(failed):
                    self._log(
                        f"重新启动失败任务 {task_id} "
                        f"({int(retry_counts.get(task_id, 0)) + 1}/3)"
                    )
                    self.platform.start_task(task_id)
                    retry_counts[task_id] = int(retry_counts.get(task_id, 0)) + 1
                _write_json(retry_path, retry_counts)
            time.sleep(self.settings.poll_interval_seconds)
        raise TimeoutError(
            f"Individual tasks did not complete before timeout; last={last}"
        )

    @staticmethod
    def _validate_individual_report(
        record: dict[str, Any], rows: list[dict[str, Any]]
    ) -> None:
        target_name = _normalized_name(str(record["file_name"]))
        matching = [
            row
            for row in rows
            if _normalized_name(
                str(
                    repair_mojibake(
                        row.get("fileName") or row.get("reviewFileName") or ""
                    )
                )
            )
            == target_name
        ]
        review_items = {
            str(row.get("reviewItem"))
            for row in matching
            if row.get("reviewItem") not in (None, "")
        }
        expected_items = {str(item) for item in record["review_item_ids"]}
        missing_items = expected_items - review_items
        if missing_items:
            raise RuntimeError(
                f"Report for {record['file_name']} is missing review items: "
                + ", ".join(sorted(missing_items, key=_review_item_sort_key))
            )
        missing_core_markers = (
            "核心文件缺失",
            "核心文件未提供",
            "未提供有效的核心文件",
        )
        if matching and all(
            any(
                marker in str(row.get("errorReason") or "")
                for marker in missing_core_markers
            )
            for row in matching
        ):
            raise RuntimeError(
                f"Platform failed to parse the core file: {record['file_name']}"
            )

    def _record_failure(self, exc: Exception) -> None:
        self._write_run_json(
            "99_运行失败.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )


def _documents_from_manifest(manifest: dict[str, Any]) -> list[SourceDocument]:
    return [
        SourceDocument(
            path=Path(item["path"]).resolve(),
            relative_path=Path(item["relative_path"]),
            expected_issue=str(item["expected_issue"]),
            archive_directory=str(item["archive_directory"]),
        )
        for item in manifest.get("documents") or []
    ]


def _normalize_opinion(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("reviewStatus") or "").upper()
    if status == "SUCCESS":
        compliant: bool | None = True
    elif status in ISSUE_STATUSES:
        compliant = False
    else:
        compliant = None

    content_result = _parse_json_value(row.get("contentResult"))
    reviewed_pages = (
        content_result.get("page_ids") if isinstance(content_result, dict) else None
    )
    return repair_mojibake(
        {
            "review_item": row.get("reviewItem"),
            "review_point_id": row.get("reviewPointId"),
            "status": status,
            "compliant": compliant,
            "opinion": row.get("errorReason"),
            "evidence": {
                "page_number": row.get("filePageNumber"),
                "material_excerpt": row.get("materialExcerpt"),
            },
            "reasoning": row.get("reasoning"),
            "review_point": row.get("point"),
            "reviewed_pages": reviewed_pages,
            "result_details": {
                "file_result": _parse_json_value(row.get("fileResult")),
                "tool_result": _parse_json_value(row.get("toolResult")),
                "case_result": _parse_json_value(row.get("caseResult")),
            },
            "platform_ids": {
                "result_id": row.get("id"),
                "task_review_file_relation_id": row.get(
                    "taskReviewFileRelationId"
                ),
            },
        }
    )


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _review_item_ids(upload: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(item.get("reviewItem"))
            for item in upload.get("reviewPoints") or []
            if isinstance(item, dict) and item.get("reviewItem") not in (None, "")
        },
        key=_review_item_sort_key,
    )


def _opinion_sort_key(opinion: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _review_item_sort_key(str(opinion.get("review_item") or "")),
        str(opinion.get("status") or ""),
        int((opinion.get("evidence") or {}).get("page_number") or 0),
    )


def _review_item_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _normalized_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
