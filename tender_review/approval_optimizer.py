from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .llm import LlmClient
from .platform import PlatformClient
from .results import parse_workflow_output, repair_mojibake
from .rules import ReviewRule, load_review_rules, write_optimized_excel_updates


TERMINAL_GROUP_STATUSES = {"optimized", "already_covered", "no_target"}


@dataclass(frozen=True)
class ApprovalCase:
    case_id: str
    source_file: str
    source_relative_path: str
    source_json: Path
    pdf_path: Path
    source_status: str
    task_id: str
    product_id: str
    review_item: str
    review_point_id: str
    expected_compliant: bool
    expected_opinions: tuple[str, ...]
    evidence: tuple[Any, ...]
    reasoning: tuple[str, ...]
    reported_review_point: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_json"] = str(self.source_json)
        result["pdf_path"] = str(self.pdf_path)
        return result


@dataclass
class ApprovalDataset:
    cases: list[ApprovalCase]
    files: list[dict[str, Any]]
    skipped_files: list[dict[str, Any]]
    invalid_opinions: list[dict[str, Any]]


@dataclass(frozen=True)
class WorkflowContext:
    task_id: str
    review_point_id: str
    workflow_id: str
    content: Any
    review_item_name: str
    upload_path: Path

    def payload(self, point: str) -> dict[str, Any]:
        payload = {
            "id": self.review_point_id,
            "workflowId": self.workflow_id,
            "content": self.content,
            "reviewItemName": self.review_item_name,
            "point": point,
            "taskId": self.task_id,
        }
        missing = [key for key, value in payload.items() if value in (None, "")]
        if missing:
            raise ValueError(
                "Workflow-test payload is missing: " + ", ".join(missing)
            )
        return payload


def load_approval_dataset(opinion_dir: Path, source_dir: Path) -> ApprovalDataset:
    opinion_dir = opinion_dir.resolve()
    source_dir = source_dir.resolve()
    if not opinion_dir.is_dir():
        raise FileNotFoundError(f"Approval opinion directory not found: {opinion_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"PDF source directory not found: {source_dir}")

    pdfs: dict[str, Path] = {}
    for pdf in source_dir.rglob("*"):
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            continue
        key = _normalized_name(pdf.name)
        if key in pdfs:
            raise ValueError(
                f"Duplicate PDF basename: {pdfs[key]} and {pdf.resolve()}"
            )
        pdfs[key] = pdf.resolve()

    cases: list[ApprovalCase] = []
    files: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []
    invalid_opinions: list[dict[str, Any]] = []

    json_files = sorted(
        (
            path
            for path in opinion_dir.rglob("*.json")
            if path.name != "汇总.json"
        ),
        key=lambda path: path.relative_to(opinion_dir).as_posix(),
    )
    for path in json_files:
        relative_json = path.relative_to(opinion_dir).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped_files.append(
                {"source_json": relative_json, "reason": f"JSON读取失败: {exc}"}
            )
            continue

        source = payload.get("source") or {}
        review = payload.get("review") or {}
        source_file = str(source.get("file_name") or "").strip()
        status = str(review.get("status") or "missing").strip().lower()
        opinions = [row for row in review.get("opinions") or [] if isinstance(row, dict)]
        file_record = {
            "source_json": relative_json,
            "source_file": source_file,
            "source_relative_path": str(source.get("relative_path") or ""),
            "status": status,
            "task_id": str(review.get("task_id") or ""),
            "product_id": str(review.get("product_id") or ""),
            "opinion_count": len(opinions),
            "missing_review_items": review.get("missing_review_items") or [],
        }
        files.append(file_record)

        if status not in {"completed", "partial"} or not opinions:
            skipped_files.append(
                {
                    **file_record,
                    "reason": "审批意见尚未完成或没有可用意见",
                }
            )
            continue
        pdf_path = pdfs.get(_normalized_name(source_file))
        if pdf_path is None:
            skipped_files.append(
                {**file_record, "reason": "找不到同名PDF文件"}
            )
            continue

        rows_by_item: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, row in enumerate(opinions, start=1):
            review_item = str(row.get("review_item") or "").strip()
            opinion = str(repair_mojibake(row.get("opinion") or "")).strip()
            compliant = row.get("compliant")
            if not review_item or not opinion or not isinstance(compliant, bool):
                invalid_opinions.append(
                    {
                        "source_json": relative_json,
                        "opinion_index": index,
                        "review_item": review_item,
                        "reason": "缺少review_item/opinion，或compliant不是布尔值",
                    }
                )
                continue
            rows_by_item.setdefault(review_item, []).append((index, row))

        for review_item, indexed_rows in rows_by_item.items():
            rows = [row for _, row in indexed_rows]
            target_rows = [row for row in rows if row.get("compliant") is False]
            expected_compliant = not target_rows
            expectation_rows = target_rows or rows
            expected_opinions = tuple(
                _dedupe(
                    str(repair_mojibake(row.get("opinion") or "")).strip()
                    for row in expectation_rows
                )
            )
            review_point_ids = _dedupe(
                str(row.get("review_point_id") or "").strip() for row in rows
            )
            if len(review_point_ids) != 1 or not review_point_ids[0]:
                invalid_opinions.append(
                    {
                        "source_json": relative_json,
                        "review_item": review_item,
                        "reason": "同一文件审核项的review_point_id缺失或不一致",
                        "values": review_point_ids,
                    }
                )
                continue
            case_id = _case_id(source_file, review_item)
            cases.append(
                ApprovalCase(
                    case_id=case_id,
                    source_file=source_file,
                    source_relative_path=str(source.get("relative_path") or ""),
                    source_json=path.resolve(),
                    pdf_path=pdf_path,
                    source_status=status,
                    task_id=str(review.get("task_id") or "").strip(),
                    product_id=str(review.get("product_id") or "").strip(),
                    review_item=review_item,
                    review_point_id=review_point_ids[0],
                    expected_compliant=expected_compliant,
                    expected_opinions=expected_opinions,
                    evidence=tuple(row.get("evidence") for row in expectation_rows),
                    reasoning=tuple(
                        str(repair_mojibake(row.get("reasoning") or "")).strip()
                        for row in expectation_rows
                    ),
                    reported_review_point=next(
                        (
                            str(row.get("review_point") or "").strip()
                            for row in rows
                            if str(row.get("review_point") or "").strip()
                        ),
                        "",
                    ),
                )
            )

    cases.sort(key=lambda case: (_review_item_sort_key(case.review_item), case.source_file))
    return ApprovalDataset(
        cases=cases,
        files=files,
        skipped_files=skipped_files,
        invalid_opinions=invalid_opinions,
    )


def group_approval_cases(
    cases: list[ApprovalCase],
) -> dict[str, list[ApprovalCase]]:
    groups: dict[str, list[ApprovalCase]] = {}
    for case in cases:
        groups.setdefault(case.review_item, []).append(case)
    return {
        key: sorted(value, key=lambda case: case.source_file)
        for key, value in sorted(
            groups.items(), key=lambda item: _review_item_sort_key(item[0])
        )
    }


class TaskArtifactResolver:
    def __init__(self, batch_run: Path) -> None:
        self.batch_run = batch_run.resolve()
        index_path = self.batch_run / "03_任务索引.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Batch task index not found: {index_path}")
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Batch task index must be a list: {index_path}")
        self.records = {
            _normalized_name(str(row.get("file_name") or "")): row
            for row in rows
            if isinstance(row, dict) and row.get("file_name")
        }
        self._cache: dict[str, WorkflowContext] = {}

    def resolve(self, case: ApprovalCase) -> WorkflowContext:
        if case.case_id in self._cache:
            return self._cache[case.case_id]
        record = self.records.get(_normalized_name(case.source_file))
        if record is None:
            raise ValueError(f"任务索引中找不到文件: {case.source_file}")
        task_dir = Path(str(record.get("task_dir") or ""))
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        exact: list[tuple[Path, dict[str, Any]]] = []
        fallback: list[tuple[Path, dict[str, Any]]] = []
        for upload_path in sorted(task_dir.rglob("*上传ZIP_响应.json")):
            try:
                upload = json.loads(upload_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for point in upload.get("reviewPoints") or []:
                if str(point.get("reviewItem") or "").strip() != case.review_item:
                    continue
                fallback.append((upload_path, point))
                if str(point.get("id") or "").strip() == case.review_point_id:
                    exact.append((upload_path, point))

        candidates = exact or ([] if case.review_point_id else fallback)
        if len(candidates) != 1:
            raise ValueError(
                f"无法唯一定位 {case.source_file} 项目 {case.review_item} 的上传元数据; "
                f"exact={len(exact)}, fallback={len(fallback)}"
            )
        upload_path, point = candidates[0]
        task_id = case.task_id
        if not task_id or task_id.lower() == "pending":
            task_id = str(
                record.get("successful_task_id") or record.get("task_id") or ""
            ).strip()
        context = WorkflowContext(
            task_id=task_id,
            review_point_id=str(point.get("id") or "").strip(),
            workflow_id=str(point.get("workFlowId") or "").strip(),
            content=point.get("content") or "",
            review_item_name=str(point.get("file") or "").strip(),
            upload_path=upload_path.resolve(),
        )
        context.payload(point=str(point.get("point") or "test"))
        self._cache[case.case_id] = context
        return context


def summarize_group_checks(
    cases: list[ApprovalCase],
    case_results: list[dict[str, Any]],
    stability_runs: int,
) -> dict[str, Any]:
    expected_ids = {case.case_id for case in cases}
    result_map = {
        str(result.get("case_id") or ""): result for result in case_results
    }
    missing = sorted(expected_ids - set(result_map))
    passed_ids: list[str] = []
    failed_ids: list[str] = []
    for case in cases:
        checks = result_map.get(case.case_id, {}).get("checks") or []
        passed = len(checks) == stability_runs and all(
            check.get("evaluation", {}).get("passed") is True for check in checks
        )
        (passed_ids if passed else failed_ids).append(case.case_id)
    targets = [case.case_id for case in cases if not case.expected_compliant]
    protections = [case.case_id for case in cases if case.expected_compliant]
    return {
        "all_passed": not missing and not failed_ids,
        "required_stability_runs": stability_runs,
        "case_count": len(cases),
        "target_count": len(targets),
        "protection_count": len(protections),
        "passed_count": len(passed_ids),
        "passed_case_ids": passed_ids,
        "failed_case_ids": failed_ids,
        "missing_case_ids": missing,
        "target_cases_passed": all(case_id in passed_ids for case_id in targets),
        "protection_cases_passed": all(
            case_id in passed_ids for case_id in protections
        ),
    }


class ApprovalOptimizer:
    def __init__(
        self,
        *,
        settings: Settings,
        opinion_dir: Path,
        source_dir: Path,
        batch_run: Path,
        max_iterations: int = 3,
        stability_runs: int = 1,
        run_dir: Path | None = None,
        platform: PlatformClient | None = None,
        llm: LlmClient | None = None,
    ) -> None:
        self.settings = settings
        self.opinion_dir = opinion_dir.resolve()
        self.source_dir = source_dir.resolve()
        self.batch_run = batch_run.resolve()
        self.max_iterations = max(0, max_iterations)
        self.stability_runs = max(1, stability_runs)
        self.run_dir = (
            run_dir.resolve()
            if run_dir is not None
            else settings.runs_dir
            / f"approval_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=run_dir is not None)
        self.items_dir = self.run_dir / "items"
        self.items_dir.mkdir(exist_ok=True)
        self.platform = platform or PlatformClient(
            settings.base_url, settings.username
        )
        self.llm = llm or LlmClient(
            settings.llm_url, settings.llm_api_key, settings.llm_model
        )
        self.resolver = TaskArtifactResolver(self.batch_run)
        self.dataset: ApprovalDataset | None = None
        self.rules: dict[str, ReviewRule] = {}

    def prepare(self) -> dict[str, Any]:
        dataset = load_approval_dataset(self.opinion_dir, self.source_dir)
        rules = load_review_rules(self.settings.excel_path)
        valid_cases: list[ApprovalCase] = []
        for case in dataset.cases:
            if case.review_item not in rules:
                dataset.invalid_opinions.append(
                    {
                        "source_json": str(case.source_json),
                        "review_item": case.review_item,
                        "reason": "Excel中找不到对应审核项目",
                    }
                )
                continue
            valid_cases.append(case)
        dataset.cases = valid_cases
        self.dataset = dataset
        self.rules = rules

        original_copy = self.run_dir / "审核要点_原始.xlsx"
        if not original_copy.exists():
            shutil.copy2(self.settings.excel_path, original_copy)
        groups = group_approval_cases(dataset.cases)
        manifest = {
            "schema_version": 1,
            "run_id": self.run_dir.name,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "excel_path": str(self.settings.excel_path),
            "excel_copy": str(original_copy),
            "opinion_dir": str(self.opinion_dir),
            "source_dir": str(self.source_dir),
            "batch_run": str(self.batch_run),
            "llm_model": self.settings.llm_model,
            "max_iterations": self.max_iterations,
            "stability_runs": self.stability_runs,
            "counts": {
                "approval_files": len(dataset.files),
                "usable_cases": len(dataset.cases),
                "review_item_groups": len(groups),
                "target_cases": sum(
                    1 for case in dataset.cases if not case.expected_compliant
                ),
                "protection_cases": sum(
                    1 for case in dataset.cases if case.expected_compliant
                ),
                "skipped_files": len(dataset.skipped_files),
                "invalid_opinions": len(dataset.invalid_opinions),
            },
            "files": dataset.files,
            "skipped_files": dataset.skipped_files,
            "invalid_opinions": dataset.invalid_opinions,
            "groups": {
                review_item: [case.to_dict() for case in cases]
                for review_item, cases in groups.items()
            },
        }
        _write_json(self.run_dir / "00_输入清单.json", manifest)
        return manifest

    def validate(self) -> dict[str, Any]:
        manifest = self.prepare()
        assert self.dataset is not None
        errors: list[dict[str, str]] = []
        groups = group_approval_cases(self.dataset.cases)
        for review_item, cases in groups.items():
            if not any(not case.expected_compliant for case in cases):
                continue
            for case in cases:
                try:
                    self.resolver.resolve(case)
                except Exception as exc:
                    errors.append(
                        {
                            "review_item": review_item,
                            "case_id": case.case_id,
                            "error": str(exc),
                        }
                    )
        result = {**manifest["counts"], "metadata_errors": errors}
        _write_json(self.run_dir / "01_输入校验.json", result)
        return result

    def run(self) -> dict[str, Any]:
        manifest = self.prepare()
        assert self.dataset is not None
        groups = group_approval_cases(self.dataset.cases)
        if any(
            any(not case.expected_compliant for case in cases)
            for cases in groups.values()
        ):
            self._log("刷新平台 token")
            self.platform.refresh_token()

        group_results: dict[str, dict[str, Any]] = {}
        optimized_points: dict[str, str] = {}
        version = self._latest_version_number()
        for position, (review_item, cases) in enumerate(groups.items(), start=1):
            self._log(
                f"处理审核项目 {review_item} ({position}/{len(groups)}), "
                f"目标={sum(not case.expected_compliant for case in cases)}, "
                f"保护={sum(case.expected_compliant for case in cases)}"
            )
            item_dir = self.items_dir / review_item
            item_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                item_dir / "00_输入.json",
                {
                    "review_item": review_item,
                    "rule": asdict(self.rules[review_item]),
                    "cases": [case.to_dict() for case in cases],
                },
            )
            result_path = item_dir / "结果.json"
            existing = _read_json_if_exists(result_path)
            if existing and existing.get("status") in TERMINAL_GROUP_STATUSES:
                result = existing
                newly_optimized = False
            else:
                try:
                    result = self._run_group(
                        review_item=review_item,
                        cases=cases,
                        item_dir=item_dir,
                    )
                except Exception as exc:
                    result = {
                        "review_item": review_item,
                        "status": "failed",
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _write_json(result_path, result)
                newly_optimized = result.get("status") == "optimized"

            group_results[review_item] = result
            if result.get("accepted") and result.get("accepted_point"):
                point = str(result["accepted_point"]).strip()
                if point != self.rules[review_item].point:
                    optimized_points[review_item] = point
                    if newly_optimized:
                        version += 1
                        self._write_excel_version(optimized_points, version)

        final_excel = self.run_dir / "审核要点_优化结果_最终.xlsx"
        write_optimized_excel_updates(
            self.settings.excel_path, final_excel, optimized_points
        )
        summary = self._write_result_outputs(
            manifest=manifest,
            groups=groups,
            group_results=group_results,
            optimized_points=optimized_points,
            final_excel=final_excel,
        )
        return summary

    def _run_group(
        self,
        *,
        review_item: str,
        cases: list[ApprovalCase],
        item_dir: Path,
    ) -> dict[str, Any]:
        rule = self.rules[review_item]
        targets = [case for case in cases if not case.expected_compliant]
        protections = [case for case in cases if case.expected_compliant]
        if not targets:
            return {
                "review_item": review_item,
                "status": "no_target",
                "accepted": False,
                "target_count": 0,
                "protection_count": len(protections),
                "reason": "该审核项目目前只有合规样本，不需要优化",
            }

        contexts = {case.case_id: self.resolver.resolve(case) for case in cases}
        baseline_path = item_dir / "01_基线.json"
        baseline = _read_json_if_exists(baseline_path)
        if baseline is None:
            baseline = self._evaluate_point(
                cases=targets,
                contexts=contexts,
                point=rule.point,
                label="基线",
                progress_path=item_dir / "01_基线_进度.json",
            )
            _write_json(baseline_path, baseline)
        if baseline["summary"]["all_passed"]:
            return {
                "review_item": review_item,
                "status": "already_covered",
                "accepted": True,
                "accepted_point": rule.point,
                "accepted_iteration": 0,
                "target_count": len(targets),
                "protection_count": len(protections),
                "final_evaluation": baseline,
            }

        attempt_paths = sorted(item_dir.glob("02_第*_轮.json"))
        attempts = [json.loads(path.read_text(encoding="utf-8")) for path in attempt_paths]
        current_point = (
            str(attempts[-1].get("candidate_point") or rule.point)
            if attempts
            else rule.point
        )
        latest_evaluation = attempts[-1].get("evaluation") if attempts else baseline

        for iteration in range(len(attempts) + 1, self.max_iterations + 1):
            progress_path = item_dir / f"02_第{iteration:02d}轮_进度.json"
            saved_progress = _read_json_if_exists(progress_path)
            if saved_progress:
                candidate = str(saved_progress.get("point") or "").strip()
                optimized = {
                    "optimized_point": candidate,
                    "change_note": "从已保存的候选回测断点恢复",
                    "coverage_strategy": "复用同一候选及已完成的平台输出",
                }
                self._log(
                    f"审核项目 {review_item} 第 {iteration} 轮从候选回测断点恢复"
                )
            else:
                self._log(f"审核项目 {review_item} 第 {iteration} 轮 Qwen 联合优化")
                optimized = self.llm.optimize_review_point_group(
                    review_item=review_item,
                    original_point=rule.point,
                    current_point=current_point,
                    target_cases=[
                        self._optimizer_case(case, latest_evaluation)
                        for case in targets
                    ],
                    protection_cases=[
                        self._optimizer_case(case, latest_evaluation)
                        for case in protections
                    ],
                    latest_evaluation=self._compact_evaluation(latest_evaluation),
                    previous_attempts=[
                        self._compact_attempt(item) for item in attempts
                    ],
                )
                candidate = str(optimized.get("optimized_point") or "").strip()
            if not candidate:
                raise RuntimeError("Qwen返回了空审核要点")
            leaked_names = [
                case.source_file for case in cases if case.source_file in candidate
            ]
            if leaked_names:
                raise RuntimeError("候选审核要点包含具体样本文件名，拒绝接受")

            evaluation = self._evaluate_point(
                cases=cases,
                contexts=contexts,
                point=candidate,
                label=f"优化{iteration}",
                progress_path=progress_path,
            )
            attempt = {
                "review_item": review_item,
                "iteration": iteration,
                "candidate_point": candidate,
                "change_note": optimized.get("change_note"),
                "coverage_strategy": optimized.get("coverage_strategy"),
                "optimizer_method": f"llm:{self.settings.llm_model}",
                "evaluation": evaluation,
                "accepted": evaluation["summary"]["all_passed"],
            }
            attempts.append(attempt)
            _write_json(item_dir / f"02_第{iteration:02d}轮.json", attempt)
            if attempt["accepted"]:
                return {
                    "review_item": review_item,
                    "status": "optimized",
                    "accepted": True,
                    "accepted_point": candidate,
                    "accepted_iteration": iteration,
                    "target_count": len(targets),
                    "protection_count": len(protections),
                    "change_note": optimized.get("change_note"),
                    "final_evaluation": evaluation,
                }
            current_point = candidate
            latest_evaluation = evaluation

        return {
            "review_item": review_item,
            "status": "failed",
            "accepted": False,
            "target_count": len(targets),
            "protection_count": len(protections),
            "iterations_run": len(attempts),
            "reason": "达到最大优化轮次，尚无候选通过全部目标和保护样本",
            "baseline_evaluation": baseline,
            "last_evaluation": latest_evaluation,
        }

    def _evaluate_point(
        self,
        *,
        cases: list[ApprovalCase],
        contexts: dict[str, WorkflowContext],
        point: str,
        label: str,
        progress_path: Path,
    ) -> dict[str, Any]:
        progress = _read_json_if_exists(progress_path) or {
            "label": label,
            "point": point,
            "outputs": {},
            "evaluations": {},
        }
        if progress.get("point") != point:
            raise ValueError(f"Progress point does not match current candidate: {progress_path}")

        target_cases = [case for case in cases if not case.expected_compliant]
        protection_cases = [case for case in cases if case.expected_compliant]
        for run_number in range(1, self.stability_runs + 1):
            run_key = str(run_number)
            outputs = progress.setdefault("outputs", {}).setdefault(run_key, {})
            evaluations = progress.setdefault("evaluations", {}).setdefault(
                run_key, {}
            )
            for stage_cases in (target_cases, protection_cases):
                if not stage_cases:
                    continue
                for position, case in enumerate(stage_cases, start=1):
                    if case.case_id in outputs:
                        continue
                    context = contexts[case.case_id]
                    payload = context.payload(point)
                    self._log(
                        f"{label} 项目{case.review_item} 样本{position}/{len(stage_cases)} "
                        f"稳定性{run_number}/{self.stability_runs}"
                    )
                    raw = self.platform.workflow_test(payload)
                    outputs[case.case_id] = parse_workflow_output(raw)
                    _write_json(progress_path, progress)

                pending_cases = [
                    case for case in stage_cases if case.case_id not in evaluations
                ]
                for start in range(0, len(pending_cases), 4):
                    chunk = pending_cases[start : start + 4]
                    chunk_result = self.llm.evaluate_approval_group(
                        [
                            {
                                "case_id": case.case_id,
                                "expected_compliant": case.expected_compliant,
                                "expected_opinions": list(case.expected_opinions),
                                "platform_output": _compact_platform_output(
                                    outputs[case.case_id]
                                ),
                            }
                            for case in chunk
                        ]
                    )
                    evaluations.update(chunk_result)
                    _write_json(progress_path, progress)
                if not all(
                    evaluations.get(case.case_id, {}).get("passed") is True
                    for case in stage_cases
                ):
                    break
            if not all(
                evaluations.get(case.case_id, {}).get("passed") is True
                for case in cases
            ):
                break

        case_results: list[dict[str, Any]] = []
        for case in cases:
            context = contexts[case.case_id]
            payload = context.payload(point)
            checks = []
            for run_number in range(1, self.stability_runs + 1):
                run_key = str(run_number)
                output = (progress.get("outputs") or {}).get(run_key, {}).get(
                    case.case_id
                )
                evaluation = (progress.get("evaluations") or {}).get(
                    run_key, {}
                ).get(case.case_id)
                if output is None or evaluation is None:
                    break
                checks.append(
                    {
                        "run": run_number,
                        "output": output,
                        "evaluation": evaluation,
                    }
                )
            case_results.append(
                {
                    "case_id": case.case_id,
                    "source_file": case.source_file,
                    "review_item": case.review_item,
                    "expected_compliant": case.expected_compliant,
                    "expected_opinions": list(case.expected_opinions),
                    "task_id": context.task_id,
                    "review_point_id": context.review_point_id,
                    "upload_path": str(context.upload_path),
                    "workflow_request": payload,
                    "checks": checks,
                }
            )
        return {
            "label": label,
            "point": point,
            "cases": case_results,
            "summary": summarize_group_checks(
                cases, case_results, self.stability_runs
            ),
        }

    @staticmethod
    def _optimizer_case(
        case: ApprovalCase, latest_evaluation: dict[str, Any]
    ) -> dict[str, Any]:
        latest = next(
            (
                row
                for row in latest_evaluation.get("cases") or []
                if row.get("case_id") == case.case_id
            ),
            {},
        )
        last_check = (latest.get("checks") or [{}])[-1]
        return {
            "case_id": case.case_id,
            "expected_compliant": case.expected_compliant,
            "expected_opinions": list(case.expected_opinions),
            "last_evaluation": {
                "passed": (last_check.get("evaluation") or {}).get("passed"),
                "missed_opinions": (last_check.get("evaluation") or {}).get(
                    "missed_opinions", []
                ),
                "reason": str(
                    (last_check.get("evaluation") or {}).get("reason") or ""
                )[:1000],
            },
        }

    @staticmethod
    def _compact_evaluation(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": value.get("summary") or {},
            "cases": [
                {
                    "case_id": row.get("case_id"),
                    "expected_compliant": row.get("expected_compliant"),
                    "last_evaluation": ((row.get("checks") or [{}])[-1]).get(
                        "evaluation"
                    ),
                }
                for row in value.get("cases") or []
            ],
        }

    @staticmethod
    def _compact_attempt(value: dict[str, Any]) -> dict[str, Any]:
        evaluation = value.get("evaluation") or {}
        return {
            "iteration": value.get("iteration"),
            "change_note": value.get("change_note"),
            "coverage_strategy": value.get("coverage_strategy"),
            "accepted": value.get("accepted"),
            "summary": evaluation.get("summary") or {},
            "failed_cases": [
                {
                    "case_id": row.get("case_id"),
                    "reason": (((row.get("checks") or [{}])[-1]).get("evaluation") or {}).get(
                        "reason"
                    ),
                }
                for row in evaluation.get("cases") or []
                if not all(
                    check.get("evaluation", {}).get("passed") is True
                    for check in row.get("checks") or []
                )
            ],
        }

    def _write_excel_version(
        self, optimized_points: dict[str, str], version: int
    ) -> Path:
        target = self.run_dir / f"审核要点_优化结果_v{version:03d}.xlsx"
        write_optimized_excel_updates(
            self.settings.excel_path, target, optimized_points
        )
        return target

    def _latest_version_number(self) -> int:
        versions = []
        for path in self.run_dir.glob("审核要点_优化结果_v*.xlsx"):
            match = re.search(r"_v(\d+)\.xlsx$", path.name)
            if match:
                versions.append(int(match.group(1)))
        return max(versions, default=0)

    def _write_result_outputs(
        self,
        *,
        manifest: dict[str, Any],
        groups: dict[str, list[ApprovalCase]],
        group_results: dict[str, dict[str, Any]],
        optimized_points: dict[str, str],
        final_excel: Path,
    ) -> dict[str, Any]:
        assert self.dataset is not None
        file_dir = self.run_dir / "文件结果"
        file_dir.mkdir(exist_ok=True)
        case_results: dict[str, dict[str, Any]] = {}
        for review_item, result in group_results.items():
            final_evaluation = result.get("final_evaluation") or result.get(
                "last_evaluation"
            )
            for case_result in (final_evaluation or {}).get("cases") or []:
                case_results[str(case_result.get("case_id") or "")] = case_result

        file_summaries: list[dict[str, Any]] = []
        for file_record in self.dataset.files:
            source_file = str(file_record.get("source_file") or "")
            file_cases = [
                case for case in self.dataset.cases if case.source_file == source_file
            ]
            payload = {
                **file_record,
                "cases": [
                    {
                        **case.to_dict(),
                        "group_status": group_results.get(case.review_item, {}).get(
                            "status", "skipped"
                        ),
                        "final_test": case_results.get(case.case_id),
                    }
                    for case in file_cases
                ],
            }
            output_name = f"{_safe_name(Path(source_file).stem)}_{_short_hash(source_file)}.json"
            output_path = file_dir / output_name
            _write_json(output_path, payload)
            file_summaries.append(
                {
                    "source_file": source_file,
                    "source_status": file_record.get("status"),
                    "usable_case_count": len(file_cases),
                    "output_file": str(output_path),
                }
            )

        status_counts: dict[str, int] = {}
        for result in group_results.values():
            status = str(result.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        summary = {
            "schema_version": 1,
            "run_id": self.run_dir.name,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_counts": manifest["counts"],
            "group_status_counts": status_counts,
            "optimized_review_items": sorted(
                optimized_points, key=_review_item_sort_key
            ),
            "optimized_count": len(optimized_points),
            "final_excel": str(final_excel),
            "group_results": group_results,
            "file_results": file_summaries,
            "skipped_files": self.dataset.skipped_files,
            "invalid_opinions": self.dataset.invalid_opinions,
        }
        _write_json(self.run_dir / "运行总结.json", summary)
        return summary

    @staticmethod
    def _log(message: str) -> None:
        print(f"[APPROVAL] {message}", flush=True)


def _dedupe(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value in (None, "") or value in result:
            continue
        result.append(value)
    return result


def _normalized_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _case_id(source_file: str, review_item: str) -> str:
    return f"item_{review_item}_{_short_hash(_normalized_name(source_file))}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return cleaned[:80] or "file"


def _review_item_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _truncate_json(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + "...[truncated]"


_EVALUATION_OUTPUT_KEYS = {
    "ai_result",
    "case_result",
    "reasoning",
    "errorReason",
    "reviewStatus",
    "reason",
    "reasons",
    "issue",
    "result",
    "detected_issue",
    "AI审评结果",
    "AI审评推理",
    "AI审评理由_reasons",
}

_NESTED_DECISION_KEYS = {
    "result",
    "reason",
    "reasons",
    "issue",
    "errorReason",
    "reviewStatus",
    "detected_issue",
    "observed_compliant",
}


def _compact_platform_output(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_decision_value(item) for item in value[:20]]
    if not isinstance(value, dict):
        return _compact_decision_value(value)
    compact = {
        key: _compact_decision_value(item)
        for key, item in value.items()
        if key in _EVALUATION_OUTPUT_KEYS
    }
    if compact:
        return compact
    return _truncate_json(value, 4000)


def _compact_decision_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:3000]
    if isinstance(value, list):
        return [_compact_decision_value(item) for item in value[:20]]
    if isinstance(value, dict):
        compact = {
            key: _compact_decision_value(item)
            for key, item in value.items()
            if key in _NESTED_DECISION_KEYS
        }
        return compact or _truncate_json(value, 2000)
    return value


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
