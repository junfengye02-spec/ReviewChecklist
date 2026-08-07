from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .approval_optimizer import ApprovalCase, ApprovalDataset, load_approval_dataset
from .batch import discover_documents
from .config import PROJECT_DIR
from .rules import load_review_rules


GENERATED_FILES = (
    "dataset_manifest.json",
    "data_quality_report.json",
    "data_quality_report.md",
    "platform_optimization_baseline.json",
    "platform_optimization_baseline.md",
    "functionality_runtime_inventory.md",
    "README.md",
    "checksums.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def normalized_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def review_item_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    required = {
        "dataset_name",
        "declared_plan_counts",
        "source_dir",
        "opinion_dir",
        "rules_workbook",
        "batch_summary",
        "mvp_run_summary",
        "approval_run_manifest",
        "approval_run_summary",
        "split_policy",
        "label_source",
        "pdf_text_policy",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Stage 0 config is missing: " + ", ".join(missing))
    return config


def pdf_text_audit(path: Path, min_characters: int) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for the Stage 0 PDF audit. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    page_stats: list[dict[str, Any]] = []
    extraction_errors: list[dict[str, Any]] = []
    try:
        with fitz.open(path) as document:
            for index, page in enumerate(document, start=1):
                try:
                    text = page.get_text("text")
                except Exception as exc:  # PyMuPDF exception classes vary by release.
                    extraction_errors.append(
                        {"page_number": index, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    text = ""
                non_whitespace_characters = len(re.sub(r"\s+", "", text))
                page_stats.append(
                    {
                        "page_number": index,
                        "non_whitespace_characters": non_whitespace_characters,
                        "replacement_character_count": text.count("\ufffd"),
                        "text_bearing": non_whitespace_characters >= min_characters,
                    }
                )
    except Exception as exc:
        return {
            "engine": {"name": "PyMuPDF", "version": str(fitz.VersionBind)},
            "page_count": None,
            "text_bearing_page_count": 0,
            "no_text_page_count": None,
            "no_text_page_ratio": None,
            "total_non_whitespace_characters": 0,
            "replacement_character_count": 0,
            "scan_or_ocr_candidate_pages": [],
            "extraction_errors": [{"page_number": None, "error": f"{type(exc).__name__}: {exc}"}],
        }

    page_count = len(page_stats)
    text_pages = [item for item in page_stats if item["text_bearing"]]
    no_text_pages = [item["page_number"] for item in page_stats if not item["text_bearing"]]
    return {
        "engine": {"name": "PyMuPDF", "version": str(fitz.VersionBind)},
        "page_count": page_count,
        "text_bearing_page_count": len(text_pages),
        "no_text_page_count": len(no_text_pages),
        "no_text_page_ratio": round(len(no_text_pages) / page_count, 6) if page_count else None,
        "total_non_whitespace_characters": sum(
            item["non_whitespace_characters"] for item in page_stats
        ),
        "replacement_character_count": sum(
            item["replacement_character_count"] for item in page_stats
        ),
        "scan_or_ocr_candidate_pages": no_text_pages,
        "extraction_errors": extraction_errors,
    }


def archive_inventory(source_dir: Path, document_hashes: dict[str, str]) -> list[dict[str, Any]]:
    archives: list[dict[str, Any]] = []
    for archive_path in sorted(source_dir.rglob("*.zip"), key=lambda path: path.relative_to(source_dir).as_posix()):
        members: list[dict[str, Any]] = []
        error = ""
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    if member.is_dir():
                        continue
                    content_hash = hashlib.sha256(archive.read(member)).hexdigest()
                    members.append(
                        {
                            "path": member.filename,
                            "size_bytes": member.file_size,
                            "sha256": content_hash,
                            "matches_source_pdf": content_hash in document_hashes.values(),
                        }
                    )
        except (OSError, zipfile.BadZipFile) as exc:
            error = f"{type(exc).__name__}: {exc}"
        archives.append(
            {
                "relative_path": archive_path.relative_to(source_dir).as_posix(),
                "size_bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
                "member_count": len(members),
                "members_matching_source_pdfs": sum(
                    1 for member in members if member["matches_source_pdf"]
                ),
                "members": members,
                "error": error or None,
            }
        )
    return archives


def source_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative_project_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def index_opinion_payloads(opinion_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(opinion_dir.rglob("*.json"), key=lambda item: item.relative_to(opinion_dir).as_posix()):
        if path.name == "汇总.json":
            continue
        payload = read_json(path)
        source_name = str((payload.get("source") or {}).get("file_name") or "")
        key = normalized_name(source_name)
        if not key:
            raise ValueError(f"Approval opinion is missing source.file_name: {path}")
        if key in indexed:
            raise ValueError(f"Duplicate approval opinion source file: {source_name}")
        indexed[key] = (path, payload)
    return indexed


def case_record(case: ApprovalCase, rules: dict[str, Any], split_policy: str, label_source: str) -> dict[str, Any]:
    rule = rules[case.review_item]
    record = {
        "case_id": case.case_id,
        "document_id": "doc_" + sha256_file(case.pdf_path)[:12],
        "source_file": case.source_file,
        "source_relative_path": case.source_relative_path,
        "source_pdf_sha256": sha256_file(case.pdf_path),
        "source_opinion_json": relative_project_path(case.source_json),
        "source_opinion_json_sha256": sha256_file(case.source_json),
        "source_type": "real_document_external_platform_output",
        "split": split_policy,
        "review_item": case.review_item,
        "review_point_id": case.review_point_id,
        "task_id": case.task_id,
        "product_id": case.product_id,
        "label_source": label_source,
        "expected_compliant": case.expected_compliant,
        "expected_opinion_count": len(case.expected_opinions),
        "expected_opinions_sha256": sha256_value(list(case.expected_opinions)),
        "evidence_entry_count": len(case.evidence),
        "evidence_sha256": sha256_value(list(case.evidence)),
        "reasoning_entry_count": len(case.reasoning),
        "reasoning_sha256": sha256_value(list(case.reasoning)),
        "reported_review_point_sha256": sha256_value(case.reported_review_point),
        "rule_workbook_row": rule.row_number,
        "rule_point_sha256": sha256_value(rule.point),
    }
    record["case_sha256"] = sha256_value(record)
    return record


def build_manifest(config_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], ApprovalDataset, list[dict[str, Any]]]:
    source_dir = project_path(str(config["source_dir"])).resolve()
    opinion_dir = project_path(str(config["opinion_dir"])).resolve()
    rules_path = project_path(str(config["rules_workbook"])).resolve()
    min_characters = int(
        (config.get("pdf_text_policy") or {}).get(
            "text_page_min_non_whitespace_characters", 1
        )
    )
    if min_characters < 1:
        raise ValueError("pdf_text_policy.text_page_min_non_whitespace_characters must be >= 1")

    documents = discover_documents(source_dir)
    opinion_payloads = index_opinion_payloads(opinion_dir)
    dataset = load_approval_dataset(opinion_dir, source_dir)
    rules = load_review_rules(rules_path)
    document_hashes = {document.file_name: sha256_file(document.path) for document in documents}
    cases_by_file: dict[str, list[ApprovalCase]] = defaultdict(list)
    for case in dataset.cases:
        cases_by_file[normalized_name(case.source_file)].append(case)

    document_records: list[dict[str, Any]] = []
    for document in documents:
        key = normalized_name(document.file_name)
        opinion_entry = opinion_payloads.get(key)
        if opinion_entry is None:
            raise ValueError(f"No approval JSON was found for source PDF: {document.file_name}")
        opinion_path, opinion_payload = opinion_entry
        source = opinion_payload.get("source") or {}
        review = opinion_payload.get("review") or {}
        pdf_audit = pdf_text_audit(document.path, min_characters)
        file_cases = sorted(
            cases_by_file[key], key=lambda case: review_item_sort_key(case.review_item)
        )
        record = {
            "document_id": "doc_" + document_hashes[document.file_name][:12],
            "file_name": document.file_name,
            "relative_path": document.relative_path.as_posix(),
            "category": document.expected_issue,
            "source_type": "real_document",
            "size_bytes": document.path.stat().st_size,
            "sha256": document_hashes[document.file_name],
            "approval_json": relative_project_path(opinion_path),
            "approval_json_sha256": sha256_file(opinion_path),
            "approval_embedded_sha256": str(source.get("sha256") or ""),
            "approval_embedded_sha256_matches": str(source.get("sha256") or "")
            == document_hashes[document.file_name],
            "approval_status": str(review.get("status") or ""),
            "task_id": str(review.get("task_id") or ""),
            "product_id": str(review.get("product_id") or ""),
            "platform_opinion_count": len(review.get("opinions") or []),
            "usable_case_count": len(file_cases),
            "case_ids": [case.case_id for case in file_cases],
            "pdf_text_audit": pdf_audit,
        }
        document_records.append(record)

    case_records = [
        case_record(case, rules, str(config["split_policy"]), str(config["label_source"]))
        for case in dataset.cases
        if case.review_item in rules
    ]
    case_records.sort(key=lambda item: (review_item_sort_key(item["review_item"]), item["source_file"]))
    archives = archive_inventory(source_dir, document_hashes)
    source_paths = [
        config_path,
        rules_path,
        project_path(str(config["batch_summary"])),
        project_path(str(config["mvp_run_summary"])),
        project_path(str(config["approval_run_manifest"])),
        project_path(str(config["approval_run_summary"])),
    ]
    manifest_core = {
        "schema_version": 1,
        "dataset_name": str(config["dataset_name"]),
        "source_type": "real_documents_with_external_platform_output",
        "label_source": str(config["label_source"]),
        "split_policy": str(config["split_policy"]),
        "declared_plan_counts": config["declared_plan_counts"],
        "actual_counts": {
            "documents": len(document_records),
            "approval_json_files": len(opinion_payloads),
            "platform_opinions": sum(item["platform_opinion_count"] for item in document_records),
            "usable_cases": len(case_records),
            "review_item_groups": len({item["review_item"] for item in case_records}),
            "noncompliant_cases": sum(
                1 for item in case_records if item["expected_compliant"] is False
            ),
            "compliant_cases": sum(
                1 for item in case_records if item["expected_compliant"] is True
            ),
        },
        "plan_count_comparison": {
            "documents_match": len(document_records)
            == int(config["declared_plan_counts"]["documents"]),
            "cases_match": len(case_records)
            == int(config["declared_plan_counts"]["cases"]),
        },
        "input_artifacts": [source_artifact(path) for path in source_paths],
        "documents": document_records,
        "cases": case_records,
        "auxiliary_archives": archives,
    }
    manifest = {
        "dataset_sha256": sha256_value(manifest_core),
        "generator_artifact": source_artifact(Path(__file__)),
        **manifest_core,
    }
    return manifest, dataset, document_records


def opinion_rows(opinion_dir: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path in sorted(opinion_dir.rglob("*.json"), key=lambda item: item.relative_to(opinion_dir).as_posix()):
        if path.name == "汇总.json":
            continue
        payload = read_json(path)
        for opinion in (payload.get("review") or {}).get("opinions") or []:
            if isinstance(opinion, dict):
                rows.append((path, payload, opinion))
    return rows


def field_present(value: Any) -> bool:
    return value not in (None, "", [], {})


def build_quality_report(config: dict[str, Any], manifest: dict[str, Any], dataset: ApprovalDataset) -> dict[str, Any]:
    opinion_dir = project_path(str(config["opinion_dir"])).resolve()
    rows = opinion_rows(opinion_dir)
    status_counts = Counter(str(row.get("status") or "") for _, _, row in rows)
    label_counts = Counter(
        "noncompliant" if row.get("compliant") is False else "compliant"
        if row.get("compliant") is True
        else "unknown"
        for _, _, row in rows
    )
    quality_fields = {
        "review_item": lambda row: row.get("review_item"),
        "review_point_id": lambda row: row.get("review_point_id"),
        "opinion": lambda row: row.get("opinion"),
        "evidence.page_number": lambda row: (row.get("evidence") or {}).get("page_number"),
        "evidence.material_excerpt": lambda row: (row.get("evidence") or {}).get("material_excerpt"),
        "reasoning": lambda row: row.get("reasoning"),
        "review_point": lambda row: row.get("review_point"),
        "platform_ids.result_id": lambda row: (row.get("platform_ids") or {}).get("result_id"),
    }
    field_completeness = {
        name: {
            "present": sum(1 for _, _, row in rows if field_present(getter(row))),
            "missing": sum(1 for _, _, row in rows if not field_present(getter(row))),
        }
        for name, getter in quality_fields.items()
    }
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, payload, row in rows:
        source_file = str((payload.get("source") or {}).get("file_name") or "")
        by_case[(source_file, str(row.get("review_item") or ""))].append(row)
    conflicts = []
    duplicate_case_rows = []
    for (source_file, review_item), grouped_rows in sorted(by_case.items()):
        labels = {row.get("compliant") for row in grouped_rows if isinstance(row.get("compliant"), bool)}
        if len(labels) > 1:
            conflicts.append(
                {
                    "source_file": source_file,
                    "review_item": review_item,
                    "labels": sorted(labels),
                    "opinion_count": len(grouped_rows),
                }
            )
        if len(grouped_rows) > 1:
            duplicate_case_rows.append(
                {
                    "source_file": source_file,
                    "review_item": review_item,
                    "opinion_count": len(grouped_rows),
                    "labels": sorted(
                        {str(row.get("compliant")) for row in grouped_rows}
                    ),
                }
            )

    document_hashes: dict[str, list[str]] = defaultdict(list)
    for document in manifest["documents"]:
        document_hashes[str(document["sha256"])].append(str(document["relative_path"]))
    duplicate_documents = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(document_hashes.items())
        if len(paths) > 1
    ]
    all_case_document_ids = Counter(case["document_id"] for case in manifest["cases"])
    cases_per_document = sorted(all_case_document_ids.values())
    split_counts = Counter(case["split"] for case in manifest["cases"])
    category_counts = Counter(document["category"] for document in manifest["documents"])
    label_by_review_item: dict[str, dict[str, int]] = {}
    for item in sorted({case["review_item"] for case in manifest["cases"]}, key=review_item_sort_key):
        item_cases = [case for case in manifest["cases"] if case["review_item"] == item]
        label_by_review_item[item] = {
            "noncompliant": sum(not case["expected_compliant"] for case in item_cases),
            "compliant": sum(case["expected_compliant"] for case in item_cases),
        }
    source_hash_failures = [
        document["relative_path"]
        for document in manifest["documents"]
        if not document["approval_embedded_sha256_matches"]
    ]
    no_text_pages = sum(
        int(document["pdf_text_audit"]["no_text_page_count"] or 0)
        for document in manifest["documents"]
    )
    total_pages = sum(
        int(document["pdf_text_audit"]["page_count"] or 0)
        for document in manifest["documents"]
    )
    extraction_error_documents = [
        document["relative_path"]
        for document in manifest["documents"]
        if document["pdf_text_audit"]["extraction_errors"]
    ]
    scan_candidates = [
        {
            "relative_path": document["relative_path"],
            "pages": document["pdf_text_audit"]["scan_or_ocr_candidate_pages"],
        }
        for document in manifest["documents"]
        if document["pdf_text_audit"]["scan_or_ocr_candidate_pages"]
    ]
    archive_members = sum(archive["member_count"] for archive in manifest["auxiliary_archives"])
    archive_matches = sum(
        archive["members_matching_source_pdfs"] for archive in manifest["auxiliary_archives"]
    )

    risks = [
        {
            "id": "LABEL_SOURCE_NOT_INDEPENDENT",
            "severity": "critical",
            "status": "detected",
            "evidence": (
                f"{manifest['actual_counts']['usable_cases']} cases derive expected_compliant "
                "from the same external platform reviewStatus captured in 审批意见 JSON."
            ),
            "impact": "The current cases are regression labels, not independent human gold labels; they cannot support an unbiased accuracy claim for the same platform.",
            "required_control": "Add immutable HumanDecision labels and record annotator/review metadata before publication-quality conclusion metrics.",
        },
        {
            "id": "NO_DOCUMENT_LEVEL_SPLIT",
            "severity": "critical",
            "status": "detected",
            "evidence": f"All {manifest['actual_counts']['documents']} documents and {manifest['actual_counts']['usable_cases']} cases are split={config['split_policy']!r}.",
            "impact": "There is no validation or frozen test set, so optimization and final evaluation can reuse the same source documents.",
            "required_control": "Create a versioned document-level optimization/validation/frozen-test split before candidate selection or release reporting.",
        },
        {
            "id": "CATEGORY_NAME_LABEL_LEAKAGE",
            "severity": "high",
            "status": "risk",
            "evidence": "Every source PDF is stored under a category directory that is copied into the manifest as category/expected_issue.",
            "impact": "Passing folder names into prompts, retrieval metadata, or features could reveal the intended issue class.",
            "required_control": "Keep category labels out of model inputs; use neutral archive names as the existing batch uploader does.",
        },
        {
            "id": "OPTIMIZATION_EVALUATION_REUSE",
            "severity": "high",
            "status": "detected",
            "evidence": "The recorded approval optimizer run has one case pool and no frozen document split in its input manifest.",
            "impact": "Candidate selection can overfit to the same documents used for optimization and protection checks.",
            "required_control": "Reserve document-disjoint validation and frozen-test sets; never feed frozen-test results back into optimization.",
        },
        {
            "id": "LLM_AS_ACCEPTANCE_JUDGE",
            "severity": "high",
            "status": "detected",
            "evidence": "ApprovalOptimizer delegates semantic case acceptance to LlmClient.evaluate_approval_group.",
            "impact": "Recorded optimization status is a model-judged experiment result and requires human review before it becomes a label or release decision.",
            "required_control": "Store human acceptance decisions separately and measure evaluator agreement on a reviewed subset.",
        },
        {
            "id": "MULTIPLE_PLATFORM_ROWS_COLLAPSED",
            "severity": "medium",
            "status": "detected",
            "evidence": f"{len(rows)} platform opinions collapse to {manifest['actual_counts']['usable_cases']} file/review-item cases; {len(duplicate_case_rows)} cases have multiple rows.",
            "impact": "A grouped case may hide multiple platform findings; semantic review is needed before treating grouped labels as atomic ground truth.",
            "required_control": "Preserve the source row IDs and adjudicate any multi-row case before changing its label.",
        },
        {
            "id": "EVIDENCE_COMPLETENESS",
            "severity": "medium",
            "status": "detected",
            "evidence": (
                f"Missing page numbers: {field_completeness['evidence.page_number']['missing']}/{len(rows)}; "
                f"missing excerpts: {field_completeness['evidence.material_excerpt']['missing']}/{len(rows)}."
            ),
            "impact": "Current findings cannot yet be treated as consistently page-citable evidence labels.",
            "required_control": "Add evidence page/chunk verification before creating retrieval labels or final findings.",
        },
    ]
    return {
        "schema_version": 1,
        "dataset_sha256": manifest["dataset_sha256"],
        "counts": {
            "documents": manifest["actual_counts"]["documents"],
            "approval_json_files": manifest["actual_counts"]["approval_json_files"],
            "platform_opinions": len(rows),
            "usable_cases": manifest["actual_counts"]["usable_cases"],
            "review_item_groups": manifest["actual_counts"]["review_item_groups"],
            "noncompliant_opinions": label_counts.get("noncompliant", 0),
            "compliant_opinions": label_counts.get("compliant", 0),
            "noncompliant_cases": manifest["actual_counts"]["noncompliant_cases"],
            "compliant_cases": manifest["actual_counts"]["compliant_cases"],
            "duplicate_extra_opinions": len(rows) - manifest["actual_counts"]["usable_cases"],
            "multi_row_cases": len(duplicate_case_rows),
            "label_conflicts_within_case": len(conflicts),
            "invalid_opinions": len(dataset.invalid_opinions),
            "skipped_files": len(dataset.skipped_files),
        },
        "source_integrity": {
            "embedded_pdf_hash_mismatches": source_hash_failures,
            "duplicate_pdf_contents": duplicate_documents,
            "approval_jsons_without_matching_pdf": [
                item["source_json"] for item in dataset.skipped_files
            ],
            "archive_count": len(manifest["auxiliary_archives"]),
            "archive_member_count": archive_members,
            "archive_members_matching_source_pdfs": archive_matches,
        },
        "case_shape": {
            "split_counts": dict(sorted(split_counts.items())),
            "cases_per_document_min": min(cases_per_document) if cases_per_document else 0,
            "cases_per_document_max": max(cases_per_document) if cases_per_document else 0,
            "cases_per_document_uniform": len(set(cases_per_document)) <= 1,
            "document_categories": dict(sorted(category_counts.items())),
            "labels_by_review_item": label_by_review_item,
        },
        "platform_opinion_fields": {
            "status_counts": dict(sorted(status_counts.items())),
            "derived_label_counts": dict(sorted(label_counts.items())),
            "field_completeness": field_completeness,
        },
        "pdf_text_quality": {
            "page_count": total_pages,
            "no_text_page_count": no_text_pages,
            "no_text_page_ratio": round(no_text_pages / total_pages, 6) if total_pages else None,
            "documents_with_extraction_errors": extraction_error_documents,
            "scan_or_ocr_candidates_by_document": scan_candidates,
            "interpretation": "No-text pages are scan/OCR candidates only; this audit does not run OCR or prove that every candidate page is scanned.",
        },
        "label_conflicts": conflicts,
        "multi_row_cases": duplicate_case_rows,
        "risks": risks,
    }


def summary_view(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    if not isinstance(summary, dict):
        return None
    keys = (
        "all_passed",
        "case_count",
        "target_count",
        "protection_count",
        "passed_count",
        "target_cases_passed",
        "protection_cases_passed",
        "required_stability_runs",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def build_platform_baseline(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    batch_summary_path = project_path(str(config["batch_summary"])).resolve()
    mvp_path = project_path(str(config["mvp_run_summary"])).resolve()
    approval_manifest_path = project_path(str(config["approval_run_manifest"])).resolve()
    approval_summary_path = project_path(str(config["approval_run_summary"])).resolve()
    batch_summary = read_json(batch_summary_path)
    mvp_summary = read_json(mvp_path)
    approval_manifest = read_json(approval_manifest_path)
    approval_summary = read_json(approval_summary_path)
    batch_totals = (batch_summary.get("totals") or {})
    raw_opinions = opinion_rows(project_path(str(config["opinion_dir"])).resolve())
    noncompliant_opinions = sum(
        1 for _, _, opinion in raw_opinions if opinion.get("compliant") is False
    )
    approval_groups: dict[str, Any] = {}
    for review_item, group in sorted(
        (approval_summary.get("group_results") or {}).items(),
        key=lambda item: review_item_sort_key(str(item[0])),
    ):
        if not isinstance(group, dict):
            continue
        final = group.get("final_evaluation") or group.get("last_evaluation")
        approval_groups[str(review_item)] = {
            "status": group.get("status"),
            "accepted": group.get("accepted"),
            "accepted_iteration": group.get("accepted_iteration"),
            "iterations_run": group.get("iterations_run"),
            "target_count": group.get("target_count"),
            "protection_count": group.get("protection_count"),
            "baseline": summary_view(group.get("baseline_evaluation")),
            "final": summary_view(final),
            "reason": group.get("reason"),
            "error": group.get("error"),
        }
    mvp_baseline = mvp_summary.get("baseline_workflow_evaluation") or {}
    mvp_final = mvp_summary.get("final_evaluation") or {}
    return {
        "schema_version": 1,
        "dataset_sha256": manifest["dataset_sha256"],
        "source_artifacts": [
            source_artifact(batch_summary_path),
            source_artifact(mvp_path),
            source_artifact(approval_manifest_path),
            source_artifact(approval_summary_path),
        ],
        "platform_batch_output": {
            "run_id": batch_summary.get("run_id"),
            "recorded_totals": batch_totals,
            "recomputed_totals": {
                "source_files": manifest["actual_counts"]["documents"],
                "opinion_count": manifest["actual_counts"]["platform_opinions"],
                "issue_opinion_count": noncompliant_opinions,
                "usable_cases": manifest["actual_counts"]["usable_cases"],
                "noncompliant_case_count": manifest["actual_counts"]["noncompliant_cases"],
            },
            "consistency": {
                "source_files_match": batch_totals.get("source_files") == manifest["actual_counts"]["documents"],
                "opinion_count_match": batch_totals.get("opinion_count") == manifest["actual_counts"]["platform_opinions"],
                "issue_opinion_count_match": batch_totals.get("issue_count")
                == noncompliant_opinions,
            },
            "reported_review_item_ids": batch_summary.get("review_item_ids") or [],
            "claim_boundary": "These are recorded external-platform output counts, not independently adjudicated accuracy metrics.",
        },
        "single_item_mvp_experiment": {
            "run_id": mvp_summary.get("run_id"),
            "review_item": mvp_summary.get("review_item"),
            "stability_runs_required": mvp_summary.get("stability_runs_required"),
            "baseline": {
                "hit": mvp_baseline.get("hit"),
                "reason": mvp_baseline.get("reason"),
            },
            "final": {
                "hit": mvp_final.get("hit"),
                "reason": mvp_final.get("reason"),
            },
            "iterations_run": mvp_summary.get("iterations_run"),
            "claim_boundary": "This is one recorded known-defect workflow experiment; it is not an overall accuracy result.",
        },
        "approval_optimizer_baseline": {
            "run_id": approval_summary.get("run_id"),
            "input_counts": approval_summary.get("input_counts") or approval_manifest.get("counts") or {},
            "group_status_counts": approval_summary.get("group_status_counts") or {},
            "optimized_review_items": approval_summary.get("optimized_review_items") or [],
            "groups": approval_groups,
            "claim_boundary": "Statuses come from recorded workflow tests whose semantic acceptance is judged by the configured LLM; they are not publication-ready human-label metrics.",
        },
        "unavailable_metrics": {
            "precision": "not computed: no independent human gold labels",
            "recall": "not computed: no independent human gold labels",
            "f1": "not computed: no independent human gold labels",
            "evidence_recall_at_k": "not computed: no document/chunk-level retrieval labels",
            "latency_percentiles": "not computed: recorded artifacts do not provide a fixed per-stage timing dataset",
            "model_cost": "not computed: recorded artifacts do not provide token or billing data",
        },
    }


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_list = list(headers)
    lines = ["| " + " | ".join(header_list) + " |", "| " + " | ".join("---" for _ in header_list) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def quality_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Stage 0 Data Quality, Label Conflict, and Leakage Report",
        "",
        "This report is generated from fixed local source files. It does not call the remote platform and does not create labels.",
        "",
        "## Actual Counts",
        "",
        markdown_table(
            ("Measure", "Value"),
            (
                ("Source PDFs", counts["documents"]),
                ("Approval JSON files", counts["approval_json_files"]),
                ("Platform opinions", counts["platform_opinions"]),
                ("Usable file/review-item cases", counts["usable_cases"]),
                ("Review item groups", counts["review_item_groups"]),
                ("Noncompliant platform opinions", counts["noncompliant_opinions"]),
                ("Compliant platform opinions", counts["compliant_opinions"]),
                ("Noncompliant grouped cases", counts["noncompliant_cases"]),
                ("Compliant grouped cases", counts["compliant_cases"]),
                ("Within-case label conflicts", counts["label_conflicts_within_case"]),
            ),
        ),
        "",
        "## Evidence and PDF Quality",
        "",
        markdown_table(
            ("Measure", "Value"),
            (
                ("PDF pages", report["pdf_text_quality"]["page_count"]),
                ("No-text page candidates", report["pdf_text_quality"]["no_text_page_count"]),
                ("No-text page ratio", report["pdf_text_quality"]["no_text_page_ratio"]),
                ("Missing evidence page numbers", report["platform_opinion_fields"]["field_completeness"]["evidence.page_number"]["missing"]),
                ("Missing evidence excerpts", report["platform_opinion_fields"]["field_completeness"]["evidence.material_excerpt"]["missing"]),
                ("Documents with PDF extraction errors", len(report["pdf_text_quality"]["documents_with_extraction_errors"])),
            ),
        ),
        "",
        report["pdf_text_quality"]["interpretation"],
        "",
        "## Risks",
        "",
        markdown_table(
            ("ID", "Severity", "Status", "Evidence"),
            (
                (risk["id"], risk["severity"], risk["status"], risk["evidence"])
                for risk in report["risks"]
            ),
        ),
        "",
        "A zero count of machine-detected label conflicts does not establish label correctness. The current labels remain platform-derived until independently reviewed.",
        "",
    ]
    return "\n".join(lines)


def platform_markdown(report: dict[str, Any]) -> str:
    batch = report["platform_batch_output"]
    mvp = report["single_item_mvp_experiment"]
    optimizer = report["approval_optimizer_baseline"]
    lines = [
        "# Stage 0 Platform and Rule Optimization Baseline",
        "",
        "All figures below are reconstructed from fixed historical local artifacts. No remote task is rerun.",
        "",
        "## Platform Batch Output",
        "",
        markdown_table(
            ("Measure", "Recorded value", "Recomputed value"),
            (
                ("Source files", batch["recorded_totals"].get("source_files"), batch["recomputed_totals"]["source_files"]),
                ("Opinions", batch["recorded_totals"].get("opinion_count"), batch["recomputed_totals"]["opinion_count"]),
                ("Issue opinions", batch["recorded_totals"].get("issue_count"), batch["recomputed_totals"]["issue_opinion_count"]),
                ("Usable cases", "n/a", batch["recomputed_totals"]["usable_cases"]),
                ("Noncompliant grouped cases", "n/a", batch["recomputed_totals"]["noncompliant_case_count"]),
            ),
        ),
        "",
        batch["claim_boundary"],
        "",
        "## Known-Defect MVP Experiment",
        "",
        markdown_table(
            ("Run", "Review item", "Stability threshold", "Baseline", "Final", "Iterations"),
            (
                (
                    mvp["run_id"],
                    mvp["review_item"],
                    mvp["stability_runs_required"],
                    mvp["baseline"]["reason"],
                    mvp["final"]["reason"],
                    mvp["iterations_run"],
                ),
            ),
        ),
        "",
        mvp["claim_boundary"],
        "",
        "## Approval Optimizer",
        "",
        "Status counts: " + json.dumps(optimizer["group_status_counts"], ensure_ascii=False, sort_keys=True),
        "",
        "Optimized review items: " + ", ".join(str(item) for item in optimizer["optimized_review_items"]),
        "",
        markdown_table(
            ("Review item", "Status", "Targets", "Protections", "Accepted iteration", "Final passed"),
            (
                (
                    item,
                    group.get("status"),
                    group.get("target_count"),
                    group.get("protection_count"),
                    group.get("accepted_iteration"),
                    (group.get("final") or {}).get("all_passed"),
                )
                for item, group in optimizer["groups"].items()
            ),
        ),
        "",
        optimizer["claim_boundary"],
        "",
        "## Metrics Not Claimed",
        "",
        markdown_table(
            ("Metric", "Reason"), report["unavailable_metrics"].items()
        ),
        "",
    ]
    return "\n".join(lines)


def functionality_markdown(manifest: dict[str, Any], platform: dict[str, Any]) -> str:
    source_files = sorted((PROJECT_DIR / "tender_review").rglob("*.py"))
    test_files = sorted((PROJECT_DIR / "tests").glob("test_*.py"))
    rows = [
        ("MVP CLI", "Validates inputs; packages a PDF; creates/uploads/starts/polls a platform task; writes run artifacts", "tender_review/cli.py, tender_review/pipeline.py"),
        ("Platform adapter", "Token, product, Excel import, ZIP upload, task lifecycle, result download, workflow test", "tender_review/platform.py"),
        ("Rule workbook", "Loads unique review items and writes versioned workbook copies without mutating the source", "tender_review/rules.py"),
        ("Batch review", "Discovers PDFs, submits per-file tasks, validates reports, writes structured JSON, supports recovery", "tender_review/batch.py, tender_review/batch_cli.py"),
        ("Result normalization", "Repairs known mojibake and derives compliant/noncompliant from platform status", "tender_review/results.py, tender_review/batch.py"),
        ("Approval optimization", "Groups platform-derived cases by review item, evaluates candidates jointly, writes accepted workbook versions", "tender_review/approval_optimizer.py, tender_review/approval_cli.py"),
        ("Stage 0 audit", "Freezes local inputs with hashes; builds the manifest and reports without remote calls", "tender_review/baseline.py"),
        ("Stage 1 API and Worker", "Provides process entry points, health/readiness, versioned API contracts, and offline Fake composition", "tender_review/api, tender_review/worker, tender_review/bootstrap"),
        ("Stage 1 infrastructure", "Provides SQLAlchemy metadata, Alembic migration, MySQL/MinIO production bootstrap, and content-addressed artifacts", "tender_review/infrastructure, alembic"),
    ]
    lines = [
        "# Stage 0 Functionality and Runtime Inventory",
        "",
        "This inventory describes code present in this repository and the fixed historical artifacts used by the Stage 0 baseline. It does not imply that later implementation-plan phases are complete.",
        "",
        "## Existing Functionality",
        "",
        markdown_table(("Area", "Observed behavior", "Evidence"), rows),
        "",
        "## Existing Automated Coverage",
        "",
        markdown_table(
            ("File", "SHA-256", "Lines"),
            (
                (relative_project_path(path), sha256_file(path), len(path.read_text(encoding="utf-8").splitlines()))
                for path in test_files
            ),
        ),
        "",
        "## Fixed Historical Runtime Data",
        "",
        markdown_table(
            ("Source", "Run/data ID", "Observed facts", "Boundary"),
            (
                (
                    "Real material directory",
                    manifest["dataset_sha256"],
                    f"{manifest['actual_counts']['documents']} PDFs, {len(manifest['auxiliary_archives'])} ZIP archives, {manifest['actual_counts']['usable_cases']} grouped cases",
                    "Local source inventory",
                ),
                (
                    "External platform batch",
                    platform["platform_batch_output"]["run_id"],
                    f"{platform['platform_batch_output']['recorded_totals'].get('opinion_count')} opinions, {platform['platform_batch_output']['recorded_totals'].get('issue_count')} issue opinions, {platform['platform_batch_output']['recorded_totals'].get('completed_files')}/{manifest['actual_counts']['documents']} completed",
                    "Historical platform output; not accuracy",
                ),
                (
                    "Known-defect MVP",
                    platform["single_item_mvp_experiment"]["run_id"],
                    f"item {platform['single_item_mvp_experiment']['review_item']}: {platform['single_item_mvp_experiment']['baseline']['reason']} -> {platform['single_item_mvp_experiment']['final']['reason']}",
                    "One known sample only",
                ),
                (
                    "Approval optimizer",
                    platform["approval_optimizer_baseline"]["run_id"],
                    json.dumps(platform["approval_optimizer_baseline"]["group_status_counts"], ensure_ascii=False, sort_keys=True),
                    "LLM-judged historical experiment",
                ),
            ),
        ),
        "",
        "## Source Inventory",
        "",
        markdown_table(
            ("File", "SHA-256", "Lines"),
            (
                (relative_project_path(path), sha256_file(path), len(path.read_text(encoding="utf-8").splitlines()))
                for path in source_files
            ),
        ),
        "",
        "## Runtime Evidence Boundaries",
        "",
        "- The historical batch and optimizer artifacts are external-platform records, not locally rerun executions.",
        "- Existing code calls the remote platform and an OpenAI-compatible LLM when live commands are used; Stage 0 does neither.",
        "- Stage 1 adds a FastAPI/Worker skeleton, SQLAlchemy/Alembic schema, and MinIO adapter; this inventory does not treat those additions as historical Stage 0 runtime evidence.",
        "- Durable MySQL job leasing, PDF/OCR processing, retrieval, LangGraph review, and human-decision workflows remain unimplemented Stage 2+ capabilities.",
        "",
    ]
    return "\n".join(lines)


def baseline_readme() -> str:
    return """# Stage 0 Baseline

This directory is generated from fixed local source data. The command makes no network calls and does not change source PDFs, approval JSON, historical runs, or the source workbook.

Rebuild all Stage 0 artifacts:

```powershell
python -m pip install -r requirements-dev.txt
python -m tender_review.baseline --config baseline/stage0_config.json --output-dir baseline
```

Verify that committed artifacts exactly match a clean rebuild:

```powershell
python -m tender_review.baseline --config baseline/stage0_config.json --output-dir baseline --check
```

Run the Stage 0 and existing regression tests:

```powershell
python -m unittest discover -s tests -v
```

`dataset_manifest.json` is an inventory of real source PDFs and grouped file/review-item cases. It records source hashes and does not manufacture labels. `data_quality_report.*` documents label provenance, conflicts, evidence completeness, and leakage risks. `platform_optimization_baseline.*` only restates fixed historical artifacts and explicitly lists metrics that cannot be claimed.
"""


def render_outputs(config_path: Path) -> dict[str, str]:
    config = load_config(config_path)
    manifest, dataset, _ = build_manifest(config_path, config)
    quality = build_quality_report(config, manifest, dataset)
    platform = build_platform_baseline(config, manifest)
    return {
        "dataset_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "data_quality_report.json": json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "data_quality_report.md": quality_markdown(quality),
        "platform_optimization_baseline.json": json.dumps(platform, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "platform_optimization_baseline.md": platform_markdown(platform),
        "functionality_runtime_inventory.md": functionality_markdown(manifest, platform),
        "README.md": baseline_readme(),
    }


def write_outputs(output_dir: Path, outputs: dict[str, str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    checksums = {name: sha256_file(output_dir / name) for name in sorted(outputs)}
    write_json(output_dir / "checksums.json", {"schema_version": 1, "files": checksums})
    return checksums


def rebuild(config_path: Path, output_dir: Path) -> dict[str, str]:
    outputs = render_outputs(config_path.resolve())
    return write_outputs(output_dir.resolve(), outputs)


def check_rebuild(config_path: Path, output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return [f"Missing output directory: {output_dir}"]
    with tempfile.TemporaryDirectory() as temporary:
        rebuilt_dir = Path(temporary) / "baseline"
        rebuild(config_path, rebuilt_dir)
        mismatches = []
        for name in GENERATED_FILES:
            expected = output_dir / name
            actual = rebuilt_dir / name
            if not expected.is_file():
                mismatches.append(f"Missing generated artifact: {name}")
                continue
            if expected.read_bytes() != actual.read_bytes():
                mismatches.append(f"Generated artifact differs: {name}")
        return mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Stage 0 tender-review baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "local-data" / "baseline" / "stage0_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "local-data" / "baseline-output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the output directory exactly matches a clean rebuild",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if args.check:
        mismatches = check_rebuild(config_path, output_dir)
        if mismatches:
            print("Stage 0 baseline check failed:")
            for mismatch in mismatches:
                print(f"- {mismatch}")
            return 1
        print("Stage 0 baseline check passed.")
        return 0
    checksums = rebuild(config_path, output_dir)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "generated_files": checksums},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
