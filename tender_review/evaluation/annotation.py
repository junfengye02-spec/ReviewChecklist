from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Self

from pydantic import Field, model_validator

from tender_review.shared.contracts import CallContext, ContractModel

from .retrieval import (
    SHA256_PATTERN,
    RetrievalDataset,
    RetrievalLabel,
    stable_sha256,
)


class CandidateChunk(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=SHA256_PATTERN)


class RebuiltChunkDocument(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parse_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_set_sha256: str = Field(pattern=SHA256_PATTERN)
    chunks: tuple[CandidateChunk, ...]


RebuildChunkDocument = Callable[[str, bytes, CallContext], RebuiltChunkDocument]


class WorkPackageDocument(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    source_manifest_document_id: str = Field(min_length=1, max_length=128)
    source_relative_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parse_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_set_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_count: int = Field(ge=1)


class AutomatedCandidatePrefill(ContractModel):
    provenance: Literal["deterministic_external_platform_hint"] = (
        "deterministic_external_platform_hint"
    )
    is_human_annotation: Literal[False] = False
    candidate_chunk_ids: tuple[str, ...] = ()
    platform_page_hints: tuple[int, ...] = ()
    platform_excerpt_hints: tuple[str, ...] = ()
    explanation: str = (
        "Candidate navigation aid only. It is not a relevance label or review."
    )


class WorkPackageCase(ContractModel):
    case_id: str = Field(min_length=1, max_length=128)
    query_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)
    query_source: Literal["historical_external_platform_review_point"] = (
        "historical_external_platform_review_point"
    )
    source_case_sha256: str = Field(pattern=SHA256_PATTERN)
    source_opinion_json: str = Field(min_length=1)
    source_opinion_json_sha256: str = Field(pattern=SHA256_PATTERN)
    review_item: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=256)
    source_manifest_document_id: str = Field(min_length=1, max_length=128)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    automated_prefill: AutomatedCandidatePrefill
    chunk_relevance: tuple[()] = ()
    annotator_id: None = None
    reviewer_id: None = None
    review_state: Literal["unreviewed"] = "unreviewed"


class AnnotationWorkPackage(ContractModel):
    package_id: str = Field(min_length=1, max_length=128)
    package_kind: Literal["chunk_retrieval_candidate_annotation"] = (
        "chunk_retrieval_candidate_annotation"
    )
    source_description: str = Field(min_length=1)
    source_dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    phase3_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    phase3_audit_report_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_catalog_sha256: str = Field(pattern=SHA256_PATTERN)
    documents: tuple[WorkPackageDocument, ...]
    cases: tuple[WorkPackageCase, ...]
    required_human_cases: int = Field(ge=1)
    annotation_policy: str = Field(min_length=1)
    work_package_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_identity(self) -> Self:
        if len(self.documents) != len({item.document_id for item in self.documents}):
            raise ValueError("work package document_id values must be unique")
        if len(self.cases) != len({item.case_id for item in self.cases}):
            raise ValueError("work package case_id values must be unique")
        if self.required_human_cases > len(self.cases):
            raise ValueError("required_human_cases exceeds candidate case count")
        payload = self.model_dump(mode="json", exclude={"work_package_sha256"})
        if self.work_package_sha256 != stable_sha256(payload):
            raise ValueError("work_package_sha256 does not match package content")
        return self


class ChunkRelevanceDecision(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    relevance: Literal["relevant", "not_relevant"]


class HumanCaseDecision(ContractModel):
    case_id: str = Field(min_length=1, max_length=128)
    query_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=256)
    no_answer: bool
    chunk_labels: tuple[ChunkRelevanceDecision, ...]
    required_chunk_groups: tuple[tuple[str, ...], ...] = ()
    annotator_id: str = Field(min_length=1, max_length=128)
    annotation_source: Literal["human"]
    annotated_at: datetime
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=128)
    review_source: Literal["human"] | None = None
    review_state: Literal["annotated", "approved", "rejected"]
    reviewed_at: datetime | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def has_consistent_human_decision(self) -> Self:
        chunk_ids = tuple(item.chunk_id for item in self.chunk_labels)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_labels must not repeat chunk_id")
        relevant = {
            item.chunk_id
            for item in self.chunk_labels
            if item.relevance == "relevant"
        }
        if self.no_answer and relevant:
            raise ValueError("no-answer decisions cannot mark chunks relevant")
        if not self.no_answer and not relevant:
            raise ValueError("answerable decisions need a relevant chunk")
        for group in self.required_chunk_groups:
            if not group:
                raise ValueError("required_chunk_groups cannot contain an empty group")
            if not set(group) <= relevant:
                raise ValueError("required_chunk_groups must use relevant chunks")
        if self.review_state in {"approved", "rejected"}:
            if not self.reviewer_id or self.review_source != "human":
                raise ValueError("reviewed decisions need a human reviewer")
            if self.reviewed_at is None:
                raise ValueError("reviewed decisions need reviewed_at")
            if self.reviewer_id == self.annotator_id:
                raise ValueError("annotator and reviewer must be different people")
        elif any((self.reviewer_id, self.review_source, self.reviewed_at)):
            raise ValueError("annotated decisions must not claim review fields")
        return self


class HumanAnnotationBundle(ContractModel):
    work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    decisions: tuple[HumanCaseDecision, ...]

    @model_validator(mode="after")
    def has_unique_cases(self) -> Self:
        case_ids = tuple(item.case_id for item in self.decisions)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("annotation bundle case_id values must be unique")
        return self


class ImportedAnnotationBundle(HumanAnnotationBundle):
    annotation_bundle_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_bundle_hash(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"annotation_bundle_sha256"})
        if self.annotation_bundle_sha256 != stable_sha256(payload):
            raise ValueError("annotation_bundle_sha256 does not match bundle content")
        return self


class AnnotationGapReport(ContractModel):
    work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_cases: int
    required_human_cases: int
    imported_cases: int
    human_annotated_cases: int
    approved_human_cases: int
    rejected_cases: int
    missing_case_ids: tuple[str, ...]
    not_approved_case_ids: tuple[str, ...]
    real_dataset_ready: bool
    blocker: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_annotation_work_package(
    *,
    baseline_manifest_path: Path,
    phase3_audit_path: Path,
    pdf_root: Path,
    project_root: Path,
    rebuild_document: RebuildChunkDocument,
) -> tuple[AnnotationWorkPackage, tuple[CandidateChunk, ...], dict[str, Any]]:
    manifest = _read_json(baseline_manifest_path)
    audit = _read_json(phase3_audit_path)
    manifest_documents_payload = manifest.get("documents")
    manifest_cases_payload = manifest.get("cases")
    if not isinstance(manifest_documents_payload, list) or not manifest_documents_payload:
        raise ValueError("Stage 0 manifest must contain at least one document")
    if not isinstance(manifest_cases_payload, list) or not manifest_cases_payload:
        raise ValueError("Stage 0 manifest must contain at least one candidate case")
    actual_counts = manifest.get("actual_counts") or {}
    if actual_counts.get("documents") != len(manifest_documents_payload):
        raise ValueError("Stage 0 document count does not match the manifest")
    if actual_counts.get("usable_cases") != len(manifest_cases_payload):
        raise ValueError("Stage 0 case count does not match the manifest")
    if audit.get("document_failures"):
        raise ValueError("Phase 3 audit contains document failures")

    audit_by_path = {item["relative_path"]: item for item in audit["documents"]}
    manifest_documents = {
        item["document_id"]: item for item in manifest["documents"]
    }
    chunks_by_document: dict[str, tuple[CandidateChunk, ...]] = {}
    work_documents: list[WorkPackageDocument] = []
    all_chunks: list[CandidateChunk] = []

    for index, source in enumerate(
        sorted(manifest["documents"], key=lambda item: item["relative_path"].casefold()),
        start=1,
    ):
        relative_path = source["relative_path"].replace("\\", "/")
        path = pdf_root / Path(relative_path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"source PDF hash differs: {relative_path}")
        expected_audit = audit_by_path.get(relative_path)
        if expected_audit is None:
            raise ValueError(f"PDF is missing from Phase 3 audit: {relative_path}")
        rebuilt = rebuild_document(
            relative_path,
            content,
            CallContext(
                call_id=f"phase4-annotation-export-{index}",
                timeout_seconds=3600,
            ),
        )
        if rebuilt.document_id != relative_path or rebuilt.document_sha256 != digest:
            raise ValueError(f"rebuilt document identity differs: {relative_path}")
        if rebuilt.parse_artifact_sha256 != expected_audit["parse_artifact_sha256"]:
            raise ValueError(f"parse artifact differs from Phase 3 audit: {relative_path}")
        if rebuilt.chunk_set_sha256 != expected_audit["chunk_set_sha256"]:
            raise ValueError(f"chunk set differs from Phase 3 audit: {relative_path}")
        candidates = rebuilt.chunks
        chunks_by_document[relative_path] = candidates
        all_chunks.extend(candidates)
        work_documents.append(
            WorkPackageDocument(
                document_id=relative_path,
                source_manifest_document_id=source["document_id"],
                source_relative_path=relative_path,
                document_sha256=digest,
                parse_artifact_sha256=rebuilt.parse_artifact_sha256,
                chunk_set_sha256=rebuilt.chunk_set_sha256,
                chunk_count=len(candidates),
            )
        )

    work_cases: list[WorkPackageCase] = []
    approvals: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source in manifest["documents"]:
        approval_path = project_root / Path(source["approval_json"])
        if sha256_file(approval_path) != source["approval_json_sha256"]:
            raise ValueError(f"approval JSON hash differs: {source['approval_json']}")
        approval = _read_json(approval_path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for opinion in approval["review"]["opinions"]:
            grouped[str(opinion["review_item"])].append(opinion)
        approvals[source["document_id"]] = grouped

    for source_case in sorted(manifest["cases"], key=lambda item: item["case_id"]):
        source_document = manifest_documents[source_case["document_id"]]
        relative_path = source_document["relative_path"].replace("\\", "/")
        opinions = approvals[source_case["document_id"]].get(
            str(source_case["review_item"]), []
        )
        query = _query_from_opinions(opinions, source_case["review_item"])
        pages, excerpts = _platform_hints(opinions)
        candidate_ids = _prefill_candidates(
            query=query,
            pages=pages,
            excerpts=excerpts,
            chunks=chunks_by_document[relative_path],
        )
        work_cases.append(
            WorkPackageCase(
                case_id=source_case["case_id"],
                query_id=source_case["case_id"],
                query=query,
                source_case_sha256=source_case["case_sha256"],
                source_opinion_json=source_case["source_opinion_json"],
                source_opinion_json_sha256=source_case[
                    "source_opinion_json_sha256"
                ],
                review_item=str(source_case["review_item"]),
                document_id=relative_path,
                source_manifest_document_id=source_case["document_id"],
                document_sha256=source_case["source_pdf_sha256"],
                automated_prefill=AutomatedCandidatePrefill(
                    candidate_chunk_ids=candidate_ids,
                    platform_page_hints=pages,
                    platform_excerpt_hints=excerpts,
                ),
            )
        )

    chunk_catalog = tuple(all_chunks)
    chunk_catalog_sha256 = stable_sha256(
        [item.model_dump(mode="json") for item in chunk_catalog]
    )
    payload = {
        "schema_version": 1,
        "package_id": "phase4-real-pdf-retrieval-candidates-v1",
        "package_kind": "chunk_retrieval_candidate_annotation",
        "source_description": (
            "Candidate queries from historical external-platform review points; "
            "chunk catalog rebuilt from the manifest documents and parser contract."
        ),
        "source_dataset_sha256": manifest["dataset_sha256"],
        "source_manifest_sha256": sha256_file(baseline_manifest_path),
        "phase3_audit_sha256": sha256_file(phase3_audit_path),
        "phase3_audit_report_sha256": audit["report_sha256"],
        "chunk_catalog_sha256": chunk_catalog_sha256,
        "documents": [item.model_dump(mode="json") for item in work_documents],
        "cases": [item.model_dump(mode="json") for item in work_cases],
        "required_human_cases": len(work_cases),
        "annotation_policy": (
            "All cases require a named human annotator and a different named human "
            "reviewer. Automated candidates are navigation hints only."
        ),
    }
    package = AnnotationWorkPackage(
        **payload,
        work_package_sha256=stable_sha256(payload),
    )
    template = annotation_template(package)
    return package, chunk_catalog, template


def annotation_template(package: AnnotationWorkPackage) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "work_package_sha256": package.work_package_sha256,
        "instructions": (
            "Replace every null field. Mark inspected chunks relevant/not_relevant; "
            "add missed chunk IDs from chunks.jsonl. annotation_source and "
            "review_source must be 'human'. Reviewer must differ from annotator."
        ),
        "decisions": [
            {
                "schema_version": 1,
                "case_id": case.case_id,
                "query_id": case.query_id,
                "document_id": case.document_id,
                "no_answer": None,
                "chunk_labels": [
                    {
                        "schema_version": 1,
                        "chunk_id": chunk_id,
                        "relevance": None,
                    }
                    for chunk_id in case.automated_prefill.candidate_chunk_ids
                ],
                "required_chunk_groups": [],
                "annotator_id": None,
                "annotation_source": None,
                "annotated_at": None,
                "reviewer_id": None,
                "review_source": None,
                "review_state": None,
                "reviewed_at": None,
                "notes": None,
            }
            for case in package.cases
        ],
    }


def import_annotations(
    *,
    package: AnnotationWorkPackage,
    chunks: Iterable[CandidateChunk],
    raw_bundle: Mapping[str, Any],
) -> ImportedAnnotationBundle:
    payload = dict(raw_bundle)
    payload.pop("instructions", None)
    bundle = HumanAnnotationBundle.model_validate(payload)
    if bundle.work_package_sha256 != package.work_package_sha256:
        raise ValueError("annotation bundle targets a different work package")
    package_cases = {item.case_id: item for item in package.cases}
    chunks_by_id = {item.chunk_id: item for item in chunks}
    for decision in bundle.decisions:
        case = package_cases.get(decision.case_id)
        if case is None:
            raise ValueError(f"unknown annotation case_id: {decision.case_id}")
        if decision.query_id != case.query_id or decision.document_id != case.document_id:
            raise ValueError(f"immutable case identity differs: {decision.case_id}")
        for label in decision.chunk_labels:
            chunk = chunks_by_id.get(label.chunk_id)
            if chunk is None:
                raise ValueError(f"unknown chunk_id: {label.chunk_id}")
            if chunk.document_id != case.document_id:
                raise ValueError(
                    f"chunk belongs to another document: {label.chunk_id}"
                )
    canonical = bundle.model_dump(mode="json")
    return ImportedAnnotationBundle(
        **canonical,
        annotation_bundle_sha256=stable_sha256(canonical),
    )


def annotation_gaps(
    package: AnnotationWorkPackage,
    bundle: ImportedAnnotationBundle | None = None,
) -> AnnotationGapReport:
    decisions = {item.case_id: item for item in bundle.decisions} if bundle else {}
    missing = tuple(
        item.case_id for item in package.cases if item.case_id not in decisions
    )
    not_approved = tuple(
        item.case_id
        for item in package.cases
        if item.case_id in decisions
        and decisions[item.case_id].review_state != "approved"
    )
    approved = sum(
        item.review_state == "approved" for item in decisions.values()
    )
    rejected = sum(
        item.review_state == "rejected" for item in decisions.values()
    )
    ready = (
        approved >= package.required_human_cases
        and not missing
        and not not_approved
    )
    return AnnotationGapReport(
        work_package_sha256=package.work_package_sha256,
        candidate_cases=len(package.cases),
        required_human_cases=package.required_human_cases,
        imported_cases=len(decisions),
        human_annotated_cases=len(decisions),
        approved_human_cases=approved,
        rejected_cases=rejected,
        missing_case_ids=missing,
        not_approved_case_ids=not_approved,
        real_dataset_ready=ready,
        blocker=(
            None
            if ready
            else f"{len(missing)} cases are missing and {len(not_approved)} are not approved"
        ),
    )


def freeze_dataset(
    *,
    package: AnnotationWorkPackage,
    bundle: ImportedAnnotationBundle,
    dataset_version_id: str,
    require_complete_package: bool = True,
) -> RetrievalDataset:
    gaps = annotation_gaps(package, bundle)
    if require_complete_package and not gaps.real_dataset_ready:
        raise ValueError(f"dataset freeze gate failed: {gaps.blocker}")
    approved = {
        item.case_id: item
        for item in bundle.decisions
        if item.review_state == "approved"
    }
    if not approved:
        raise ValueError("dataset freeze gate failed: no approved human decisions")
    package_cases = {item.case_id: item for item in package.cases}
    labels: list[RetrievalLabel] = []
    for case_id in sorted(approved):
        decision = approved[case_id]
        case = package_cases[case_id]
        relevant = tuple(
            item.chunk_id
            for item in decision.chunk_labels
            if item.relevance == "relevant"
        )
        labels.append(
            RetrievalLabel(
                query_id=case.query_id,
                case_id=case.case_id,
                query=case.query,
                document_id=case.document_id,
                document_sha256=case.document_sha256,
                source_case_sha256=case.source_case_sha256,
                relevant_chunk_ids=relevant,
                required_chunk_groups=decision.required_chunk_groups,
                no_answer=decision.no_answer,
                annotator_id=decision.annotator_id,
                annotation_source="human",
                reviewer_id=decision.reviewer_id,
                review_source="human",
                review_status="approved",
            )
        )
    payload = {
        "schema_version": 1,
        "dataset_version_id": dataset_version_id,
        "source_description": (
            "Frozen chunk-level retrieval labels imported from work package "
            f"{package.package_id}."
        ),
        "status": "frozen",
        "source_package_sha256": package.work_package_sha256,
        "labels": [item.model_dump(mode="json") for item in labels],
    }
    payload["dataset_sha256"] = stable_sha256(payload)
    return RetrievalDataset.model_validate(payload)


def load_work_package(directory: Path) -> tuple[AnnotationWorkPackage, tuple[CandidateChunk, ...]]:
    package = AnnotationWorkPackage.model_validate(
        _read_json(directory / "work_package.json")
    )
    chunks = tuple(
        CandidateChunk.model_validate(json.loads(line))
        for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    actual_hash = stable_sha256([item.model_dump(mode="json") for item in chunks])
    if actual_hash != package.chunk_catalog_sha256:
        raise ValueError("chunks.jsonl does not match chunk_catalog_sha256")
    document_counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        document_counts[chunk.document_id] += 1
    for document in package.documents:
        if document_counts[document.document_id] != document.chunk_count:
            raise ValueError(f"chunk count differs for {document.document_id}")
    return package, chunks


def render_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_jsonl(values: Iterable[ContractModel]) -> str:
    return "".join(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for value in values
    )


def work_package_outputs(
    package: AnnotationWorkPackage,
    chunks: tuple[CandidateChunk, ...],
    template: dict[str, Any],
) -> dict[str, str]:
    gaps = annotation_gaps(package)
    gap_markdown = (
        "# Phase 4 Retrieval Annotation Gap Report\n\n"
        f"- Candidate cases: {gaps.candidate_cases}\n"
        f"- Required human annotations: {gaps.required_human_cases}\n"
        f"- Approved human reviews: {gaps.approved_human_cases}\n"
        f"- Missing cases: {len(gaps.missing_case_ids)}\n"
        f"- Real dataset ready: {'yes' if gaps.real_dataset_ready else 'no'}\n\n"
        "This is an acceptance blocker for Stage 4. Historical platform output "
        "and automated candidate hints are not human chunk-relevance labels.\n"
    )
    readme = f"""# Phase 4 Retrieval Annotation Work Package

This package was rebuilt from {len(package.documents)} private source documents, the Stage 0 manifest,
the Phase 3 parser/chunker contract, and historical external-platform review
materials. `automated_prefill` only narrows navigation; it is never a human
annotation or review.

## Human annotation and review

1. Open `annotation_template.json` and use `chunks.jsonl` to inspect every
   candidate case. Add missed chunk IDs when the prefill is incomplete.
2. Set explicit `relevant` / `not_relevant` chunk decisions and `no_answer`.
   For cross-section evidence, put each required side in a separate
   `required_chunk_groups` group.
3. Set `annotator_id`, `annotation_source: "human"`, and `annotated_at`.
4. A different person checks the PDF/chunks and sets `reviewer_id`,
   `review_source: "human"`, `review_state: "approved"`, and `reviewed_at`.
5. Import, freeze, and evaluate with the commands documented in the project
   README. Freeze fails until all {len(package.cases)} cases are independently approved.
"""
    outputs = {
        "work_package.json": render_json(package),
        "chunks.jsonl": render_jsonl(chunks),
        "annotation_template.json": render_json(template),
        "gap_report.json": render_json(gaps),
        "gap_report.md": gap_markdown,
        "README.md": readme,
    }
    checksums = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in sorted(outputs.items())
    }
    outputs["checksums.json"] = render_json(
        {"schema_version": 1, "files": checksums}
    )
    return outputs


def validate_work_package_directory(directory: Path) -> AnnotationGapReport:
    package, _ = load_work_package(directory)
    checksums_path = directory / "checksums.json"
    if checksums_path.is_file():
        checksums = _read_json(checksums_path).get("files", {})
        if "checksums.json" in checksums:
            raise ValueError(
                "checksums.json must not contain its own checksum; self-hashing is circular"
            )
        for name, expected in checksums.items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"work package checksum differs: {name}")
    return annotation_gaps(package)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_from_opinions(opinions: list[dict[str, Any]], review_item: object) -> str:
    for opinion in opinions:
        value = opinion.get("review_point")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    for opinion in opinions:
        value = opinion.get("opinion")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"审评项 {review_item}"


def _platform_hints(
    opinions: list[dict[str, Any]],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    pages: set[int] = set()
    excerpts: set[str] = set()
    for opinion in opinions:
        evidence = opinion.get("evidence") or {}
        page = evidence.get("page_number")
        if isinstance(page, int) and page > 0:
            pages.add(page)
        excerpt = evidence.get("material_excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            excerpts.add(excerpt.strip())
    return tuple(sorted(pages)), tuple(sorted(excerpts))


def _prefill_candidates(
    *,
    query: str,
    pages: tuple[int, ...],
    excerpts: tuple[str, ...],
    chunks: tuple[CandidateChunk, ...],
    limit: int = 10,
) -> tuple[str, ...]:
    query_tokens = _tokens(" ".join((query, *excerpts)))
    page_set = set(pages)
    ranked: list[tuple[float, str]] = []
    for chunk in chunks:
        score = 0.0
        if any(chunk.page_start <= page <= chunk.page_end for page in page_set):
            score += 1000.0
        normalized_text = _normalize(chunk.text)
        if any(_normalize(excerpt) in normalized_text for excerpt in excerpts):
            score += 2000.0
        chunk_tokens = _tokens(chunk.text)
        if query_tokens:
            score += len(query_tokens & chunk_tokens) / len(query_tokens)
        ranked.append((score, chunk.chunk_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    positive = [chunk_id for score, chunk_id in ranked if score > 0]
    return tuple((positive or [item[1] for item in ranked])[:limit])


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _tokens(value: str) -> set[str]:
    normalized = _normalize(value)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}
