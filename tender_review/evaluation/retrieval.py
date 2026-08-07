from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil
from typing import Iterable, Literal, Self

from pydantic import Field, model_validator

from tender_review.shared.contracts import ContractModel


SHA256_PATTERN = r"^[0-9a-f]{64}$"


def stable_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class RankedChunkHit(ContractModel):
    """Evaluation-owned interchange DTO for a ranked retrieval hit."""

    chunk_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    rank: int = Field(ge=1)
    score: float
    source: str = Field(min_length=1, max_length=128)
    text: str = ""


class RankedSearchResult(ContractModel):
    """JSON-stable result contract, intentionally independent of retriever code."""

    retriever: str = Field(min_length=1, max_length=128)
    hits: tuple[RankedChunkHit, ...] = ()

    @model_validator(mode="after")
    def has_canonical_ranking(self) -> Self:
        ranks = tuple(hit.rank for hit in self.hits)
        if ranks != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("hit ranks must be contiguous and ordered from 1")
        chunk_ids = tuple(hit.chunk_id for hit in self.hits)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("ranked results must not repeat a chunk_id")
        return self


class RetrievalLabel(ContractModel):
    """One query with explicit chunk-level evidence labels and provenance."""

    query_id: str = Field(min_length=1, max_length=128)
    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    query: str = Field(min_length=1)
    document_id: str | None = Field(default=None, min_length=1, max_length=256)
    document_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_case_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    relevant_chunk_ids: tuple[str, ...] = ()
    required_chunk_groups: tuple[tuple[str, ...], ...] = ()
    no_answer: bool = False
    annotator_id: str = Field(min_length=1, max_length=128)
    annotation_source: Literal["human", "ai_prefill"] = "human"
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=128)
    review_source: Literal["human", "ai_prefill"] | None = None
    review_status: Literal["draft", "reviewed", "approved", "rejected"]

    @model_validator(mode="after")
    def has_consistent_relevance(self) -> Self:
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids must be unique")
        if self.no_answer and (
            self.relevant_chunk_ids or self.required_chunk_groups
        ):
            raise ValueError("no-answer labels cannot contain relevant chunks")
        if not self.no_answer and not self.relevant_chunk_ids:
            # Draft candidate labels remain representable, but are never baseline-ready.
            if self.review_status in {"reviewed", "approved"}:
                raise ValueError("answerable reviewed labels need relevant chunks")
        relevant = set(self.relevant_chunk_ids)
        for group in self.required_chunk_groups:
            if not group:
                raise ValueError("required chunk groups must not be empty")
            if not set(group) <= relevant:
                raise ValueError("required chunk groups must reference relevant chunks")
        return self


class RetrievalDataset(ContractModel):
    """Versioned retrieval labels, separate from historical conclusion labels."""

    dataset_version_id: str = Field(min_length=1, max_length=128)
    source_description: str = Field(min_length=1)
    status: Literal["provisional", "frozen"] = "provisional"
    source_package_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    labels: tuple[RetrievalLabel, ...]
    dataset_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_identity(self) -> Self:
        query_ids = tuple(label.query_id for label in self.labels)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("dataset query_id values must be unique")
        case_ids = tuple(label.case_id for label in self.labels if label.case_id)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("dataset case_id values must be unique")
        if self.status == "frozen":
            if not self.labels:
                raise ValueError("frozen datasets must contain labels")
            if self.dataset_sha256 is None:
                raise ValueError("frozen datasets require dataset_sha256")
            expected = dataset_sha256(self)
            if self.dataset_sha256 != expected:
                raise ValueError("dataset_sha256 does not match dataset content")
        return self


class DatasetReadiness(ContractModel):
    eligible_for_real_baseline: bool
    total_cases: int
    approved_human_cases: int
    missing_human_annotations: int
    missing_human_reviews: int
    same_person_reviews: int
    invalid_provenance_cases: int
    reasons: tuple[str, ...]


class RetrievalVariantMetrics(ContractModel):
    variant: str
    cases_evaluated: int
    relevant_cases: int
    no_answer_cases: int
    two_sided_cases: int
    latency_samples: int
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    two_sided_evidence_rate_at_10: float | None
    no_answer_false_positive_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None

    @property
    def two_sided_evidence_rate(self) -> float | None:
        """Compatibility spelling for the explicitly Top-10 metric."""

        return self.two_sided_evidence_rate_at_10


class RetrievalEvaluationReport(ContractModel):
    report_kind: Literal["provisional", "real_baseline"]
    claims_allowed: bool
    dataset_version_id: str
    dataset_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    readiness: DatasetReadiness
    variants: tuple[RetrievalVariantMetrics, ...]
    interpretation: str
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_report_hash(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match report content")
        return self


class RetrievalCaseResult(ContractModel):
    case_id: str = Field(min_length=1, max_length=128)
    query_id: str = Field(min_length=1, max_length=128)
    result: RankedSearchResult
    latency_ms: float = Field(ge=0)


class RetrievalVariantRun(ContractModel):
    """Stage 4A -> 4B ranked-results artifact contract."""

    dataset_version_id: str = Field(min_length=1, max_length=128)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    variant: str = Field(min_length=1, max_length=128)
    retriever_config_sha256: str = Field(pattern=SHA256_PATTERN)
    cases: tuple[RetrievalCaseResult, ...]
    results_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_results(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        query_ids = tuple(item.query_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("variant result case_id values must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("variant result query_id values must be unique")
        payload = self.model_dump(mode="json", exclude={"results_sha256"})
        if self.results_sha256 != stable_sha256(payload):
            raise ValueError("results_sha256 does not match variant results")
        return self


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationInput:
    label: RetrievalLabel
    result: RankedSearchResult
    latency_ms: float | None = None


def dataset_sha256(dataset: RetrievalDataset) -> str:
    payload = dataset.model_dump(mode="json", exclude={"dataset_sha256"})
    return stable_sha256(payload)


def assess_dataset_readiness(dataset: RetrievalDataset) -> DatasetReadiness:
    missing_annotations = 0
    missing_reviews = 0
    same_person_reviews = 0
    invalid_provenance = 0
    approved = 0
    for label in dataset.labels:
        if label.annotation_source != "human":
            missing_annotations += 1
        if label.review_status != "approved" or not label.reviewer_id:
            missing_reviews += 1
        if label.reviewer_id and label.reviewer_id == label.annotator_id:
            same_person_reviews += 1
        if label.annotation_source != "human" or label.review_source != "human":
            invalid_provenance += 1
        if (
            label.annotation_source == "human"
            and label.review_source == "human"
            and label.review_status == "approved"
            and label.reviewer_id
            and label.reviewer_id != label.annotator_id
        ):
            approved += 1

    reasons: list[str] = []
    reasons.append(
        "legacy retrieval datasets are not bound to persisted A4 FROZEN dataset provenance"
    )
    if dataset.status != "frozen":
        reasons.append("dataset is not frozen")
    if not dataset.labels:
        reasons.append("dataset contains no cases")
    if dataset.dataset_sha256 is None:
        reasons.append("dataset hash is missing")
    if missing_annotations:
        reasons.append(f"{missing_annotations} cases lack human annotation")
    if missing_reviews:
        reasons.append(f"{missing_reviews} cases lack approved human review")
    if same_person_reviews:
        reasons.append(f"{same_person_reviews} cases were self-reviewed")
    if invalid_provenance:
        reasons.append(f"{invalid_provenance} cases have non-human provenance")
    return DatasetReadiness(
        eligible_for_real_baseline=not reasons,
        total_cases=len(dataset.labels),
        approved_human_cases=approved,
        missing_human_annotations=missing_annotations,
        missing_human_reviews=missing_reviews,
        same_person_reviews=same_person_reviews,
        invalid_provenance_cases=invalid_provenance,
        reasons=tuple(reasons),
    )


def evaluate_variant(
    *,
    variant: str,
    cases: Iterable[RetrievalEvaluationInput],
) -> RetrievalVariantMetrics:
    """Compute metrics from one ranked result per explicitly labeled query.

    Recall and MRR exclude no-answer cases. Cross-section success requires at
    least one chunk from every required group in the first ten hits. A
    no-answer false positive is any returned hit, so callers must apply their
    retrieval acceptance threshold before constructing this result contract.
    Latency percentiles use the deterministic nearest-rank definition.
    """

    normalized = tuple(cases)
    if not variant.strip():
        raise ValueError("variant must not be blank")
    query_ids = tuple(case.label.query_id for case in normalized)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("each query_id must occur exactly once per variant")
    if any(case.latency_ms is not None and case.latency_ms < 0 for case in normalized):
        raise ValueError("latency_ms must not be negative")

    relevant_cases = tuple(case for case in normalized if not case.label.no_answer)
    recalls_5 = tuple(_recall(case.label, case.result, limit=5) for case in relevant_cases)
    recalls_10 = tuple(
        _recall(case.label, case.result, limit=10) for case in relevant_cases
    )
    reciprocal_ranks = tuple(
        _reciprocal_rank(case.label, case.result) for case in relevant_cases
    )
    two_sided = tuple(
        _required_groups_hit(case.label, case.result, limit=10)
        for case in relevant_cases
        if case.label.required_chunk_groups
    )
    no_answer_cases = tuple(case for case in normalized if case.label.no_answer)
    false_positives = sum(bool(case.result.hits) for case in no_answer_cases)
    latencies = tuple(
        case.latency_ms for case in normalized if case.latency_ms is not None
    )
    return RetrievalVariantMetrics(
        variant=variant.strip(),
        cases_evaluated=len(normalized),
        relevant_cases=len(relevant_cases),
        no_answer_cases=len(no_answer_cases),
        two_sided_cases=len(two_sided),
        latency_samples=len(latencies),
        recall_at_5=_mean(recalls_5),
        recall_at_10=_mean(recalls_10),
        mrr=_mean(reciprocal_ranks),
        two_sided_evidence_rate_at_10=_mean(two_sided),
        no_answer_false_positive_rate=(
            false_positives / len(no_answer_cases) if no_answer_cases else None
        ),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def build_evaluation_report(
    *,
    dataset: RetrievalDataset,
    variants: Iterable[RetrievalVariantMetrics],
    require_real_baseline: bool = True,
) -> RetrievalEvaluationReport:
    readiness = assess_dataset_readiness(dataset)
    if require_real_baseline and not readiness.eligible_for_real_baseline:
        raise ValueError(
            "real baseline gate failed: " + "; ".join(readiness.reasons)
        )
    normalized = tuple(sorted(variants, key=lambda item: item.variant))
    if len({item.variant for item in normalized}) != len(normalized):
        raise ValueError("evaluation report variants must be unique")
    if any(item.cases_evaluated != len(dataset.labels) for item in normalized):
        raise ValueError("each variant must evaluate every dataset case")
    is_real = readiness.eligible_for_real_baseline
    payload = {
        "schema_version": 1,
        "report_kind": "real_baseline" if is_real else "provisional",
        "claims_allowed": is_real,
        "dataset_version_id": dataset.dataset_version_id,
        "dataset_sha256": dataset.dataset_sha256,
        "readiness": readiness.model_dump(mode="json"),
        "variants": [item.model_dump(mode="json") for item in normalized],
        "interpretation": (
            "Metrics are backed by frozen, independently reviewed human labels."
            if is_real
            else "Provisional diagnostics only; Recall, MRR and latency must not be claimed as real baseline metrics."
        ),
    }
    return RetrievalEvaluationReport(
        **payload,
        report_sha256=stable_sha256(payload),
    )


def evaluate_variant_run(
    *,
    dataset: RetrievalDataset,
    run: RetrievalVariantRun,
) -> RetrievalVariantMetrics:
    if run.dataset_version_id != dataset.dataset_version_id:
        raise ValueError("variant results target another dataset version")
    if run.dataset_sha256 != dataset.dataset_sha256:
        raise ValueError("variant results target another dataset hash")
    labels_by_case = {
        (label.case_id or label.query_id): label for label in dataset.labels
    }
    results_by_case = {item.case_id: item for item in run.cases}
    if set(results_by_case) != set(labels_by_case):
        missing = sorted(set(labels_by_case) - set(results_by_case))
        extra = sorted(set(results_by_case) - set(labels_by_case))
        raise ValueError(
            f"variant case coverage differs: missing={missing}, extra={extra}"
        )
    cases: list[RetrievalEvaluationInput] = []
    for case_id in sorted(labels_by_case):
        label = labels_by_case[case_id]
        artifact = results_by_case[case_id]
        if artifact.query_id != label.query_id:
            raise ValueError(f"variant query_id differs for case {case_id}")
        cases.append(
            RetrievalEvaluationInput(
                label=label,
                result=artifact.result,
                latency_ms=artifact.latency_ms,
            )
        )
    return evaluate_variant(variant=run.variant, cases=cases)


def _recall(
    label: RetrievalLabel,
    result: RankedSearchResult,
    *,
    limit: int,
) -> float:
    relevant = set(label.relevant_chunk_ids)
    if not relevant:
        return 0.0
    found = {hit.chunk_id for hit in result.hits[:limit]}
    return len(relevant & found) / len(relevant)


def _reciprocal_rank(label: RetrievalLabel, result: RankedSearchResult) -> float:
    relevant = set(label.relevant_chunk_ids)
    for rank, hit in enumerate(result.hits, start=1):
        if hit.chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _required_groups_hit(
    label: RetrievalLabel,
    result: RankedSearchResult,
    *,
    limit: int,
) -> float:
    found = {hit.chunk_id for hit in result.hits[:limit]}
    return float(
        all(set(group) & found for group in label.required_chunk_groups)
    )


def _mean(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: tuple[float, ...], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, ceil(quantile * len(ordered)) - 1)
    return float(ordered[position])
