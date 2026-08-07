"""Stage 4C provisional retrieval comparison.

This module deliberately does not construct :class:`RetrievalLabel`.  The
candidate chunk IDs in the Stage 4 work package are navigation hints derived
from historical platform material, not human relevance labels.  The
provisional runner therefore has a separate input/result/report contract and
cannot emit a real-baseline report.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from tender_review.retrieval import (
    BM25Config,
    EmbeddingRequest,
    FakeEmbeddingProvider,
    HybridConfig,
    RetrievalDocument,
    SearchRequest,
    VectorConfig,
    build_retriever,
)
from tender_review.shared.contracts import CallContext, ContractModel

from .annotation import AnnotationWorkPackage, CandidateChunk
from .retrieval import (
    RankedChunkHit,
    RankedSearchResult,
    SHA256_PATTERN,
    stable_sha256,
)


PROVISIONAL_INPUT_VERSION = "phase4-provisional-input-v1"
PROVISIONAL_CONFIG_VERSION = "phase4-provisional-config-v1"
PROVISIONAL_REPORT_VERSION = "phase4-provisional-report-v1"
PROVISIONAL_IMPLEMENTATION_VERSION = "phase4-provisional-implementation-v1"
PROVISIONAL_IMPLEMENTATION_SHA256 = stable_sha256(
    {"version": PROVISIONAL_IMPLEMENTATION_VERSION}
)


class ProvisionalCandidateCase(ContractModel):
    """One candidate query with navigation hints and no human fields."""

    case_id: str = Field(min_length=1, max_length=128)
    query_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    source_case_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_chunk_ids: tuple[str, ...] = ()
    candidate_label_provenance: Literal["navigation_hint"] = "navigation_hint"
    is_human_annotation: Literal[False] = False

    @model_validator(mode="after")
    def has_unique_identity(self) -> Self:
        if len(self.candidate_chunk_ids) != len(set(self.candidate_chunk_ids)):
            raise ValueError("candidate_chunk_ids must be unique")
        return self


class ProvisionalEvaluationInput(ContractModel):
    """Frozen-by-hash input shared by BM25, vector and hybrid runs."""

    input_version: Literal["phase4-provisional-input-v1"] = PROVISIONAL_INPUT_VERSION
    dataset_version_id: str = Field(min_length=1, max_length=128)
    source_work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_catalog_sha256: str = Field(pattern=SHA256_PATTERN)
    cases: tuple[ProvisionalCandidateCase, ...]
    status: Literal["provisional"] = "provisional"
    claims_allowed: Literal[False] = False
    input_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_input_hash(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        query_ids = tuple(item.query_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("provisional case_id values must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("provisional query_id values must be unique")
        payload = self.model_dump(mode="json", exclude={"input_sha256"})
        if self.input_sha256 != stable_sha256(payload):
            raise ValueError("input_sha256 does not match provisional input")
        return self


class ProvisionalSharedConfig(ContractModel):
    """Configuration shared across every strategy in one comparison."""

    config_version: Literal["phase4-provisional-config-v1"] = PROVISIONAL_CONFIG_VERSION
    search_limit: int = Field(default=10, ge=1, le=100)
    document_scope: Literal["case_document_only"] = "case_document_only"
    embedding_model: str = Field(default="fake-embedding", min_length=1)
    embedding_dimensions: int = Field(default=32, ge=1, le=256)
    bm25: BM25Config = Field(default_factory=BM25Config)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    @property
    def config_sha256(self) -> str:
        return stable_sha256(self.model_dump(mode="json"))


class ProvisionalCaseResult(ContractModel):
    case_id: str = Field(min_length=1, max_length=128)
    query_id: str = Field(min_length=1, max_length=128)
    result: RankedSearchResult
    latency_ms: float = Field(ge=0)
    navigation_hint_chunk_ids: tuple[str, ...] = ()


class ProvisionalVariantRun(ContractModel):
    """Stage 4C result artifact exchanged by the three strategy runs."""

    artifact_version: Literal["phase4-provisional-run-v1"] = "phase4-provisional-run-v1"
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    source_work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_catalog_sha256: str = Field(pattern=SHA256_PATTERN)
    shared_config_sha256: str = Field(pattern=SHA256_PATTERN)
    implementation_version_sha256: str = Field(pattern=SHA256_PATTERN)
    variant: Literal["bm25", "vector", "hybrid_rrf"]
    retriever_config_sha256: str = Field(pattern=SHA256_PATTERN)
    cases: tuple[ProvisionalCaseResult, ...]
    results_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_results(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        query_ids = tuple(item.query_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("provisional result case_id values must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("provisional result query_id values must be unique")
        payload = self.model_dump(mode="json", exclude={"results_sha256"})
        if self.results_sha256 != stable_sha256(payload):
            raise ValueError("results_sha256 does not match provisional results")
        return self


class ProvisionalVariantDiagnostics(ContractModel):
    variant: Literal["bm25", "vector", "hybrid_rrf"]
    cases_evaluated: int = Field(ge=0)
    cases_with_results: int = Field(ge=0)
    empty_result_cases: int = Field(ge=0)
    cases_with_navigation_hint_overlap_at_10: int = Field(ge=0)
    mean_navigation_hint_overlap_at_10: float | None = None
    navigation_hint_overlap_rate_at_10: float | None = None
    result_count_mean: float | None = None
    latency_samples: int = Field(ge=0)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    diagnostic_error_cases: Mapping[str, tuple[str, ...]]


class ProvisionalEvaluationReport(ContractModel):
    """Non-claimable comparison report with explicit human-data gate status."""

    report_version: Literal["phase4-provisional-report-v1"] = PROVISIONAL_REPORT_VERSION
    report_kind: Literal["provisional"] = "provisional"
    claims_allowed: Literal[False] = False
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    source_work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    chunk_catalog_sha256: str = Field(pattern=SHA256_PATTERN)
    shared_config_sha256: str = Field(pattern=SHA256_PATTERN)
    implementation_version_sha256: str = Field(pattern=SHA256_PATTERN)
    required_human_cases: int = Field(ge=0)
    human_annotation_cases: int = Field(ge=0)
    approved_human_cases: int = Field(ge=0)
    variants: tuple[ProvisionalVariantDiagnostics, ...]
    default_strategy_candidate: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_valid_report_hash(self) -> Self:
        if self.human_annotation_cases != 0 or self.approved_human_cases != 0:
            raise ValueError("provisional report cannot contain human label counts")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match provisional report")
        if "must not" not in self.interpretation.casefold():
            raise ValueError("provisional interpretation must state claim boundary")
        return self


def build_provisional_input(
    package: AnnotationWorkPackage,
    chunks: Iterable[CandidateChunk],
) -> ProvisionalEvaluationInput:
    """Project the 4B work package into a candidate-only input contract."""

    chunk_ids_by_document: dict[str, set[str]] = {}
    for chunk in chunks:
        chunk_ids_by_document.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
    cases = tuple(
        ProvisionalCandidateCase(
            case_id=item.case_id,
            query_id=item.query_id,
            query=item.query,
            document_id=item.document_id,
            document_sha256=item.document_sha256,
            source_case_sha256=item.source_case_sha256,
            candidate_chunk_ids=tuple(item.automated_prefill.candidate_chunk_ids),
        )
        for item in package.cases
    )
    valid_chunks = set().union(*chunk_ids_by_document.values()) if chunk_ids_by_document else set()
    for case in cases:
        unknown = sorted(set(case.candidate_chunk_ids) - valid_chunks)
        if unknown:
            raise ValueError(f"provisional case references unknown chunks: {unknown[0]}")
        wrong_document = sorted(
            chunk_id
            for chunk_id in case.candidate_chunk_ids
            if chunk_id not in chunk_ids_by_document.get(case.document_id, set())
        )
        if wrong_document:
            raise ValueError(
                f"provisional case references another document: {wrong_document[0]}"
            )
    payload = {
        "schema_version": 1,
        "input_version": PROVISIONAL_INPUT_VERSION,
        "dataset_version_id": "phase4-provisional-navigation-v1",
        "source_work_package_sha256": package.work_package_sha256,
        "chunk_catalog_sha256": package.chunk_catalog_sha256,
        "cases": [item.model_dump(mode="json") for item in cases],
        "status": "provisional",
        "claims_allowed": False,
    }
    return ProvisionalEvaluationInput(**payload, input_sha256=stable_sha256(payload))


def default_provisional_config() -> ProvisionalSharedConfig:
    return ProvisionalSharedConfig()


def run_provisional_comparison(
    *,
    input_contract: ProvisionalEvaluationInput,
    chunks: Iterable[CandidateChunk],
    config: ProvisionalSharedConfig | None = None,
) -> tuple[tuple[ProvisionalVariantRun, ...], ProvisionalEvaluationReport]:
    """Run all strategies over identical cases, document scope and config."""

    config = config or default_provisional_config()
    chunk_values = tuple(chunks)
    documents = tuple(
        RetrievalDocument(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            text=item.text,
            section_path=item.section_path,
            page_start=item.page_start,
            page_end=item.page_end,
        )
        for item in chunk_values
    )
    provider = FakeEmbeddingProvider(
        dimensions=config.embedding_dimensions,
        model=config.embedding_model,
    )
    index_call = CallContext(call_id="phase4-provisional-index", timeout_seconds=3600)
    vectors_result = provider.embed(
        EmbeddingRequest(
            texts=tuple(document.text for document in documents),
            call=index_call,
        )
    )
    vectors = dict(zip((document.chunk_id for document in documents), vectors_result.vectors, strict=True))
    configs = {
        "bm25": config.bm25,
        "vector": config.vector,
        "hybrid_rrf": config.hybrid,
    }
    runs: list[ProvisionalVariantRun] = []
    for variant, variant_config in configs.items():
        retriever = build_retriever(
            variant_config,
            documents,
            embedding_provider=provider,
            vectors=vectors,
            index_call=index_call,
        )
        case_results: list[ProvisionalCaseResult] = []
        for case in input_contract.cases:
            call = CallContext(
                call_id=f"phase4-provisional-{variant}-{case.case_id}",
                timeout_seconds=3600,
            )
            request = SearchRequest(
                query=case.query,
                document_ids=(case.document_id,),
                limit=config.search_limit,
                call=call,
            )
            started = time.perf_counter()
            result = retriever.search(request)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            ranked = RankedSearchResult(
                retriever=result.retriever,
                hits=tuple(
                    RankedChunkHit(
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        rank=hit.rank,
                        score=hit.score,
                        source=hit.source,
                        text=hit.text,
                    )
                    for hit in result.hits
                ),
            )
            case_results.append(
                ProvisionalCaseResult(
                    case_id=case.case_id,
                    query_id=case.query_id,
                    result=ranked,
                    latency_ms=elapsed_ms,
                    navigation_hint_chunk_ids=case.candidate_chunk_ids,
                )
            )
        run_payload = {
            "schema_version": 1,
            "artifact_version": "phase4-provisional-run-v1",
            "input_sha256": input_contract.input_sha256,
            "dataset_version_id": input_contract.dataset_version_id,
            "source_work_package_sha256": input_contract.source_work_package_sha256,
            "chunk_catalog_sha256": input_contract.chunk_catalog_sha256,
            "shared_config_sha256": config.config_sha256,
            "implementation_version_sha256": PROVISIONAL_IMPLEMENTATION_SHA256,
            "variant": variant,
            "retriever_config_sha256": stable_sha256(variant_config.model_dump(mode="json")),
            "cases": [item.model_dump(mode="json") for item in case_results],
        }
        runs.append(ProvisionalVariantRun(**run_payload, results_sha256=stable_sha256(run_payload)))

    diagnostics = tuple(_diagnostics(run) for run in runs)
    recommendation = _recommend(diagnostics)
    report_payload = {
        "schema_version": 1,
        "report_version": PROVISIONAL_REPORT_VERSION,
        "report_kind": "provisional",
        "claims_allowed": False,
        "input_sha256": input_contract.input_sha256,
        "dataset_version_id": input_contract.dataset_version_id,
        "source_work_package_sha256": input_contract.source_work_package_sha256,
        "chunk_catalog_sha256": input_contract.chunk_catalog_sha256,
        "shared_config_sha256": config.config_sha256,
        "implementation_version_sha256": PROVISIONAL_IMPLEMENTATION_SHA256,
        "required_human_cases": len(input_contract.cases),
        "human_annotation_cases": 0,
        "approved_human_cases": 0,
        "variants": [item.model_dump(mode="json") for item in diagnostics],
        "default_strategy_candidate": recommendation,
        "interpretation": (
            "Provisional navigation diagnostics only. Candidate chunks come from "
            "historical external-platform hints and are not human relevance labels. "
            "Recall, MRR, latency, accuracy, and production baseline claims must not "
            f"be made from this report. Human annotation/review remains 0/{len(input_contract.cases)}."
        ),
    }
    report = ProvisionalEvaluationReport(
        **report_payload,
        report_sha256=stable_sha256(report_payload),
    )
    return tuple(runs), report


def _diagnostics(run: ProvisionalVariantRun) -> ProvisionalVariantDiagnostics:
    overlap_counts: list[float] = []
    result_counts: list[int] = []
    empty: list[str] = []
    no_overlap: list[str] = []
    hint_cases: list[str] = []
    for case in run.cases:
        returned = {hit.chunk_id for hit in case.result.hits[:10]}
        hints = set(case.navigation_hint_chunk_ids)
        result_counts.append(len(case.result.hits))
        overlap = len(returned & hints) / len(hints) if hints else None
        if not case.result.hits:
            empty.append(case.case_id)
        if overlap is not None:
            hint_cases.append(case.case_id)
            overlap_counts.append(overlap)
            if not returned & hints:
                no_overlap.append(case.case_id)
    latencies = tuple(case.latency_ms for case in run.cases)
    return ProvisionalVariantDiagnostics(
        variant=run.variant,
        cases_evaluated=len(run.cases),
        cases_with_results=len(run.cases) - len(empty),
        empty_result_cases=len(empty),
        cases_with_navigation_hint_overlap_at_10=len(hint_cases) - len(no_overlap),
        mean_navigation_hint_overlap_at_10=(
            sum(overlap_counts) / len(overlap_counts) if overlap_counts else None
        ),
        navigation_hint_overlap_rate_at_10=(
            (len(hint_cases) - len(no_overlap)) / len(hint_cases)
            if overlap_counts
            else None
        ),
        result_count_mean=sum(result_counts) / len(result_counts) if result_counts else None,
        latency_samples=len(latencies),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        diagnostic_error_cases={
            "empty_result": tuple(empty[:25]),
            "no_navigation_hint_overlap_at_10": tuple(no_overlap[:25]),
        },
    )


def _recommend(diagnostics: Sequence[ProvisionalVariantDiagnostics]) -> str:
    if not diagnostics:
        return "none"
    ordered = sorted(
        diagnostics,
        key=lambda item: (
            -(item.mean_navigation_hint_overlap_at_10 or 0.0),
            item.latency_p95_ms if item.latency_p95_ms is not None else float("inf"),
            item.variant,
        ),
    )
    return ordered[0].variant


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, ceil(quantile * len(ordered)) - 1)])


def render_provisional_artifacts(
    *,
    input_contract: ProvisionalEvaluationInput,
    runs: Iterable[ProvisionalVariantRun],
    report: ProvisionalEvaluationReport,
    config: ProvisionalSharedConfig | None = None,
) -> dict[str, str]:
    """Render JSON artifacts and an explicit non-claimability error analysis."""

    normalized_runs = tuple(runs)
    errors = {
        "schema_version": 1,
        "report_kind": "provisional",
        "claims_allowed": False,
        "source_input_sha256": input_contract.input_sha256,
        "variants": {
            item.variant: item.diagnostic_error_cases
            for item in report.variants
        },
        "boundary": (
            "These are navigation diagnostics against historical candidate hints; "
            "they are not false-positive/false-negative analysis against human truth."
        ),
    }
    config = config or default_provisional_config()
    manifest = {
        "schema_version": 1,
        "report_kind": "provisional",
        "claims_allowed": False,
        "input_sha256": input_contract.input_sha256,
        "shared_config_sha256": config.config_sha256,
        "implementation_version_sha256": PROVISIONAL_IMPLEMENTATION_SHA256,
        "source_work_package_sha256": input_contract.source_work_package_sha256,
        "chunk_catalog_sha256": input_contract.chunk_catalog_sha256,
        "variant_results": {
            item.variant: item.results_sha256 for item in normalized_runs
        },
        "human_annotation_cases": 0,
        "required_human_cases": len(input_contract.cases),
        "command": "python -m tender_review.evaluation provisional --check",
    }
    artifacts = {
        "input.json": _render(input_contract),
        "config.json": _render(config),
        "bm25.run.json": _render(next(item for item in normalized_runs if item.variant == "bm25")),
        "vector.run.json": _render(next(item for item in normalized_runs if item.variant == "vector")),
        "hybrid_rrf.run.json": _render(next(item for item in normalized_runs if item.variant == "hybrid_rrf")),
        "report.json": _render(report),
        "error_analysis.json": _render(errors),
        "manifest.json": _render(manifest),
        "README.md": (
            "# Stage 4C Provisional Retrieval Comparison\n\n"
            "This artifact compares BM25, Vector-only, and Hybrid/RRF on the same "
            f"{len({item.document_id for item in input_contract.cases})}-document/"
            f"{len(input_contract.cases)}-case "
            "input and document-scoped query contract. Candidate "
            "chunks are historical navigation hints only; no human/reviewer fields "
            "were created. `claims_allowed` is permanently false for this run.\n\n"
            "The result and error files are diagnostics for choosing what to validate "
            "after human annotation. Their latency values are local run observations, "
            "not real production latency or a production baseline.\n"
        ),
    }
    checksums = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in sorted(artifacts.items())
    }
    artifacts["checksums.json"] = _render({"schema_version": 1, "files": checksums})
    return artifacts


def write_provisional_artifacts(directory: Path, artifacts: Mapping[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (directory / name).write_text(content, encoding="utf-8", newline="\n")


def validate_provisional_artifacts(directory: Path) -> None:
    required = {
        "input.json",
        "config.json",
        "bm25.run.json",
        "vector.run.json",
        "hybrid_rrf.run.json",
        "report.json",
        "error_analysis.json",
        "manifest.json",
        "checksums.json",
        "README.md",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise ValueError("missing provisional artifacts: " + ", ".join(missing))
    input_contract = ProvisionalEvaluationInput.model_validate(_read(directory / "input.json"))
    config = ProvisionalSharedConfig.model_validate(_read(directory / "config.json"))
    report = ProvisionalEvaluationReport.model_validate(_read(directory / "report.json"))
    if report.input_sha256 != input_contract.input_sha256:
        raise ValueError("provisional report targets another input")
    if report.shared_config_sha256 != config.config_sha256:
        raise ValueError("provisional report targets another shared config")
    if report.implementation_version_sha256 != PROVISIONAL_IMPLEMENTATION_SHA256:
        raise ValueError("provisional report targets another implementation version")
    for name, variant in (
        ("bm25.run.json", "bm25"),
        ("vector.run.json", "vector"),
        ("hybrid_rrf.run.json", "hybrid_rrf"),
    ):
        run = ProvisionalVariantRun.model_validate(_read(directory / name))
        if run.variant != variant or run.input_sha256 != input_contract.input_sha256:
            raise ValueError(f"invalid provisional artifact identity: {name}")
        if run.shared_config_sha256 != config.config_sha256:
            raise ValueError(f"invalid provisional shared config identity: {name}")
        if run.implementation_version_sha256 != PROVISIONAL_IMPLEMENTATION_SHA256:
            raise ValueError(f"invalid provisional implementation identity: {name}")
        if run.dataset_version_id != input_contract.dataset_version_id:
            raise ValueError(f"invalid provisional dataset version identity: {name}")
    checksums = _read(directory / "checksums.json")
    if not isinstance(checksums, dict) or not isinstance(checksums.get("files"), dict):
        raise ValueError("invalid provisional checksums.json")
    if "checksums.json" in checksums["files"]:
        raise ValueError("checksums.json must not contain its own checksum")
    for name, expected in checksums["files"].items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing provisional checksum target: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"provisional artifact checksum differs: {name}")


def _render(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
