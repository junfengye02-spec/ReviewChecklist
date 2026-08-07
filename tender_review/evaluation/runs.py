from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

from tender_review.rule_management.public import RuleVersionRepository
from tender_review.shared.clock import Clock
from tender_review.shared.contracts import ContractModel
from tender_review.shared.errors import ConflictError, NotFoundError, PermanentError
from tender_review.shared.ids import IdGenerator

from .annotation_workflow import (
    AnnotationDatasetRepository,
    AnnotationSampleStatus,
    DatasetSplit,
    DatasetStatus,
)


SHA256_PATTERN = r"^[0-9a-f]{64}$"
_A4_VERIFY_COMMAND = "python -m tender_review.evaluation a4-verify"


def stable_sha256(value: Any) -> str:
    rendered = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class EvaluationSourceType(str, Enum):
    REAL = "real"
    PROVISIONAL = "provisional"
    SYNTHETIC = "synthetic"
    EXTERNAL_PLATFORM = "external-platform"


class EvaluationPurpose(str, Enum):
    REAL_BASELINE = "REAL_BASELINE"
    RELEASE_GATE = "RELEASE_GATE"
    CANDIDATE_DIAGNOSTIC = "CANDIDATE_DIAGNOSTIC"


class EvaluationRunStatus(str, Enum):
    NOT_READY = "NOT_READY"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GateAssessmentStatus(str, Enum):
    NOT_READY = "NOT_READY"
    PASSED = "PASSED"
    FAILED = "FAILED"


class EvaluationDatasetSnapshot(ContractModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_type: EvaluationSourceType
    status: DatasetStatus
    provenance_status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    required_human_cases: int = Field(ge=0)
    independently_verified_cases: int = Field(ge=0)
    frozen_test_cases: int = Field(ge=0)

    @model_validator(mode="after")
    def claim_boundary_is_consistent(self) -> Self:
        if self.claims_allowed and not self.is_release_grade:
            raise ValueError(
                "claimable evaluation datasets must be real, verified, frozen, and fully reviewed"
            )
        return self

    @property
    def is_release_grade(self) -> bool:
        return (
            self.source_type is EvaluationSourceType.REAL
            and self.status is DatasetStatus.FROZEN
            and self.provenance_status == "verified"
            and self.required_human_cases > 0
            and self.independently_verified_cases == self.required_human_cases
            and self.frozen_test_cases > 0
        )


class EvaluationRunBinding(ContractModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_split: DatasetSplit
    rule_version_id: str = Field(min_length=1, max_length=128)
    rule_version_sha256: str = Field(pattern=SHA256_PATTERN)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    code_sha256: str = Field(pattern=SHA256_PATTERN)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    binding_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def binding_hash_matches(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if self.binding_sha256 != stable_sha256(payload):
            raise ValueError("binding_sha256 does not match evaluation inputs")
        return self


class RetrievalMetrics(ContractModel):
    evidence_recall_at_5: float | None = Field(default=None, ge=0, le=1)
    evidence_recall_at_10: float | None = Field(default=None, ge=0, le=1)
    mrr: float | None = Field(default=None, ge=0, le=1)
    cross_section_bilateral_hit_rate: float | None = Field(default=None, ge=0, le=1)
    no_answer_false_retrieval_rate: float | None = Field(default=None, ge=0, le=1)


class ReviewMetrics(ContractModel):
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)
    false_positive_rate: float | None = Field(default=None, ge=0, le=1)
    false_negative_rate: float | None = Field(default=None, ge=0, le=1)
    evidence_completeness_rate: float | None = Field(default=None, ge=0, le=1)
    evidence_conclusion_consistency_rate: float | None = Field(default=None, ge=0, le=1)


class StabilityMetrics(ContractModel):
    repeated_run_consistency_rate: float | None = Field(default=None, ge=0, le=1)
    model_exception_rate: float | None = Field(default=None, ge=0, le=1)
    human_handoff_rate: float | None = Field(default=None, ge=0, le=1)


class EngineeringMetrics(ContractModel):
    task_success_rate: float | None = Field(default=None, ge=0, le=1)
    worker_recovery_success_rate: float | None = Field(default=None, ge=0, le=1)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    token_usage: int | None = Field(default=None, ge=0)
    cost_per_document: float | None = Field(default=None, ge=0)


class EvaluationMetrics(ContractModel):
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    review: ReviewMetrics = Field(default_factory=ReviewMetrics)
    stability: StabilityMetrics = Field(default_factory=StabilityMetrics)
    engineering: EngineeringMetrics = Field(default_factory=EngineeringMetrics)

    def values(self) -> dict[str, int | float | None]:
        values: dict[str, int | float | None] = {}
        for group_name in ("retrieval", "review", "stability", "engineering"):
            group = getattr(self, group_name)
            for name, value in group.model_dump(
                mode="python", exclude={"schema_version"}
            ).items():
                values[f"{group_name}.{name}"] = value
        return values

    @property
    def is_complete(self) -> bool:
        return all(value is not None for value in self.values().values())


class EvaluationFailureSample(ContractModel):
    sample_id: str = Field(min_length=1, max_length=128)
    stage: Literal["retrieval", "review", "stability", "engineering"]
    category: str = Field(min_length=1, max_length=128)
    expected_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    actual_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evidence_sha256s: tuple[str, ...] = ()
    detail: str = Field(min_length=1, max_length=4000)

    @field_validator("evidence_sha256s")
    @classmethod
    def evidence_hashes_are_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_sha256s must be unique")
        if any(len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in value):
            raise ValueError("evidence_sha256s must contain SHA-256 digests")
        return value


class EvaluationDifferenceSource(ContractModel):
    source: Literal[
        "data", "retrieval", "review", "model", "prompt", "config", "code", "infrastructure"
    ]
    sample_ids: tuple[str, ...] = ()
    detail: str = Field(min_length=1, max_length=4000)


class EvaluationResult(ContractModel):
    binding_sha256: str = Field(pattern=SHA256_PATTERN)
    metrics: EvaluationMetrics
    failure_samples: tuple[EvaluationFailureSample, ...] = ()
    difference_sources: tuple[EvaluationDifferenceSource, ...] = ()
    case_results_sha256: str = Field(pattern=SHA256_PATTERN)
    repeated_runs_sha256: str = Field(pattern=SHA256_PATTERN)
    engineering_telemetry_sha256: str = Field(pattern=SHA256_PATTERN)
    result_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def result_hash_matches(self) -> Self:
        sample_ids = tuple(item.sample_id for item in self.failure_samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("failure sample IDs must be unique")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if self.result_sha256 != stable_sha256(payload):
            raise ValueError("result_sha256 does not match evaluation result")
        return self


class ThresholdRule(ContractModel):
    metric_id: str = Field(min_length=1, max_length=128)
    operator: Literal["gte", "lte"]
    threshold: float = Field(ge=0)
    baseline_value: float = Field(ge=0)


class FrozenThresholdPolicy(ContractModel):
    policy_id: str = Field(min_length=1, max_length=128)
    baseline_run_id: str = Field(min_length=1, max_length=128)
    baseline_report_sha256: str = Field(pattern=SHA256_PATTERN)
    approved_by: str = Field(min_length=1, max_length=255)
    frozen_at: datetime
    rules: tuple[ThresholdRule, ...] = Field(min_length=1)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def policy_is_frozen_and_hashed(self) -> Self:
        if self.frozen_at.tzinfo is None:
            raise ValueError("frozen_at must be timezone-aware")
        if not self.approved_by.strip():
            raise ValueError("approved_by must name a human approver")
        metric_ids = tuple(item.metric_id for item in self.rules)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("threshold metric IDs must be unique")
        payload = self.model_dump(mode="json", exclude={"policy_sha256"})
        if self.policy_sha256 != stable_sha256(payload):
            raise ValueError("policy_sha256 does not match frozen threshold policy")
        return self


class MetricDifference(ContractModel):
    metric_id: str
    baseline_value: float
    candidate_value: float
    delta: float
    threshold: float
    operator: Literal["gte", "lte"]
    passed: bool


class ReleaseGateAssessment(ContractModel):
    status: GateAssessmentStatus
    eligible: bool
    passed: bool
    threshold_policy_id: str | None = None
    threshold_policy_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def assessment_is_consistent(self) -> Self:
        if self.passed and (not self.eligible or self.status is not GateAssessmentStatus.PASSED):
            raise ValueError("a passed release gate must be eligible")
        if not self.passed and self.status is GateAssessmentStatus.PASSED:
            raise ValueError("PASSED status requires passed=true")
        if self.passed and (not self.threshold_policy_id or not self.threshold_policy_sha256):
            raise ValueError("a passed release gate requires a frozen threshold policy")
        if self.status is GateAssessmentStatus.NOT_READY and not self.blockers:
            raise ValueError("NOT_READY assessments require blockers")
        return self


class EvaluationReport(ContractModel):
    run_id: str = Field(min_length=1, max_length=128)
    purpose: EvaluationPurpose
    source_type: EvaluationSourceType
    status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    dataset: EvaluationDatasetSnapshot
    binding: EvaluationRunBinding
    metrics: EvaluationMetrics
    baseline_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    metric_differences: tuple[MetricDifference, ...] = ()
    failure_samples: tuple[EvaluationFailureSample, ...] = ()
    difference_sources: tuple[EvaluationDifferenceSource, ...] = ()
    release_gate: ReleaseGateAssessment
    limitations: tuple[str, ...] = Field(min_length=1)
    generated_at: datetime
    result_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def report_boundary_and_hash_match(self) -> Self:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        claimable = (
            self.source_type is EvaluationSourceType.REAL
            and self.status == "verified"
            and self.dataset.claims_allowed
            and self.dataset.is_release_grade
            and self.binding.dataset_manifest_sha256 == self.dataset.manifest_sha256
            and self.result_sha256 is not None
            and self.metrics.is_complete
        )
        if self.claims_allowed != claimable:
            raise ValueError("claims_allowed does not match real verified report provenance")
        if self.release_gate.passed and (
            not claimable or self.purpose is not EvaluationPurpose.RELEASE_GATE
        ):
            raise ValueError("only a claimable release-gate report can pass publication")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match evaluation report")
        return self


class EvaluationRun(ContractModel):
    run_id: str = Field(min_length=1, max_length=128)
    purpose: EvaluationPurpose
    status: EvaluationRunStatus
    source_type: EvaluationSourceType
    provenance_status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    dataset: EvaluationDatasetSnapshot
    binding: EvaluationRunBinding
    model_config_id: str = Field(min_length=1, max_length=128)
    retriever_version: str = Field(min_length=1, max_length=128)
    evaluator_version: str = Field(min_length=1, max_length=128)
    reproducibility_command: str = Field(min_length=1, max_length=4000)
    blockers: tuple[str, ...] = ()
    result_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    report_sha256: str = Field(pattern=SHA256_PATTERN)
    started_at: datetime
    completed_at: datetime | None = None
    run_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("reproducibility_command")
    @classmethod
    def command_is_supported(cls, value: str) -> str:
        if not value.strip().startswith(_A4_VERIFY_COMMAND) or "--run" not in value or "--report" not in value:
            raise ValueError("reproducibility_command must use the supported A4 verifier")
        return value.strip()

    @property
    def rule_identity(self) -> tuple[str, str]:
        return self.binding.rule_version_id, self.binding.dataset_version_id

    @model_validator(mode="after")
    def run_state_and_hash_match(self) -> Self:
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        if self.status is EvaluationRunStatus.NOT_READY:
            if not self.blockers or self.result_sha256 is not None or self.claims_allowed:
                raise ValueError("NOT_READY runs require blockers and cannot contain results")
        if self.status is EvaluationRunStatus.COMPLETED:
            if self.completed_at is None or self.result_sha256 is None:
                raise ValueError("completed evaluation runs require a result and completion time")
        if self.status in {EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING} and self.completed_at:
            raise ValueError("unfinished evaluation runs cannot have completed_at")
        if self.claims_allowed and (
            self.status is not EvaluationRunStatus.COMPLETED
            or self.source_type is not EvaluationSourceType.REAL
            or self.provenance_status != "verified"
            or not self.dataset.claims_allowed
        ):
            raise ValueError("only completed real verified runs may allow claims")
        payload = self.model_dump(mode="json", exclude={"run_sha256"})
        if self.run_sha256 != stable_sha256(payload):
            raise ValueError("run_sha256 does not match evaluation run")
        return self


class CreateEvaluationRun(ContractModel):
    rule_version_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    purpose: EvaluationPurpose
    dataset_split: DatasetSplit
    model_config_id: str = Field(min_length=1, max_length=128)
    retriever_version: str = Field(min_length=1, max_length=128)
    evaluator_version: str = Field(min_length=1, max_length=128)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    code_sha256: str = Field(pattern=SHA256_PATTERN)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    reproducibility_command: str = Field(min_length=1, max_length=4000)

    @field_validator("reproducibility_command")
    @classmethod
    def command_is_supported(cls, value: str) -> str:
        if not value.strip().startswith(_A4_VERIFY_COMMAND) or "--run" not in value or "--report" not in value:
            raise ValueError("reproducibility_command must use the supported A4 verifier")
        return value.strip()


class CompleteEvaluationRun(ContractModel):
    run_id: str = Field(min_length=1, max_length=128)
    result: EvaluationResult
    threshold_policy_id: str | None = Field(default=None, min_length=1, max_length=128)


class FreezeThresholdPolicy(ContractModel):
    baseline_run_id: str = Field(min_length=1, max_length=128)
    approved_by: str = Field(min_length=1, max_length=255)
    rules: tuple[ThresholdRule, ...] = Field(min_length=1)


@runtime_checkable
class EvaluationDatasetResolver(Protocol):
    def resolve(self, dataset_version_id: str) -> EvaluationDatasetSnapshot: ...


@runtime_checkable
class EvaluationRunRepository(Protocol):
    def add(self, run: EvaluationRun, report: EvaluationReport) -> EvaluationRun: ...

    def get(self, run_id: str) -> EvaluationRun: ...

    def get_report(self, run_id: str) -> EvaluationReport: ...

    def list(self, limit: int = 100) -> tuple[EvaluationRun, ...]: ...

    def complete(self, run: EvaluationRun, report: EvaluationReport) -> EvaluationRun: ...

    def add_policy(self, policy: FrozenThresholdPolicy) -> FrozenThresholdPolicy: ...

    def get_policy(self, policy_id: str) -> FrozenThresholdPolicy: ...


class AnnotationEvaluationDatasetResolver:
    def __init__(self, repository: AnnotationDatasetRepository) -> None:
        self._repository = repository

    def resolve(self, dataset_version_id: str) -> EvaluationDatasetSnapshot:
        version = self._repository.get_version(dataset_version_id)
        verified = sum(
            item.status in {AnnotationSampleStatus.VERIFIED, AnnotationSampleStatus.FROZEN}
            and item.annotation is not None
            and item.review is not None
            and item.annotation.actor_id != item.review.actor_id
            for item in version.samples
        )
        return EvaluationDatasetSnapshot(
            dataset_version_id=version.dataset_version_id,
            manifest_sha256=version.manifest_sha256,
            source_type=EvaluationSourceType.REAL,
            status=version.status,
            provenance_status=version.provenance.status,
            claims_allowed=version.provenance.claims_allowed,
            required_human_cases=version.required_human_cases,
            independently_verified_cases=verified,
            frozen_test_cases=sum(item.split is DatasetSplit.FROZEN_TEST for item in version.samples),
        )


class InMemoryEvaluationRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, EvaluationRun] = {}
        self._reports: dict[str, EvaluationReport] = {}
        self._policies: dict[str, FrozenThresholdPolicy] = {}
        self._lock = RLock()

    def add(self, run: EvaluationRun, report: EvaluationReport) -> EvaluationRun:
        with self._lock:
            if run.run_id in self._runs:
                raise ConflictError("evaluation run already exists", code="evaluation_run_conflict")
            self._runs[run.run_id] = run
            self._reports[run.run_id] = report
            return run

    def get(self, run_id: str) -> EvaluationRun:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise NotFoundError("evaluation run does not exist", code="evaluation_run_not_found") from exc

    def get_report(self, run_id: str) -> EvaluationReport:
        with self._lock:
            try:
                return self._reports[run_id]
            except KeyError as exc:
                raise NotFoundError("evaluation report does not exist", code="evaluation_report_not_found") from exc

    def list(self, limit: int = 100) -> tuple[EvaluationRun, ...]:
        with self._lock:
            ordered = sorted(self._runs.values(), key=lambda item: item.started_at, reverse=True)
            return tuple(ordered[: max(1, min(limit, 500))])

    def complete(self, run: EvaluationRun, report: EvaluationReport) -> EvaluationRun:
        with self._lock:
            current = self.get(run.run_id)
            if current.run_sha256 == run.run_sha256:
                return current
            if current.status not in {EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING}:
                raise ConflictError("evaluation run is immutable", code="evaluation_run_immutable")
            if current.binding != run.binding:
                raise ConflictError("evaluation run binding changed", code="evaluation_binding_changed")
            self._runs[run.run_id] = run
            self._reports[run.run_id] = report
            return run

    def add_policy(self, policy: FrozenThresholdPolicy) -> FrozenThresholdPolicy:
        with self._lock:
            if policy.policy_id in self._policies or any(
                item.baseline_run_id == policy.baseline_run_id for item in self._policies.values()
            ):
                raise ConflictError("threshold policy already exists", code="threshold_policy_conflict")
            self._policies[policy.policy_id] = policy
            return policy

    def get_policy(self, policy_id: str) -> FrozenThresholdPolicy:
        with self._lock:
            try:
                return self._policies[policy_id]
            except KeyError as exc:
                raise NotFoundError("threshold policy does not exist", code="threshold_policy_not_found") from exc


class EvaluationRunService:
    def __init__(
        self,
        repository: EvaluationRunRepository,
        datasets: EvaluationDatasetResolver,
        rules: RuleVersionRepository,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._datasets = datasets
        self._rules = rules
        self._ids = ids
        self._clock = clock

    def create(self, command: CreateEvaluationRun) -> EvaluationRun:
        dataset = self._datasets.resolve(command.dataset_version_id)
        rule = self._rules.get_version(command.rule_version_id)
        binding_payload = {
            "schema_version": 1,
            "dataset_version_id": dataset.dataset_version_id,
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "dataset_split": command.dataset_split,
            "rule_version_id": rule.rule_version_id,
            "rule_version_sha256": rule.content_sha256,
            "input_sha256": command.input_sha256,
            "config_sha256": command.config_sha256,
            "code_sha256": command.code_sha256,
            "model_sha256": command.model_sha256,
            "prompt_sha256": command.prompt_sha256,
        }
        binding = EvaluationRunBinding(
            **binding_payload, binding_sha256=stable_sha256(binding_payload)
        )
        blockers = self._dataset_blockers(dataset, command.purpose, command.dataset_split)
        now = self._clock.now()
        run_id = self._ids.new()
        report = self._not_ready_report(run_id, command.purpose, dataset, binding, blockers, now)
        run_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "purpose": command.purpose,
            "status": EvaluationRunStatus.NOT_READY if blockers else EvaluationRunStatus.PENDING,
            "source_type": dataset.source_type,
            "provenance_status": dataset.provenance_status,
            "claims_allowed": False,
            "dataset": dataset.model_dump(mode="json"),
            "binding": binding.model_dump(mode="json"),
            "model_config_id": command.model_config_id,
            "retriever_version": command.retriever_version,
            "evaluator_version": command.evaluator_version,
            "reproducibility_command": command.reproducibility_command,
            "blockers": blockers,
            "result_sha256": None,
            "report_sha256": report.report_sha256,
            "started_at": now,
            "completed_at": None,
        }
        run = EvaluationRun(**run_payload, run_sha256=stable_sha256(run_payload))
        return self._repository.add(run, report)

    def get(self, run_id: str) -> EvaluationRun:
        return self._repository.get(run_id.strip())

    def get_report(self, run_id: str) -> EvaluationReport:
        return self._repository.get_report(run_id.strip())

    def list(self, limit: int = 100) -> tuple[EvaluationRun, ...]:
        return self._repository.list(limit)

    def complete(self, command: CompleteEvaluationRun) -> EvaluationRun:
        current = self._repository.get(command.run_id)
        if current.status not in {EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING}:
            raise ConflictError("evaluation run cannot be completed", code="evaluation_run_state_invalid")
        if command.result.binding_sha256 != current.binding.binding_sha256:
            raise PermanentError("evaluation result targets another binding", code="evaluation_binding_mismatch")
        dataset = self._datasets.resolve(current.dataset.dataset_version_id)
        if dataset != current.dataset or dataset.manifest_sha256 != current.binding.dataset_manifest_sha256:
            raise PermanentError("evaluation dataset changed after run creation", code="evaluation_dataset_changed")
        blockers = self._dataset_blockers(dataset, current.purpose, current.binding.dataset_split)
        if blockers:
            raise PermanentError("evaluation dataset is not release-grade", code="evaluation_dataset_not_ready", details={"blockers": blockers})
        if not command.result.metrics.is_complete:
            raise PermanentError("real evaluation requires every A4 metric", code="evaluation_metrics_incomplete")

        policy: FrozenThresholdPolicy | None = None
        differences: tuple[MetricDifference, ...] = ()
        gate_blockers: tuple[str, ...] = ()
        passed = False
        if current.purpose is EvaluationPurpose.RELEASE_GATE:
            if command.threshold_policy_id is None:
                gate_blockers = ("frozen threshold policy is missing",)
            else:
                policy = self._repository.get_policy(command.threshold_policy_id)
                differences = self._compare(command.result.metrics, policy)
                missing = sorted(set(command.result.metrics.values()) - {item.metric_id for item in policy.rules})
                if missing:
                    gate_blockers = ("threshold policy does not cover every A4 metric",)
                else:
                    passed = all(item.passed for item in differences)
                    if not passed:
                        gate_blockers = ("one or more frozen thresholds were not met",)

        gate = ReleaseGateAssessment(
            status=(GateAssessmentStatus.PASSED if passed else GateAssessmentStatus.FAILED),
            eligible=True,
            passed=passed,
            threshold_policy_id=policy.policy_id if policy else None,
            threshold_policy_sha256=policy.policy_sha256 if policy else None,
            blockers=gate_blockers,
        )
        now = self._clock.now()
        report_payload = {
            "schema_version": 1,
            "run_id": current.run_id,
            "purpose": current.purpose,
            "source_type": EvaluationSourceType.REAL,
            "status": "verified",
            "claims_allowed": True,
            "dataset": dataset.model_dump(mode="json"),
            "binding": current.binding.model_dump(mode="json"),
            "metrics": command.result.metrics.model_dump(mode="json"),
            "baseline_run_id": policy.baseline_run_id if policy else None,
            "metric_differences": [item.model_dump(mode="json") for item in differences],
            "failure_samples": [item.model_dump(mode="json") for item in command.result.failure_samples],
            "difference_sources": [item.model_dump(mode="json") for item in command.result.difference_sources],
            "release_gate": gate.model_dump(mode="json"),
            "limitations": (
                "Metrics apply only to the bound frozen dataset, input, config, code, model, and prompt.",
                "Publication eligibility is separate from metric claimability and requires every frozen threshold.",
            ),
            "generated_at": now,
            "result_sha256": command.result.result_sha256,
        }
        report = EvaluationReport(**report_payload, report_sha256=stable_sha256(report_payload))
        run_payload = current.model_dump(mode="json", exclude={"run_sha256"})
        run_payload.update({
            "status": EvaluationRunStatus.COMPLETED,
            "source_type": EvaluationSourceType.REAL,
            "provenance_status": "verified",
            "claims_allowed": True,
            "blockers": (),
            "result_sha256": command.result.result_sha256,
            "report_sha256": report.report_sha256,
            "completed_at": now,
        })
        run = EvaluationRun(**run_payload, run_sha256=stable_sha256(run_payload))
        return self._repository.complete(run, report)

    def freeze_threshold_policy(self, command: FreezeThresholdPolicy) -> FrozenThresholdPolicy:
        run = self._repository.get(command.baseline_run_id)
        report = self._repository.get_report(command.baseline_run_id)
        if run.purpose is not EvaluationPurpose.REAL_BASELINE or run.status is not EvaluationRunStatus.COMPLETED:
            raise PermanentError("thresholds require a completed real baseline", code="threshold_baseline_invalid")
        if not report.claims_allowed or report.source_type is not EvaluationSourceType.REAL or report.status != "verified":
            raise PermanentError("thresholds require a claimable real baseline", code="threshold_baseline_not_claimable")
        values = report.metrics.values()
        expected = set(values)
        supplied = {item.metric_id for item in command.rules}
        if supplied != expected:
            raise PermanentError("threshold policy must cover every A4 metric", code="threshold_policy_incomplete")
        for rule in command.rules:
            value = values[rule.metric_id]
            if value is None or float(value) != rule.baseline_value:
                raise PermanentError("threshold baseline value differs from report", code="threshold_baseline_value_mismatch")
        payload = {
            "schema_version": 1,
            "policy_id": self._ids.new(),
            "baseline_run_id": run.run_id,
            "baseline_report_sha256": report.report_sha256,
            "approved_by": command.approved_by.strip(),
            "frozen_at": self._clock.now(),
            "rules": [item.model_dump(mode="json") for item in command.rules],
        }
        return self._repository.add_policy(
            FrozenThresholdPolicy(**payload, policy_sha256=stable_sha256(payload))
        )

    def assert_dataset_release_ready(self, dataset_version_id: str) -> None:
        dataset = self._datasets.resolve(dataset_version_id)
        blockers = self._dataset_blockers(dataset, EvaluationPurpose.RELEASE_GATE, DatasetSplit.FROZEN_TEST)
        if blockers:
            raise PermanentError("release evaluation dataset is not ready", code="release_dataset_not_ready", details={"blockers": blockers})

    def assert_release_eligible(
        self,
        *,
        rule_version_id: str,
        dataset_version_id: str,
        evaluation_run_id: str,
        report_sha256: str,
    ) -> EvaluationReport:
        run = self._repository.get(evaluation_run_id)
        report = self._repository.get_report(evaluation_run_id)
        dataset = self._datasets.resolve(dataset_version_id)
        valid = (
            run.rule_identity == (rule_version_id, dataset_version_id)
            and run.status is EvaluationRunStatus.COMPLETED
            and run.purpose is EvaluationPurpose.RELEASE_GATE
            and run.source_type is EvaluationSourceType.REAL
            and run.provenance_status == "verified"
            and run.claims_allowed
            and run.report_sha256 == report_sha256 == report.report_sha256
            and report.claims_allowed
            and report.status == "verified"
            and report.source_type is EvaluationSourceType.REAL
            and report.release_gate.passed
            and dataset == run.dataset == report.dataset
            and dataset.status is DatasetStatus.FROZEN
            and dataset.claims_allowed
        )
        if not valid:
            raise PermanentError("persisted A4 report is not release-eligible", code="release_gate_not_eligible")
        return report

    @staticmethod
    def _dataset_blockers(
        dataset: EvaluationDatasetSnapshot,
        purpose: EvaluationPurpose,
        split: DatasetSplit,
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        if dataset.source_type is not EvaluationSourceType.REAL:
            blockers.append("dataset source_type is not real")
        if dataset.status is not DatasetStatus.FROZEN:
            blockers.append("dataset status is not FROZEN")
        if dataset.provenance_status != "verified":
            blockers.append("dataset provenance status is not verified")
        if not dataset.claims_allowed:
            blockers.append("dataset claims_allowed is false")
        if dataset.required_human_cases < 1:
            blockers.append("dataset requires no human-reviewed cases")
        if dataset.independently_verified_cases != dataset.required_human_cases:
            blockers.append(
                "independently verified cases do not match the required case count"
            )
        if dataset.frozen_test_cases < 1:
            blockers.append("dataset has no frozen-test cases")
        if purpose is EvaluationPurpose.RELEASE_GATE and split is not DatasetSplit.FROZEN_TEST:
            blockers.append("release gates must use the frozen-test split")
        if purpose is EvaluationPurpose.CANDIDATE_DIAGNOSTIC and split is DatasetSplit.FROZEN_TEST:
            blockers.append("frozen-test is forbidden for candidate generation or diagnostics")
        return tuple(blockers)

    @staticmethod
    def _not_ready_report(
        run_id: str,
        purpose: EvaluationPurpose,
        dataset: EvaluationDatasetSnapshot,
        binding: EvaluationRunBinding,
        blockers: tuple[str, ...],
        now: datetime,
    ) -> EvaluationReport:
        effective = blockers or ("evaluation has not completed; metrics are not available",)
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "purpose": purpose,
            "source_type": dataset.source_type,
            "status": "provisional" if dataset.provenance_status != "unknown" else "unknown",
            "claims_allowed": False,
            "dataset": dataset.model_dump(mode="json"),
            "binding": binding.model_dump(mode="json"),
            "metrics": EvaluationMetrics().model_dump(mode="json"),
            "baseline_run_id": None,
            "metric_differences": (),
            "failure_samples": (),
            "difference_sources": (),
            "release_gate": ReleaseGateAssessment(
                status=GateAssessmentStatus.NOT_READY,
                eligible=False,
                passed=False,
                blockers=effective,
            ).model_dump(mode="json"),
            "limitations": effective,
            "generated_at": now,
            "result_sha256": None,
        }
        return EvaluationReport(**payload, report_sha256=stable_sha256(payload))

    @staticmethod
    def _compare(
        metrics: EvaluationMetrics, policy: FrozenThresholdPolicy
    ) -> tuple[MetricDifference, ...]:
        values = metrics.values()
        differences: list[MetricDifference] = []
        for rule in sorted(policy.rules, key=lambda item: item.metric_id):
            raw = values.get(rule.metric_id)
            if raw is None:
                continue
            candidate = float(raw)
            passed = candidate >= rule.threshold if rule.operator == "gte" else candidate <= rule.threshold
            differences.append(MetricDifference(
                metric_id=rule.metric_id,
                baseline_value=rule.baseline_value,
                candidate_value=candidate,
                delta=candidate - rule.baseline_value,
                threshold=rule.threshold,
                operator=rule.operator,
                passed=passed,
            ))
        return tuple(differences)
