from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import to_jsonable_python

from tender_review.shared.contracts import ContractModel


SHA256_PATTERN = r"^[0-9a-f]{64}$"


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_value(value: str, field_name: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if canonical_json(parsed) != value:
        raise ValueError(f"{field_name} must use canonical JSON")
    return parsed


def _canonical_object(value: str, field_name: str) -> dict[str, Any]:
    parsed = _canonical_value(value, field_name)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return parsed


class OptimizationStatus(str, Enum):
    NOT_READY = "NOT_READY"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_HUMAN = "WAITING_HUMAN"
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    CANCELLED = "CANCELLED"


class AttemptStatus(str, Enum):
    STARTED = "STARTED"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WAITING_HUMAN = "WAITING_HUMAN"


class RootCause(str, Enum):
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    RULE_GAP = "RULE_GAP"
    MODEL_INSTABILITY = "MODEL_INSTABILITY"
    LABEL_UNCERTAIN = "LABEL_UNCERTAIN"


class CandidateType(str, Enum):
    RETRIEVAL_CONFIG = "RETRIEVAL_CONFIG"
    EXTRACTION_PROMPT_SCHEMA = "EXTRACTION_PROMPT_SCHEMA"
    TOOL_CONFIG = "TOOL_CONFIG"
    RULE_CONTENT = "RULE_CONTENT"
    STABILITY_CONFIG = "STABILITY_CONFIG"


class SampleRole(str, Enum):
    TARGET = "TARGET"
    PROTECTION = "PROTECTION"


class SourceType(str, Enum):
    REAL = "REAL"
    EXTERNAL_PLATFORM = "EXTERNAL_PLATFORM"
    PROVISIONAL = "PROVISIONAL"
    SYNTHETIC = "SYNTHETIC"


class RegressionStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PROVISIONAL = "PROVISIONAL"


class OptimizationReadinessStatus(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    BLOCKED = "BLOCKED"


class OptimizationTraceOutcome(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


ROOT_CAUSE_CANDIDATE_TYPES: dict[RootCause, CandidateType | None] = {
    RootCause.RETRIEVAL_MISS: CandidateType.RETRIEVAL_CONFIG,
    RootCause.EXTRACTION_ERROR: CandidateType.EXTRACTION_PROMPT_SCHEMA,
    RootCause.TOOL_ERROR: CandidateType.TOOL_CONFIG,
    RootCause.RULE_GAP: CandidateType.RULE_CONTENT,
    RootCause.MODEL_INSTABILITY: CandidateType.STABILITY_CONFIG,
    RootCause.LABEL_UNCERTAIN: None,
}


ALLOWED_CHANGE_PATHS: dict[CandidateType, tuple[tuple[str, str], ...]] = {
    CandidateType.RETRIEVAL_CONFIG: (
        ("execution_config", "$.retrieval.query"),
        ("execution_config", "$.retrieval.chunk"),
        ("execution_config", "$.retrieval.fusion"),
    ),
    CandidateType.EXTRACTION_PROMPT_SCHEMA: (
        ("execution_config", "$.extraction.prompt"),
        ("execution_config", "$.extraction.schema"),
    ),
    CandidateType.TOOL_CONFIG: (("execution_config", "$.tools"),),
    CandidateType.RULE_CONTENT: (
        ("content", "$.rules"),
        ("content", "$.rule_text"),
    ),
    CandidateType.STABILITY_CONFIG: (
        ("execution_config", "$.model.temperature"),
        ("execution_config", "$.model.seed"),
        ("execution_config", "$.model.retry"),
    ),
}


TERMINAL_OPTIMIZATION_STATUSES = {
    OptimizationStatus.NOT_READY,
    OptimizationStatus.BLOCKED,
    OptimizationStatus.WAITING_APPROVAL,
    OptimizationStatus.WAITING_HUMAN,
    OptimizationStatus.OPTIMIZATION_FAILED,
    OptimizationStatus.CANCELLED,
}


class SourceArtifact(ContractModel):
    path: str = Field(min_length=1, max_length=2048)
    sha256: str = Field(pattern=SHA256_PATTERN)
    kind: Literal["approval_opinion", "platform_run", "retrieval_report", "manifest"]


class OptimizationProvenance(ContractModel):
    source_type: SourceType
    status: Literal["provisional", "verified"]
    claims_allowed: bool
    source_description: str = Field(min_length=1, max_length=8000)
    source_artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    human_annotation_cases: int = Field(default=0, ge=0)
    required_human_cases: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def claims_match_source(self) -> Self:
        if self.status == "provisional" and self.claims_allowed:
            raise ValueError("provisional optimization provenance cannot allow claims")
        if self.source_type is not SourceType.REAL and (
            self.status != "provisional" or self.claims_allowed
        ):
            raise ValueError("non-real optimization sources must remain provisional")
        if self.human_annotation_cases > self.required_human_cases:
            raise ValueError("human annotation count cannot exceed required cases")
        if self.claims_allowed and (
            self.source_type is not SourceType.REAL
            or self.status != "verified"
            or self.human_annotation_cases != self.required_human_cases
        ):
            raise ValueError(
                "claimable optimization provenance requires all cases to be independently verified"
            )
        return self


class ExecutionHashes(ContractModel):
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    rule_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    retriever_sha256: str = Field(pattern=SHA256_PATTERN)
    tool_sha256: str = Field(pattern=SHA256_PATTERN)


class FailureSignals(ContractModel):
    failure_summary: str = Field(min_length=1, max_length=8000)
    label_conflict: bool | None = None
    evidence_conflict: bool | None = None
    evidence_in_top_k: bool | None = None
    extraction_matches_expected: bool | None = None
    tool_matches_expected: bool | None = None
    repeated_outputs_consistent: bool | None = None


class OptimizationSample(ContractModel):
    sample_id: str = Field(min_length=1, max_length=256)
    role: SampleRole
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    source_type: SourceType
    provenance_status: Literal["provisional", "verified"]
    claims_allowed: bool
    source_reference: str = Field(min_length=1, max_length=2048)
    review_input_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    finding_id: str | None = Field(default=None, min_length=1, max_length=128)
    human_decision_id: str | None = Field(default=None, min_length=1, max_length=128)
    signals: FailureSignals | None = None

    @model_validator(mode="after")
    def source_and_role_are_consistent(self) -> Self:
        if self.source_type is not SourceType.REAL and (
            self.provenance_status != "provisional" or self.claims_allowed
        ):
            raise ValueError("non-real samples must remain non-claimable and provisional")
        if self.source_type is SourceType.REAL and (
            self.human_decision_id is None or self.finding_id is None
        ):
            raise ValueError("REAL optimization samples require a finding and human decision")
        if self.source_type is not SourceType.REAL and self.human_decision_id is not None:
            raise ValueError("non-real samples cannot claim a human decision")
        if self.claims_allowed and (
            self.source_type is not SourceType.REAL
            or self.provenance_status != "verified"
            or self.finding_id is None
            or self.human_decision_id is None
        ):
            raise ValueError("claimable optimization samples must be real and verified")
        if self.role is SampleRole.TARGET and self.signals is None:
            raise ValueError("target samples require failure signals")
        if self.role is SampleRole.PROTECTION and self.signals is not None:
            raise ValueError("protection samples cannot be used as failure labels")
        return self


class RootCauseDecision(ContractModel):
    root_cause: RootCause
    classifier: Literal["deterministic", "llm"]
    rationale: str = Field(min_length=1, max_length=8000)
    target_sample_ids: tuple[str, ...] = Field(min_length=1)
    call_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def call_identity_matches_classifier(self) -> Self:
        if (self.classifier == "llm") != (self.call_id is not None):
            raise ValueError("only LLM root-cause decisions require a call_id")
        if len(self.target_sample_ids) != len(set(self.target_sample_ids)):
            raise ValueError("target_sample_ids must be unique")
        return self


class RootCauseLlmOutput(ContractModel):
    root_cause: RootCause
    rationale: str = Field(min_length=1, max_length=8000)


class CandidateChange(ContractModel):
    scope: Literal["content", "execution_config"]
    path: str = Field(min_length=2, max_length=512)
    before_json: str | None = None
    after_json: str

    @model_validator(mode="after")
    def values_are_canonical(self) -> Self:
        if self.before_json is not None:
            _canonical_value(self.before_json, "before_json")
        _canonical_value(self.after_json, "after_json")
        if self.before_json == self.after_json:
            raise ValueError("candidate change must alter one value")
        return self


class CandidateProvenance(ContractModel):
    optimization_job_id: str = Field(min_length=1, max_length=128)
    attempt_number: int = Field(ge=1)
    base_rule_version_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    hashes: ExecutionHashes
    source_type: SourceType
    status: Literal["provisional", "verified"]
    claims_allowed: bool
    source_artifact_sha256s: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def claim_boundary_is_preserved(self) -> Self:
        if self.status == "provisional" and self.claims_allowed:
            raise ValueError("provisional candidates cannot allow claims")
        if len(self.source_artifact_sha256s) != len(set(self.source_artifact_sha256s)):
            raise ValueError("source artifact hashes must be unique")
        return self


class OptimizationCandidate(ContractModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    candidate_type: CandidateType
    root_cause: RootCause
    content_json: str = Field(min_length=2)
    execution_config_json: str = Field(min_length=2)
    change: CandidateChange
    rationale: str = Field(min_length=1, max_length=8000)
    target_sample_ids: tuple[str, ...] = Field(min_length=1)
    affected_protection_sample_ids: tuple[str, ...] = ()
    provenance: CandidateProvenance

    @model_validator(mode="after")
    def route_and_minimal_change_are_valid(self) -> Self:
        _canonical_object(self.content_json, "content_json")
        _canonical_object(self.execution_config_json, "execution_config_json")
        expected = ROOT_CAUSE_CANDIDATE_TYPES[self.root_cause]
        if expected is None or self.candidate_type is not expected:
            raise ValueError("candidate type is not allowed for the classified root cause")
        allowed = ALLOWED_CHANGE_PATHS[self.candidate_type]
        if not any(
            self.change.scope == scope
            and (
                self.change.path == path
                or self.change.path.startswith(f"{path}.")
            )
            for scope, path in allowed
        ):
            raise ValueError("candidate change path exceeds its root-cause boundary")
        if len(self.target_sample_ids) != len(set(self.target_sample_ids)):
            raise ValueError("target sample IDs must be unique")
        if len(self.affected_protection_sample_ids) != len(
            set(self.affected_protection_sample_ids)
        ):
            raise ValueError("protection sample IDs must be unique")
        return self


class RegressionCaseOutcome(ContractModel):
    sample_id: str = Field(min_length=1, max_length=256)
    role: SampleRole
    run_number: int = Field(ge=1)
    passed: bool
    result_sha256: str = Field(pattern=SHA256_PATTERN)


class JointRegressionResult(ContractModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    required_stability_runs: int = Field(ge=2, le=20)
    outcomes: tuple[RegressionCaseOutcome, ...] = Field(min_length=1)
    target_gate_passed: bool
    protection_gate_passed: bool
    stability_gate_passed: bool
    status: RegressionStatus
    provisional: bool
    claims_allowed: bool
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def gates_and_report_are_consistent(self) -> Self:
        keys = tuple((item.sample_id, item.run_number) for item in self.outcomes)
        if len(keys) != len(set(keys)):
            raise ValueError("regression outcomes must be unique by sample and run")
        target = [item for item in self.outcomes if item.role is SampleRole.TARGET]
        protection = [
            item for item in self.outcomes if item.role is SampleRole.PROTECTION
        ]
        if not target:
            raise ValueError("joint regression requires target outcomes")
        expected_target = all(item.passed for item in target)
        expected_protection = bool(protection) and all(item.passed for item in protection)
        by_sample: dict[str, list[RegressionCaseOutcome]] = {}
        for outcome in self.outcomes:
            by_sample.setdefault(outcome.sample_id, []).append(outcome)
        expected_stability = all(
            len(items) == self.required_stability_runs
            and len({item.result_sha256 for item in items}) == 1
            for items in by_sample.values()
        )
        if (
            self.target_gate_passed != expected_target
            or self.protection_gate_passed != expected_protection
            or self.stability_gate_passed != expected_stability
        ):
            raise ValueError("joint regression gate flags do not match outcomes")
        all_gates = expected_target and expected_protection and expected_stability
        if self.provisional:
            expected_status = (
                RegressionStatus.PROVISIONAL if all_gates else RegressionStatus.FAILED
            )
            if self.claims_allowed or self.status is not expected_status:
                raise ValueError("provisional regression cannot become a release claim")
        elif self.status is RegressionStatus.PASSED:
            if not all_gates or not self.claims_allowed:
                raise ValueError("passed regression requires all gates and claimable data")
        elif self.status is RegressionStatus.PROVISIONAL:
            raise ValueError("verified regression cannot use PROVISIONAL status")
        if self.status is RegressionStatus.FAILED and all_gates:
            raise ValueError("all passing gates cannot be marked failed")
        if self.status is RegressionStatus.FAILED and self.claims_allowed:
            raise ValueError("failed regression cannot allow claims")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match regression result")
        return self

    @property
    def accepted_for_manual_review(self) -> bool:
        return (
            self.target_gate_passed
            and self.protection_gate_passed
            and self.stability_gate_passed
        )


class AttemptFailure(ContractModel):
    phase: Literal["root_cause", "candidate_generation", "evaluation", "staging"]
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    retryable: bool
    call_id: str | None = Field(default=None, min_length=1, max_length=128)


class OptimizationReadiness(ContractModel):
    status: OptimizationReadinessStatus
    claims_allowed: bool
    blockers: tuple[str, ...] = ()
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    a4_evaluation_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    a4_run_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    a4_report_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    a4_binding_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    a4_result_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    verified_failure_sample_ids: tuple[str, ...] = ()
    dataset_sample_ids: tuple[str, ...] = ()
    assessed_at: datetime

    @model_validator(mode="after")
    def readiness_is_consistent(self) -> Self:
        if self.assessed_at.tzinfo is None:
            raise ValueError("readiness assessment time must be timezone-aware")
        if len(self.verified_failure_sample_ids) != len(
            set(self.verified_failure_sample_ids)
        ):
            raise ValueError("verified failure sample IDs must be unique")
        if len(self.dataset_sample_ids) != len(set(self.dataset_sample_ids)):
            raise ValueError("dataset sample IDs must be unique")
        if self.status is OptimizationReadinessStatus.READY:
            identities = (
                self.dataset_manifest_sha256,
                self.a4_evaluation_run_id,
                self.a4_run_sha256,
                self.a4_report_sha256,
                self.a4_binding_sha256,
                self.a4_result_sha256,
            )
            if (
                not self.claims_allowed
                or self.blockers
                or any(value is None for value in identities)
                or not self.verified_failure_sample_ids
                or not self.dataset_sample_ids
            ):
                raise ValueError(
                    "READY optimization requires complete claimable A4 evidence"
                )
        elif self.claims_allowed or not self.blockers:
            raise ValueError(
                "non-ready optimization requires blockers and cannot allow claims"
            )
        return self


class OptimizationTraceEvent(ContractModel):
    event_id: str = Field(pattern=SHA256_PATTERN)
    node: Literal[
        "load_failure_samples",
        "classify_root_cause",
        "generate_minimal_candidate",
        "run_target_gate",
        "run_protection_gate",
        "run_stability_gate",
        "stage_draft_rule",
        "wait_for_human_approval",
    ]
    outcome: OptimizationTraceOutcome
    attempt_number: int = Field(ge=0)
    candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    root_cause: RootCause | None = None
    call_id: str | None = Field(default=None, min_length=1, max_length=128)
    gate_passed: bool | None = None
    result_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    failure_code: str | None = Field(default=None, min_length=1, max_length=128)
    recorded_at: datetime
    event_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def event_is_hashed(self) -> Self:
        if self.recorded_at.tzinfo is None:
            raise ValueError("trace event time must be timezone-aware")
        payload = self.model_dump(mode="json", exclude={"event_sha256"})
        if self.event_sha256 != stable_sha256(payload):
            raise ValueError("event_sha256 does not match optimization trace event")
        return self


class OptimizationAttempt(ContractModel):
    attempt_id: str = Field(min_length=1, max_length=128)
    optimization_job_id: str = Field(min_length=1, max_length=128)
    attempt_number: int = Field(ge=1)
    status: AttemptStatus
    root_cause: RootCauseDecision | None = None
    candidates: tuple[OptimizationCandidate, ...] = ()
    evaluations: tuple[JointRegressionResult, ...] = ()
    selected_candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_rule_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    failure: AttemptFailure | None = None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def state_and_checkpoint_are_consistent(self) -> Self:
        if self.started_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("attempt timestamps must be timezone-aware")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique per attempt")
        if any(item.candidate_id not in candidate_ids for item in self.evaluations):
            raise ValueError("evaluation references an unknown candidate")
        if self.selected_candidate_id is not None and self.selected_candidate_id not in candidate_ids:
            raise ValueError("selected candidate is not part of the attempt")
        if self.candidate_rule_version_id is not None and self.selected_candidate_id is None:
            raise ValueError("staged rule version requires a selected candidate")
        if self.status in {
            AttemptStatus.COMPLETED,
            AttemptStatus.FAILED,
            AttemptStatus.WAITING_HUMAN,
        } and self.completed_at is None:
            raise ValueError("terminal attempts require completed_at")
        if self.status is AttemptStatus.FAILED and self.failure is None:
            raise ValueError("failed attempt requires a failure record")
        if self.status is AttemptStatus.WAITING_HUMAN and (
            self.root_cause is None
            or self.root_cause.root_cause is not RootCause.LABEL_UNCERTAIN
        ):
            raise ValueError("human handoff is reserved for LABEL_UNCERTAIN")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != stable_sha256(payload):
            raise ValueError("attempt checkpoint hash does not match its state")
        return self


class OptimizationJob(ContractModel):
    optimization_job_id: str = Field(min_length=1, max_length=128)
    base_rule_version_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    status: OptimizationStatus
    max_rounds: int = Field(ge=1, le=20)
    candidates_per_round: int = Field(ge=1, le=10)
    required_stability_runs: int = Field(ge=2, le=20)
    current_round: int = Field(ge=0)
    samples: tuple[OptimizationSample, ...] = Field(min_length=2)
    hashes: ExecutionHashes
    provenance: OptimizationProvenance
    readiness: OptimizationReadiness
    candidate_rule_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    last_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    failure_trajectory: tuple[AttemptFailure, ...] = ()
    graph_trace: tuple[OptimizationTraceEvent, ...] = ()
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def lifecycle_and_inputs_are_consistent(self) -> Self:
        if self.current_round > self.max_rounds:
            raise ValueError("current round cannot exceed max rounds")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("job timestamps must be timezone-aware")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("optimization sample IDs must be unique")
        if not any(item.role is SampleRole.TARGET for item in self.samples):
            raise ValueError("optimization requires at least one target sample")
        if not any(item.role is SampleRole.PROTECTION for item in self.samples):
            raise ValueError("optimization requires at least one protection sample")
        contains_provisional = any(
            item.provenance_status == "provisional" or not item.claims_allowed
            for item in self.samples
        )
        if contains_provisional and (
            self.provenance.status != "provisional" or self.provenance.claims_allowed
        ):
            raise ValueError("provisional samples must propagate to the optimization job")
        if self.provenance.claims_allowed != self.readiness.claims_allowed:
            raise ValueError("optimization provenance must match A5 readiness claims")
        if self.readiness.status is OptimizationReadinessStatus.READY:
            if any(
                item.source_type is not SourceType.REAL
                or item.provenance_status != "verified"
                or not item.claims_allowed
                for item in self.samples
            ):
                raise ValueError("READY optimization requires only real verified samples")
            target_ids = {
                item.sample_id for item in self.samples if item.role is SampleRole.TARGET
            }
            if not target_ids.issubset(
                set(self.readiness.verified_failure_sample_ids)
            ):
                raise ValueError("target samples must be verified A4 failure samples")
        expected_initial_status = {
            OptimizationReadinessStatus.NOT_READY: OptimizationStatus.NOT_READY,
            OptimizationReadinessStatus.BLOCKED: OptimizationStatus.BLOCKED,
        }.get(self.readiness.status)
        if expected_initial_status is not None and self.status is not expected_initial_status:
            raise ValueError("non-ready jobs must remain in their readiness terminal state")
        if self.status in TERMINAL_OPTIMIZATION_STATUSES and self.completed_at is None:
            raise ValueError("terminal optimization jobs require completed_at")
        if self.status not in TERMINAL_OPTIMIZATION_STATUSES and self.completed_at is not None:
            raise ValueError("non-terminal optimization jobs cannot be completed")
        if self.candidate_rule_version_id is not None and (
            self.status is not OptimizationStatus.WAITING_APPROVAL
        ):
            raise ValueError("candidate rule version is only exposed at approval boundary")
        return self


class CreateOptimizationJob(ContractModel):
    base_rule_version_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    max_rounds: int = Field(default=3, ge=1, le=20)
    candidates_per_round: int = Field(default=2, ge=1, le=10)
    required_stability_runs: int = Field(default=2, ge=2, le=20)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    retriever_sha256: str = Field(pattern=SHA256_PATTERN)
    tool_sha256: str = Field(pattern=SHA256_PATTERN)
    a4_evaluation_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    a4_report_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    samples: tuple[OptimizationSample, ...] = Field(min_length=2)
    provenance: OptimizationProvenance


class OptimizationJobSummary(ContractModel):
    optimization_job_id: str
    status: OptimizationStatus
    current_round: int
    max_rounds: int
    candidates_per_round: int
    candidate_rule_version_id: str | None = None
    last_checkpoint_sha256: str | None = None

    @classmethod
    def from_job(cls, job: OptimizationJob) -> "OptimizationJobSummary":
        return cls(
            optimization_job_id=job.optimization_job_id,
            status=job.status,
            current_round=job.current_round,
            max_rounds=job.max_rounds,
            candidates_per_round=job.candidates_per_round,
            candidate_rule_version_id=job.candidate_rule_version_id,
            last_checkpoint_sha256=job.last_checkpoint_sha256,
        )
