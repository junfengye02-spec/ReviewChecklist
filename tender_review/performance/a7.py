from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import to_jsonable_python

from tender_review.shared.contracts import ContractModel


SHA256_PATTERN = r"^[0-9a-f]{64}$"
COLLECTOR_VERSION = "tender-review-a7/1"
CONCURRENCY_LEVELS = (1, 5, 10, 20, 50)
WORKER_LEVELS = (1, 2, 5, 10)
NODE_NAMES = ("parsing", "retrieval", "review", "report")


def stable_sha256(value: Any) -> str:
    rendered = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class FileProfile(str, Enum):
    TEXT_20 = "TEXT_20"
    MIXED_100 = "MIXED_100"
    SCANNED_300 = "SCANNED_300"


FILE_PROFILE_PAGES = {
    FileProfile.TEXT_20: 20,
    FileProfile.MIXED_100: 100,
    FileProfile.SCANNED_300: 300,
}


class A7RunStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    NOT_READY = "NOT_READY"
    COMPLETED = "COMPLETED"


class A7SourceType(str, Enum):
    REAL = "real"
    PROVISIONAL = "provisional"
    SYNTHETIC = "synthetic"
    FAKE = "fake"
    SQLITE = "sqlite"


# Public compatibility names stay concise while OpenAPI components remain unique.
RunStatus = A7RunStatus
SourceType = A7SourceType


class QueueDecision(str, Enum):
    NO_DECISION = "NO_DECISION"
    KEEP_MYSQL_QUEUE = "KEEP_MYSQL_QUEUE"
    PROPOSE_ROCKETMQ_ADMISSION = "PROPOSE_ROCKETMQ_ADMISSION"


class RedisDecision(str, Enum):
    NO_DECISION = "NO_DECISION"
    KEEP_REDIS_OUT = "KEEP_REDIS_OUT"
    PROPOSE_REDIS_ADMISSION = "PROPOSE_REDIS_ADMISSION"


class ObservationKind(str, Enum):
    MYSQL_CPU_PERCENT = "mysql_cpu_percent"
    CLAIM_LATENCY_MS = "claim_latency_ms"
    EMPTY_POLL = "empty_poll"
    LOCK_WAIT_MS = "lock_wait_ms"
    QUEUE_LATENCY_MS = "queue_latency_ms"
    NODE_DURATION_MS = "node_duration_ms"
    JOB_OUTCOME = "job_outcome"
    RECOVERY_OUTCOME = "recovery_outcome"


class ObservationSource(str, Enum):
    MYSQL_PERFORMANCE_SCHEMA = "mysql_performance_schema"
    MYSQL_SERVER_STATUS = "mysql_server_status"
    WORKER_STRUCTURED_LOG = "worker_structured_log"
    API_JOB_SNAPSHOT = "api_job_snapshot"
    PROCESS_SUPERVISOR = "process_supervisor"


class AdmissionEvidenceType(str, Enum):
    MYSQL_SCAN_CLAIM_BOTTLENECK = "MYSQL_SCAN_CLAIM_BOTTLENECK"
    INDEPENDENT_STAGE_SCALING = "INDEPENDENT_STAGE_SCALING"
    MULTI_CONSUMER_REQUIREMENT = "MULTI_CONSUMER_REQUIREMENT"
    DATABASE_WORKER_SLO_FAILURE = "DATABASE_WORKER_SLO_FAILURE"
    PROGRESS_QUERY_DB_PRESSURE = "PROGRESS_QUERY_DB_PRESSURE"
    HOT_RULE_READ_BOTTLENECK = "HOT_RULE_READ_BOTTLENECK"
    CROSS_INSTANCE_RATE_LIMIT = "CROSS_INSTANCE_RATE_LIMIT"
    REDIS_COMPARATIVE_IMPROVEMENT = "REDIS_COMPARATIVE_IMPROVEMENT"


ROCKETMQ_EVIDENCE_TYPES = frozenset(
    {
        AdmissionEvidenceType.MYSQL_SCAN_CLAIM_BOTTLENECK,
        AdmissionEvidenceType.INDEPENDENT_STAGE_SCALING,
        AdmissionEvidenceType.MULTI_CONSUMER_REQUIREMENT,
        AdmissionEvidenceType.DATABASE_WORKER_SLO_FAILURE,
    }
)
REDIS_NEED_TYPES = frozenset(
    {
        AdmissionEvidenceType.PROGRESS_QUERY_DB_PRESSURE,
        AdmissionEvidenceType.HOT_RULE_READ_BOTTLENECK,
        AdmissionEvidenceType.CROSS_INSTANCE_RATE_LIMIT,
    }
)


class MetricId(str, Enum):
    MYSQL_CPU_P95_PERCENT = "mysql_cpu_p95_percent"
    CLAIM_LATENCY_P95_MS = "claim_latency_p95_ms"
    EMPTY_POLL_RATIO = "empty_poll_ratio"
    LOCK_WAIT_P95_MS = "lock_wait_p95_ms"
    QUEUE_LATENCY_P95_MS = "queue_latency_p95_ms"
    NODE_PARSING_P95_MS = "node_parsing_p95_ms"
    NODE_RETRIEVAL_P95_MS = "node_retrieval_p95_ms"
    NODE_REVIEW_P95_MS = "node_review_p95_ms"
    NODE_REPORT_P95_MS = "node_report_p95_ms"
    THROUGHPUT_JOBS_PER_MINUTE = "throughput_jobs_per_minute"
    FAILURE_RATE = "failure_rate"
    RECOVERY_RATE = "recovery_rate"


REQUIRED_METRIC_IDS = tuple(MetricId)


def scenario_id(profile: FileProfile, concurrency: int, workers: int) -> str:
    return f"{profile.value.lower()}-c{concurrency}-w{workers}"


def expected_scenarios() -> tuple[tuple[FileProfile, int, int], ...]:
    return tuple(product(tuple(FileProfile), CONCURRENCY_LEVELS, WORKER_LEVELS))


class A7ExecutionBinding(ContractModel):
    environment_id: str = Field(min_length=1, max_length=255)
    compose_file_sha256: str = Field(pattern=SHA256_PATTERN)
    compose_config_sha256: str = Field(pattern=SHA256_PATTERN)
    api_image_digest: str = Field(pattern=SHA256_PATTERN)
    worker_image_digest: str = Field(pattern=SHA256_PATTERN)
    mysql_image_digest: str = Field(pattern=SHA256_PATTERN)
    minio_image_digest: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    git_dirty: bool
    code_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_sha256: str = Field(pattern=SHA256_PATTERN)
    collector_version: Literal[COLLECTOR_VERSION] = COLLECTOR_VERSION
    binding_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def binding_hash_matches(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if self.binding_sha256 != stable_sha256(payload):
            raise ValueError("binding_sha256 does not match the A7 execution binding")
        return self


class A7Authenticity(ContractModel):
    source_type: SourceType
    provenance_status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    environment_kind: Literal["dedicated-real", "local", "ci", "unknown"]
    adapter_mode: Literal["production", "fake"]
    database_dialect: Literal["mysql", "sqlite", "other", "unknown"]
    mysql_version: str | None = Field(default=None, max_length=255)
    mysql_server_uuid_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    real_mysql_exercised: bool = False
    real_minio_exercised: bool = False
    real_model_exercised: bool = False
    real_pdf_end_to_end: bool = False
    independent_worker_processes_verified: bool = False
    fake_adapters_used: bool = False
    sqlite_used: bool = False
    synthetic_artifacts_used: bool = False
    attested_by: str | None = Field(default=None, max_length=255)
    attested_at: datetime | None = None

    @property
    def is_claimable_real_capture(self) -> bool:
        return (
            self.source_type is SourceType.REAL
            and self.provenance_status == "verified"
            and self.environment_kind == "dedicated-real"
            and self.adapter_mode == "production"
            and self.database_dialect == "mysql"
            and bool(self.mysql_version)
            and self.mysql_server_uuid_sha256 is not None
            and self.real_mysql_exercised
            and self.real_minio_exercised
            and self.real_model_exercised
            and self.real_pdf_end_to_end
            and self.independent_worker_processes_verified
            and not self.fake_adapters_used
            and not self.sqlite_used
            and not self.synthetic_artifacts_used
            and bool(self.attested_by and self.attested_by.strip())
            and self.attested_at is not None
        )

    @model_validator(mode="after")
    def claim_boundary_is_consistent(self) -> Self:
        if self.attested_at is not None and self.attested_at.tzinfo is None:
            raise ValueError("attested_at must be timezone-aware")
        if self.claims_allowed != self.is_claimable_real_capture:
            raise ValueError(
                "claims_allowed must exactly match verified real MySQL/Worker provenance"
            )
        return self


class RawObservation(ContractModel):
    observation_id: str = Field(min_length=1, max_length=255)
    scenario_id: str = Field(min_length=1, max_length=255)
    observed_at: datetime
    kind: ObservationKind
    value: float = Field(ge=0)
    unit: Literal["percent", "milliseconds", "boolean", "count"]
    source: ObservationSource
    source_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    worker_process_id: str | None = Field(default=None, max_length=255)
    job_id: str | None = Field(default=None, max_length=255)
    node_name: Literal["parsing", "retrieval", "review", "report"] | None = None
    record_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def observation_is_consistent_and_hashed(self) -> Self:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        expected_units = {
            ObservationKind.MYSQL_CPU_PERCENT: "percent",
            ObservationKind.CLAIM_LATENCY_MS: "milliseconds",
            ObservationKind.EMPTY_POLL: "boolean",
            ObservationKind.LOCK_WAIT_MS: "milliseconds",
            ObservationKind.QUEUE_LATENCY_MS: "milliseconds",
            ObservationKind.NODE_DURATION_MS: "milliseconds",
            ObservationKind.JOB_OUTCOME: "boolean",
            ObservationKind.RECOVERY_OUTCOME: "boolean",
        }
        if self.unit != expected_units[self.kind]:
            raise ValueError(
                f"{self.kind.value} observations require {expected_units[self.kind]}"
            )
        if self.kind in {
            ObservationKind.EMPTY_POLL,
            ObservationKind.JOB_OUTCOME,
            ObservationKind.RECOVERY_OUTCOME,
        } and self.value not in {0, 1}:
            raise ValueError(f"{self.kind.value} observations must be 0 or 1")
        if self.kind is ObservationKind.MYSQL_CPU_PERCENT and self.value > 100:
            raise ValueError("MySQL CPU percent must not exceed 100")
        if self.kind is ObservationKind.NODE_DURATION_MS:
            if self.node_name is None:
                raise ValueError("node duration observations require node_name")
        elif self.node_name is not None:
            raise ValueError("node_name is only valid for node duration observations")
        mysql_kinds = {
            ObservationKind.MYSQL_CPU_PERCENT,
            ObservationKind.LOCK_WAIT_MS,
        }
        if self.kind in mysql_kinds and self.source not in {
            ObservationSource.MYSQL_PERFORMANCE_SCHEMA,
            ObservationSource.MYSQL_SERVER_STATUS,
        }:
            raise ValueError("MySQL observations require a MySQL telemetry source")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if self.record_sha256 != stable_sha256(payload):
            raise ValueError("record_sha256 does not match the raw observation")
        return self


class ScenarioCapture(ContractModel):
    scenario_id: str = Field(min_length=1, max_length=255)
    file_profile: FileProfile
    page_count: int = Field(ge=1)
    concurrency: int
    workers: int
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worker_process_ids: tuple[str, ...] = ()
    observations: tuple[RawObservation, ...] = ()
    blockers: tuple[str, ...] = ()
    capture_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def capture_is_consistent_and_hashed(self) -> Self:
        if self.scenario_id != scenario_id(
            self.file_profile, self.concurrency, self.workers
        ):
            raise ValueError("scenario_id does not match profile/concurrency/workers")
        if self.page_count != FILE_PROFILE_PAGES[self.file_profile]:
            raise ValueError("page_count does not match the A7 file profile")
        if self.concurrency not in CONCURRENCY_LEVELS:
            raise ValueError("unsupported A7 concurrency level")
        if self.workers not in WORKER_LEVELS:
            raise ValueError("unsupported A7 Worker level")
        if len(self.worker_process_ids) != len(set(self.worker_process_ids)):
            raise ValueError("worker_process_ids must be unique")
        if any(item.scenario_id != self.scenario_id for item in self.observations):
            raise ValueError("raw observation references a different scenario")
        observation_ids = tuple(item.observation_id for item in self.observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique within a scenario")
        if self.status is RunStatus.COMPLETED:
            if self.started_at is None or self.finished_at is None:
                raise ValueError(
                    "completed captures require start and finish timestamps"
                )
            if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
                raise ValueError("capture timestamps must be timezone-aware")
            if self.finished_at <= self.started_at:
                raise ValueError("finished_at must be later than started_at")
            if len(self.worker_process_ids) != self.workers:
                raise ValueError(
                    "completed capture must prove the requested Worker count"
                )
            if self.blockers:
                raise ValueError("completed captures cannot contain blockers")
            _assert_complete_observations(self)
        else:
            if self.observations or self.worker_process_ids:
                raise ValueError(
                    "unfinished captures cannot contain observations or Workers"
                )
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError(
                    "unfinished captures cannot contain collection timestamps"
                )
            if not self.blockers:
                raise ValueError("NOT_RUN/NOT_READY captures require blockers")
        payload = self.model_dump(mode="json", exclude={"capture_sha256"})
        if self.capture_sha256 != stable_sha256(payload):
            raise ValueError("capture_sha256 does not match the scenario capture")
        return self


def _assert_complete_observations(capture: ScenarioCapture) -> None:
    kinds = {item.kind for item in capture.observations}
    missing = set(ObservationKind) - kinds
    if missing:
        raise ValueError(
            "completed capture is missing raw observation kinds: "
            + ", ".join(sorted(item.value for item in missing))
        )
    nodes = {
        item.node_name
        for item in capture.observations
        if item.kind is ObservationKind.NODE_DURATION_MS
    }
    if nodes != set(NODE_NAMES):
        raise ValueError("completed capture requires observations for all A7 nodes")
    outcomes = [
        item
        for item in capture.observations
        if item.kind is ObservationKind.JOB_OUTCOME
    ]
    if len(outcomes) != capture.concurrency:
        raise ValueError(
            "completed capture requires one job outcome per concurrent task"
        )
    observed_worker_ids = {
        item.worker_process_id
        for item in capture.observations
        if item.worker_process_id is not None
    }
    if not set(capture.worker_process_ids).issubset(observed_worker_ids):
        raise ValueError("every Worker process must appear in raw observations")


class AdmissionEvidence(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=255)
    evidence_type: AdmissionEvidenceType
    status: Literal["verified"]
    claims_allowed: Literal[True]
    verified_by: str = Field(min_length=1, max_length=255)
    verified_at: datetime
    scenario_ids: tuple[str, ...] = Field(min_length=1)
    observation_sha256s: tuple[str, ...] = Field(min_length=1)
    consumer_ids: tuple[str, ...] = ()
    detail_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def evidence_is_consistent_and_hashed(self) -> Self:
        if self.verified_at.tzinfo is None:
            raise ValueError("verified_at must be timezone-aware")
        if len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError("scenario_ids must be unique")
        if len(self.observation_sha256s) != len(set(self.observation_sha256s)):
            raise ValueError("observation_sha256s must be unique")
        if (
            self.evidence_type is AdmissionEvidenceType.MULTI_CONSUMER_REQUIREMENT
            and len(set(self.consumer_ids)) < 2
        ):
            raise ValueError("multi-consumer evidence requires at least two consumers")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != stable_sha256(payload):
            raise ValueError("evidence_sha256 does not match admission evidence")
        return self


class A7EvidenceBundle(ContractModel):
    run_id: str = Field(min_length=1, max_length=255)
    status: RunStatus
    binding: A7ExecutionBinding
    authenticity: A7Authenticity
    captures: tuple[ScenarioCapture, ...]
    admission_evidence: tuple[AdmissionEvidence, ...] = ()
    collected_at: datetime
    raw_observations_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    attestation_key_id: str | None = Field(default=None, max_length=255)
    attestation_hmac_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def bundle_is_complete_real_and_hashed(self) -> Self:
        if self.collected_at.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware")
        expected = {
            scenario_id(profile, concurrency, workers)
            for profile, concurrency, workers in expected_scenarios()
        }
        actual = {item.scenario_id for item in self.captures}
        if len(self.captures) != len(actual) or actual != expected:
            raise ValueError("A7 evidence must contain the exact 60-scenario matrix")
        if self.status is RunStatus.COMPLETED:
            if not self.authenticity.is_claimable_real_capture:
                raise ValueError(
                    "completed A7 evidence requires verified real provenance"
                )
            if any(item.status is not RunStatus.COMPLETED for item in self.captures):
                raise ValueError("completed A7 evidence requires all 60 captures")
            if not self.attestation_key_id or not self.attestation_hmac_sha256:
                raise ValueError("completed A7 evidence requires a trusted attestation")
        elif any(item.status is RunStatus.COMPLETED for item in self.captures):
            raise ValueError(
                "unfinished bundles cannot contain claimable completed captures"
            )
        observations = [
            item.model_dump(mode="json")
            for capture in self.captures
            for item in capture.observations
        ]
        if self.raw_observations_sha256 != stable_sha256(observations):
            raise ValueError(
                "raw_observations_sha256 does not match captured observations"
            )
        scenario_ids = {item.scenario_id for item in self.captures}
        observation_hashes = {
            item.record_sha256
            for capture in self.captures
            for item in capture.observations
        }
        observations_by_hash = {
            item.record_sha256: item
            for capture in self.captures
            for item in capture.observations
        }
        for evidence in self.admission_evidence:
            if not set(evidence.scenario_ids).issubset(scenario_ids):
                raise ValueError("admission evidence references an unknown scenario")
            if not set(evidence.observation_sha256s).issubset(observation_hashes):
                raise ValueError(
                    "admission evidence references unknown raw observations"
                )
            _assert_admission_observation_contract(evidence, observations_by_hash)
        payload = self.model_dump(
            mode="json",
            exclude={"evidence_sha256", "attestation_hmac_sha256"},
        )
        if self.evidence_sha256 != stable_sha256(payload):
            raise ValueError("evidence_sha256 does not match the A7 evidence bundle")
        return self


def _assert_admission_observation_contract(
    evidence: AdmissionEvidence, observations_by_hash: dict[str, RawObservation]
) -> None:
    kinds = {observations_by_hash[item].kind for item in evidence.observation_sha256s}
    requirements = {
        AdmissionEvidenceType.MYSQL_SCAN_CLAIM_BOTTLENECK: {
            ObservationKind.CLAIM_LATENCY_MS,
            ObservationKind.LOCK_WAIT_MS,
        },
        AdmissionEvidenceType.INDEPENDENT_STAGE_SCALING: {
            ObservationKind.NODE_DURATION_MS,
        },
        AdmissionEvidenceType.MULTI_CONSUMER_REQUIREMENT: {
            ObservationKind.JOB_OUTCOME,
        },
        AdmissionEvidenceType.DATABASE_WORKER_SLO_FAILURE: {
            ObservationKind.CLAIM_LATENCY_MS,
            ObservationKind.QUEUE_LATENCY_MS,
            ObservationKind.JOB_OUTCOME,
        },
        AdmissionEvidenceType.PROGRESS_QUERY_DB_PRESSURE: {
            ObservationKind.MYSQL_CPU_PERCENT,
        },
        AdmissionEvidenceType.HOT_RULE_READ_BOTTLENECK: {
            ObservationKind.MYSQL_CPU_PERCENT,
            ObservationKind.LOCK_WAIT_MS,
        },
        AdmissionEvidenceType.CROSS_INSTANCE_RATE_LIMIT: {
            ObservationKind.JOB_OUTCOME,
        },
        AdmissionEvidenceType.REDIS_COMPARATIVE_IMPROVEMENT: {
            ObservationKind.MYSQL_CPU_PERCENT,
            ObservationKind.QUEUE_LATENCY_MS,
        },
    }
    missing = requirements[evidence.evidence_type] - kinds
    if missing:
        raise ValueError(
            f"{evidence.evidence_type.value} lacks required raw observation kinds: "
            + ", ".join(sorted(item.value for item in missing))
        )


class ThresholdRule(ContractModel):
    metric_id: MetricId
    operator: Literal["lte", "gte"]
    threshold: float = Field(ge=0)

    @model_validator(mode="after")
    def operator_matches_metric(self) -> Self:
        gte_metrics = {
            MetricId.THROUGHPUT_JOBS_PER_MINUTE,
            MetricId.RECOVERY_RATE,
        }
        expected = "gte" if self.metric_id in gte_metrics else "lte"
        if self.operator != expected:
            raise ValueError(f"{self.metric_id.value} requires operator={expected}")
        if (
            self.metric_id
            in {
                MetricId.MYSQL_CPU_P95_PERCENT,
            }
            and self.threshold > 100
        ):
            raise ValueError("percent threshold must not exceed 100")
        if (
            self.metric_id
            in {
                MetricId.EMPTY_POLL_RATIO,
                MetricId.FAILURE_RATE,
                MetricId.RECOVERY_RATE,
            }
            and self.threshold > 1
        ):
            raise ValueError("ratio threshold must not exceed 1")
        return self


class FrozenA7ThresholdPolicy(ContractModel):
    policy_id: str = Field(min_length=1, max_length=255)
    baseline_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    source_type: Literal[SourceType.REAL]
    provenance_status: Literal["verified"]
    claims_allowed: Literal[True]
    approved_by: str = Field(min_length=1, max_length=255)
    frozen_at: datetime
    rules: tuple[ThresholdRule, ...]
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    attestation_key_id: str = Field(min_length=1, max_length=255)
    attestation_hmac_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def policy_is_complete_frozen_and_hashed(self) -> Self:
        if self.frozen_at.tzinfo is None:
            raise ValueError("frozen_at must be timezone-aware")
        metric_ids = tuple(item.metric_id for item in self.rules)
        if len(metric_ids) != len(set(metric_ids)) or set(metric_ids) != set(
            REQUIRED_METRIC_IDS
        ):
            raise ValueError("frozen A7 policy must define every metric exactly once")
        payload = self.model_dump(
            mode="json", exclude={"policy_sha256", "attestation_hmac_sha256"}
        )
        if self.policy_sha256 != stable_sha256(payload):
            raise ValueError("policy_sha256 does not match the frozen A7 policy")
        return self


class ScenarioMetrics(ContractModel):
    scenario_id: str
    mysql_cpu_p95_percent: float = Field(ge=0, le=100)
    claim_latency_p95_ms: float = Field(ge=0)
    empty_poll_ratio: float = Field(ge=0, le=1)
    lock_wait_p95_ms: float = Field(ge=0)
    queue_latency_p95_ms: float = Field(ge=0)
    node_p95_ms: dict[Literal["parsing", "retrieval", "review", "report"], float]
    throughput_jobs_per_minute: float = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    recovery_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def all_nodes_are_present(self) -> Self:
        if set(self.node_p95_ms) != set(NODE_NAMES):
            raise ValueError("node_p95_ms must contain parsing/retrieval/review/report")
        if any(value < 0 for value in self.node_p95_ms.values()):
            raise ValueError("node P95 values must not be negative")
        return self

    def values(self) -> dict[MetricId, float]:
        return {
            MetricId.MYSQL_CPU_P95_PERCENT: self.mysql_cpu_p95_percent,
            MetricId.CLAIM_LATENCY_P95_MS: self.claim_latency_p95_ms,
            MetricId.EMPTY_POLL_RATIO: self.empty_poll_ratio,
            MetricId.LOCK_WAIT_P95_MS: self.lock_wait_p95_ms,
            MetricId.QUEUE_LATENCY_P95_MS: self.queue_latency_p95_ms,
            MetricId.NODE_PARSING_P95_MS: self.node_p95_ms["parsing"],
            MetricId.NODE_RETRIEVAL_P95_MS: self.node_p95_ms["retrieval"],
            MetricId.NODE_REVIEW_P95_MS: self.node_p95_ms["review"],
            MetricId.NODE_REPORT_P95_MS: self.node_p95_ms["report"],
            MetricId.THROUGHPUT_JOBS_PER_MINUTE: self.throughput_jobs_per_minute,
            MetricId.FAILURE_RATE: self.failure_rate,
            MetricId.RECOVERY_RATE: self.recovery_rate,
        }


class ThresholdAssessment(ContractModel):
    scenario_id: str
    metric_id: MetricId
    value: float = Field(ge=0)
    operator: Literal["lte", "gte"]
    threshold: float = Field(ge=0)
    passed: bool


class A7AdmissionReport(ContractModel):
    run_id: str
    status: RunStatus
    source_type: SourceType
    provenance_status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    matrix_expected: Literal[60] = 60
    matrix_completed: int = Field(ge=0, le=60)
    scenario_metrics: tuple[ScenarioMetrics, ...] = ()
    threshold_assessments: tuple[ThresholdAssessment, ...] = ()
    evidence_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    raw_observations_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    threshold_policy_id: str | None = None
    threshold_policy_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    binding: A7ExecutionBinding | None = None
    authenticity: A7Authenticity | None = None
    collected_at: datetime | None = None
    collector_version: Literal[COLLECTOR_VERSION]
    queue_decision: QueueDecision
    operational_action: Literal["KEEP_MYSQL_QUEUE"] = "KEEP_MYSQL_QUEUE"
    redis_decision: RedisDecision
    automatic_stack_change_allowed: Literal[False] = False
    decision_reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    generated_at: datetime
    reproducibility_command: str
    report_sha256: str = Field(pattern=SHA256_PATTERN)
    attestation_key_id: str | None = Field(default=None, max_length=255)
    attestation_hmac_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def report_boundary_and_hash_match(self) -> Self:
        for name in ("generated_at", "collected_at"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        claimable = (
            self.status is RunStatus.COMPLETED
            and self.source_type is SourceType.REAL
            and self.provenance_status == "verified"
            and self.matrix_completed == self.matrix_expected == 60
            and self.evidence_sha256 is not None
            and self.raw_observations_sha256 is not None
            and self.binding is not None
            and self.authenticity is not None
            and self.authenticity.is_claimable_real_capture
            and len(self.scenario_metrics) == 60
            and self.queue_decision is not QueueDecision.NO_DECISION
            and bool(self.attestation_key_id and self.attestation_hmac_sha256)
        )
        if self.claims_allowed != claimable:
            raise ValueError("claims_allowed does not match A7 real-report provenance")
        if self.status is not RunStatus.COMPLETED:
            if self.scenario_metrics or self.threshold_assessments:
                raise ValueError(
                    "unfinished A7 reports cannot expose performance metrics"
                )
            if self.queue_decision is not QueueDecision.NO_DECISION:
                raise ValueError(
                    "unfinished A7 reports cannot make an admission decision"
                )
            if not self.blockers:
                raise ValueError("unfinished A7 reports require blockers")
        else:
            expected_ids = {
                scenario_id(profile, concurrency, workers)
                for profile, concurrency, workers in expected_scenarios()
            }
            metric_ids = tuple(item.scenario_id for item in self.scenario_metrics)
            if len(metric_ids) != 60 or set(metric_ids) != expected_ids:
                raise ValueError("completed A7 reports require the exact 60 scenarios")
        if bool(self.threshold_policy_id) != bool(self.threshold_policy_sha256):
            raise ValueError(
                "threshold policy ID and SHA-256 must be supplied together"
            )
        if self.queue_decision is QueueDecision.KEEP_MYSQL_QUEUE:
            expected_assessments = {
                (scenario, metric)
                for scenario in {
                    scenario_id(profile, concurrency, workers)
                    for profile, concurrency, workers in expected_scenarios()
                }
                for metric in MetricId
            }
            actual_assessments = {
                (item.scenario_id, item.metric_id)
                for item in self.threshold_assessments
            }
            if (
                not self.threshold_policy_id
                or len(self.threshold_assessments) != len(expected_assessments)
                or actual_assessments != expected_assessments
                or not all(item.passed for item in self.threshold_assessments)
            ):
                raise ValueError(
                    "KEEP_MYSQL_QUEUE requires all 60x12 frozen thresholds to pass"
                )
        if self.queue_decision is QueueDecision.PROPOSE_ROCKETMQ_ADMISSION:
            if not any(
                reason.startswith("verified:") for reason in self.decision_reasons
            ):
                raise ValueError(
                    "RocketMQ proposal requires verified admission evidence"
                )
        if self.redis_decision is RedisDecision.PROPOSE_REDIS_ADMISSION:
            required = {"verified:REDIS_COMPARATIVE_IMPROVEMENT"}
            if not required.issubset(set(self.decision_reasons)):
                raise ValueError(
                    "Redis proposal requires verified comparative evidence"
                )
        if not self.reproducibility_command.startswith(
            "python -m tender_review.performance assess"
        ):
            raise ValueError("unsupported A7 reproducibility command")
        payload = self.model_dump(
            mode="json", exclude={"report_sha256", "attestation_hmac_sha256"}
        )
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match the A7 report")
        return self


def seal_model(
    model_type: type[ContractModel],
    payload: dict[str, Any],
    hash_field: str,
    *,
    attestation_key: str | None = None,
    attestation_key_id: str | None = None,
):
    draft_values = dict(payload)
    draft_values.setdefault("schema_version", 1)
    has_attestation = "attestation_hmac_sha256" in model_type.model_fields
    if attestation_key is not None:
        if not has_attestation:
            raise ValueError("this contract does not support attestations")
        if not attestation_key_id or not attestation_key_id.strip():
            raise ValueError("attestation_key_id is required when signing")
        _validate_attestation_key(attestation_key)
        draft_values["attestation_key_id"] = attestation_key_id.strip()
    if has_attestation:
        draft_values["attestation_hmac_sha256"] = None
    draft_values[hash_field] = "0" * 64
    draft = model_type.model_construct(**draft_values)
    sealed = draft.model_dump(mode="python")
    hash_exclusions = {hash_field}
    if has_attestation:
        hash_exclusions.add("attestation_hmac_sha256")
    sealed[hash_field] = stable_sha256(
        draft.model_dump(mode="json", exclude=hash_exclusions)
    )
    if attestation_key is not None:
        sealed["attestation_hmac_sha256"] = _attestation_hmac(
            attestation_key, sealed[hash_field]
        )
    return model_type.model_validate(sealed)


def _validate_attestation_key(key: str) -> None:
    if len(key.encode("utf-8")) < 32:
        raise ValueError("A7 attestation key must contain at least 32 bytes")


def _attestation_hmac(key: str, digest: str) -> str:
    _validate_attestation_key(key)
    return hmac.new(
        key.encode("utf-8"), digest.encode("ascii"), hashlib.sha256
    ).hexdigest()


def _verify_attestation(
    *,
    digest: str,
    signature: str | None,
    key_id: str | None,
    trusted_key: str,
    trusted_key_id: str,
) -> None:
    if key_id != trusted_key_id:
        raise ValueError("A7 attestation key ID is not trusted")
    expected = _attestation_hmac(trusted_key, digest)
    if signature is None or not hmac.compare_digest(signature, expected):
        raise ValueError("A7 attestation HMAC verification failed")


def _percentile_95(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate P95 from an empty observation set")
    ordered = sorted(values)
    rank = max(0, ((95 * len(ordered) + 99) // 100) - 1)
    return float(ordered[rank])


def metrics_from_capture(capture: ScenarioCapture) -> ScenarioMetrics:
    if capture.status is not RunStatus.COMPLETED:
        raise ValueError("metrics can only be calculated from completed captures")

    def values(kind: ObservationKind, *, node: str | None = None) -> list[float]:
        return [
            item.value
            for item in capture.observations
            if item.kind is kind and (node is None or item.node_name == node)
        ]

    outcomes = values(ObservationKind.JOB_OUTCOME)
    recoveries = values(ObservationKind.RECOVERY_OUTCOME)
    assert capture.started_at is not None and capture.finished_at is not None
    elapsed_minutes = (capture.finished_at - capture.started_at).total_seconds() / 60
    return ScenarioMetrics(
        scenario_id=capture.scenario_id,
        mysql_cpu_p95_percent=_percentile_95(values(ObservationKind.MYSQL_CPU_PERCENT)),
        claim_latency_p95_ms=_percentile_95(values(ObservationKind.CLAIM_LATENCY_MS)),
        empty_poll_ratio=sum(values(ObservationKind.EMPTY_POLL))
        / len(values(ObservationKind.EMPTY_POLL)),
        lock_wait_p95_ms=_percentile_95(values(ObservationKind.LOCK_WAIT_MS)),
        queue_latency_p95_ms=_percentile_95(values(ObservationKind.QUEUE_LATENCY_MS)),
        node_p95_ms={
            node: _percentile_95(values(ObservationKind.NODE_DURATION_MS, node=node))
            for node in NODE_NAMES
        },
        throughput_jobs_per_minute=len(outcomes) / elapsed_minutes,
        failure_rate=1 - (sum(outcomes) / len(outcomes)),
        recovery_rate=sum(recoveries) / len(recoveries),
    )


def _unfinished_captures(
    status: RunStatus, blocker: str
) -> tuple[ScenarioCapture, ...]:
    captures: list[ScenarioCapture] = []
    for profile, concurrency, workers in expected_scenarios():
        payload = {
            "scenario_id": scenario_id(profile, concurrency, workers),
            "file_profile": profile,
            "page_count": FILE_PROFILE_PAGES[profile],
            "concurrency": concurrency,
            "workers": workers,
            "status": status,
            "blockers": (blocker,),
        }
        captures.append(seal_model(ScenarioCapture, payload, "capture_sha256"))
    return tuple(captures)


def create_unavailable_report(
    *,
    status: RunStatus,
    blockers: tuple[str, ...],
    run_id: str = "a7-not-run",
    now: datetime | None = None,
) -> A7AdmissionReport:
    if status not in {RunStatus.NOT_RUN, RunStatus.NOT_READY}:
        raise ValueError("unavailable reports must be NOT_RUN or NOT_READY")
    if not blockers:
        raise ValueError("unavailable reports require blockers")
    generated_at = now or datetime.now(timezone.utc)
    payload = {
        "run_id": run_id,
        "status": status,
        "source_type": SourceType.PROVISIONAL,
        "provenance_status": "unknown",
        "claims_allowed": False,
        "matrix_expected": 60,
        "matrix_completed": 0,
        "collector_version": COLLECTOR_VERSION,
        "queue_decision": QueueDecision.NO_DECISION,
        "operational_action": "KEEP_MYSQL_QUEUE",
        "redis_decision": RedisDecision.NO_DECISION,
        "automatic_stack_change_allowed": False,
        "decision_reasons": (
            "No technology admission conclusion is permitted without verified real evidence.",
        ),
        "blockers": blockers,
        "generated_at": generated_at,
        "reproducibility_command": "python -m tender_review.performance assess",
    }
    return seal_model(A7AdmissionReport, payload, "report_sha256")


def create_not_run_plan(now: datetime | None = None) -> dict[str, Any]:
    blocker = (
        "Real MySQL/MinIO/model/PDF workload and independent Workers were not run."
    )
    report = create_unavailable_report(
        status=RunStatus.NOT_RUN,
        blockers=(
            blocker,
            "No frozen A7 threshold policy was supplied.",
            "RocketMQ and Redis admission remain untriggered.",
        ),
        now=now,
    )
    return {
        "schema_version": 1,
        "status": RunStatus.NOT_RUN,
        "matrix": [
            item.model_dump(mode="json")
            for item in _unfinished_captures(RunStatus.NOT_RUN, blocker)
        ],
        "required_metrics": [item.value for item in REQUIRED_METRIC_IDS],
        "report": report.model_dump(mode="json"),
    }


def assess_evidence(
    evidence: A7EvidenceBundle | None,
    policy: FrozenA7ThresholdPolicy | None,
    *,
    now: datetime | None = None,
    trusted_attestation_key: str | None = None,
    trusted_attestation_key_id: str | None = None,
) -> A7AdmissionReport:
    generated_at = now or datetime.now(timezone.utc)
    if evidence is None:
        return create_unavailable_report(
            status=RunStatus.NOT_READY,
            blockers=(
                "No A7 evidence bundle was supplied.",
                "Verified real MySQL/Worker observations are required.",
            ),
            now=generated_at,
        )
    if evidence.status is not RunStatus.COMPLETED:
        return create_unavailable_report(
            status=RunStatus.NOT_READY,
            blockers=(
                "A7 evidence matrix is not completed in a verified real environment.",
            ),
            run_id=evidence.run_id,
            now=generated_at,
        )
    if not trusted_attestation_key or not trusted_attestation_key_id:
        return create_unavailable_report(
            status=RunStatus.NOT_READY,
            blockers=("No trusted A7 evidence attestation key is configured.",),
            run_id=evidence.run_id,
            now=generated_at,
        )
    _verify_attestation(
        digest=evidence.evidence_sha256,
        signature=evidence.attestation_hmac_sha256,
        key_id=evidence.attestation_key_id,
        trusted_key=trusted_attestation_key,
        trusted_key_id=trusted_attestation_key_id,
    )
    if policy is not None:
        _verify_attestation(
            digest=policy.policy_sha256,
            signature=policy.attestation_hmac_sha256,
            key_id=policy.attestation_key_id,
            trusted_key=trusted_attestation_key,
            trusted_key_id=trusted_attestation_key_id,
        )

    metrics = tuple(metrics_from_capture(item) for item in evidence.captures)
    rules = {item.metric_id: item for item in policy.rules} if policy else {}
    assessments: list[ThresholdAssessment] = []
    if policy is not None:
        for metric in metrics:
            for metric_id, value in metric.values().items():
                rule = rules[metric_id]
                passed = (
                    value <= rule.threshold
                    if rule.operator == "lte"
                    else value >= rule.threshold
                )
                assessments.append(
                    ThresholdAssessment(
                        scenario_id=metric.scenario_id,
                        metric_id=metric_id,
                        value=value,
                        operator=rule.operator,
                        threshold=rule.threshold,
                        passed=passed,
                    )
                )

    evidence_types = {item.evidence_type for item in evidence.admission_evidence}
    rocketmq_types = evidence_types & ROCKETMQ_EVIDENCE_TYPES
    decision_reasons = tuple(
        f"verified:{item.value}" for item in sorted(rocketmq_types, key=str)
    )
    blockers: list[str] = []
    if rocketmq_types:
        queue_decision = QueueDecision.PROPOSE_ROCKETMQ_ADMISSION
    elif policy is None:
        queue_decision = QueueDecision.NO_DECISION
        blockers.append("No verified frozen A7 threshold policy was supplied.")
    elif assessments and all(item.passed for item in assessments):
        queue_decision = QueueDecision.KEEP_MYSQL_QUEUE
        decision_reasons = (
            "All 60 real scenarios passed every frozen threshold and no admission trigger was verified.",
        )
    else:
        queue_decision = QueueDecision.NO_DECISION
        blockers.append(
            "One or more frozen targets failed, but no permitted queue bottleneck or consumer need was verified."
        )

    redis_need = bool(evidence_types & REDIS_NEED_TYPES)
    redis_comparison = (
        AdmissionEvidenceType.REDIS_COMPARATIVE_IMPROVEMENT in evidence_types
    )
    if redis_need and redis_comparison:
        redis_decision = RedisDecision.PROPOSE_REDIS_ADMISSION
        decision_reasons = decision_reasons + (
            "verified:REDIS_COMPARATIVE_IMPROVEMENT",
        )
    elif queue_decision is QueueDecision.NO_DECISION:
        redis_decision = RedisDecision.NO_DECISION
    else:
        redis_decision = RedisDecision.KEEP_REDIS_OUT

    claims_allowed = queue_decision is not QueueDecision.NO_DECISION
    report_status = RunStatus.COMPLETED if claims_allowed else RunStatus.NOT_READY
    if report_status is RunStatus.NOT_READY:
        # A non-decision must not leak otherwise real-looking performance values.
        return create_unavailable_report(
            status=RunStatus.NOT_READY,
            blockers=tuple(blockers) or ("A7 admission decision is not ready.",),
            run_id=evidence.run_id,
            now=generated_at,
        )

    payload = {
        "run_id": evidence.run_id,
        "status": report_status,
        "source_type": SourceType.REAL,
        "provenance_status": "verified",
        "claims_allowed": True,
        "matrix_expected": 60,
        "matrix_completed": 60,
        "scenario_metrics": metrics,
        "threshold_assessments": tuple(assessments),
        "evidence_sha256": evidence.evidence_sha256,
        "raw_observations_sha256": evidence.raw_observations_sha256,
        "threshold_policy_id": policy.policy_id if policy else None,
        "threshold_policy_sha256": policy.policy_sha256 if policy else None,
        "binding": evidence.binding,
        "authenticity": evidence.authenticity,
        "collected_at": evidence.collected_at,
        "collector_version": evidence.binding.collector_version,
        "queue_decision": queue_decision,
        "operational_action": "KEEP_MYSQL_QUEUE",
        "redis_decision": redis_decision,
        "automatic_stack_change_allowed": False,
        "decision_reasons": decision_reasons,
        "blockers": tuple(blockers),
        "generated_at": generated_at,
        "reproducibility_command": (
            "python -m tender_review.performance assess --evidence <bundle.json> "
            "--threshold-policy <policy.json> --output <report.json>"
        ),
    }
    return seal_model(
        A7AdmissionReport,
        payload,
        "report_sha256",
        attestation_key=trusted_attestation_key,
        attestation_key_id=trusted_attestation_key_id,
    )


def load_report(
    path: str | Path,
    *,
    trusted_attestation_key: str | None = None,
    trusted_attestation_key_id: str | None = None,
) -> A7AdmissionReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = A7AdmissionReport.model_validate(payload)
    if report.claims_allowed:
        if not trusted_attestation_key or not trusted_attestation_key_id:
            raise ValueError("claimable A7 report requires a configured trusted key")
        _verify_attestation(
            digest=report.report_sha256,
            signature=report.attestation_hmac_sha256,
            key_id=report.attestation_key_id,
            trusted_key=trusted_attestation_key,
            trusted_key_id=trusted_attestation_key_id,
        )
    return report
