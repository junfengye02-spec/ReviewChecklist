from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from pydantic import ValidationError

from tender_review.evaluation.public import (
    AnnotationDatasetVersion,
    EvaluationReport as A4EvaluationReport,
    EvaluationRun as A4EvaluationRun,
)

from tender_review.optimization.public import (
    AttemptFailure,
    OptimizationAttempt as AttemptDto,
    OptimizationJob as JobDto,
    OptimizationProvenance,
    OptimizationReadiness,
    OptimizationReadinessStatus,
    OptimizationSample,
    OptimizationTraceEvent,
    OptimizationStatus as OptimizationStatusDto,
    SampleRole,
    ExecutionHashes,
    RootCauseDecision,
    JointRegressionResult,
    OptimizationCandidate,
)
from tender_review.shared.errors import (
    ConflictError,
    NotFoundError,
    PermanentError,
    RetryableError,
)

from .models import (
    DatasetVersion,
    EvaluationRun,
    OptimizationAttempt,
    OptimizationJob,
)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class SqlAlchemyOptimizationRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create_job(self, job: JobDto) -> JobDto:
        try:
            with self._sessions.begin() as session:
                _assert_a5_readiness_transaction(session, job)
                session.add(_job_row(job))
                session.flush()
                return job
        except IntegrityError as exc:
            raise ConflictError(
                "optimization job already exists", code="optimization_job_conflict"
            ) from exc
        except SQLAlchemyError as exc:
            raise RetryableError(
                "optimization storage is unavailable",
                code="optimization_storage_unavailable",
            ) from exc

    def get_job(self, optimization_job_id: str) -> JobDto:
        try:
            with self._sessions() as session:
                row = session.get(OptimizationJob, optimization_job_id)
                if row is None:
                    raise NotFoundError(
                        "optimization job does not exist",
                        code="optimization_job_not_found",
                    )
                return _job_dto(row)
        except SQLAlchemyError as exc:
            raise RetryableError(
                "optimization storage is unavailable",
                code="optimization_storage_unavailable",
            ) from exc

    def save_job(self, job: JobDto) -> JobDto:
        try:
            with self._sessions.begin() as session:
                row = session.scalar(
                    select(OptimizationJob)
                    .where(OptimizationJob.id == job.optimization_job_id)
                    .with_for_update()
                )
                if row is None:
                    raise NotFoundError(
                        "optimization job does not exist",
                        code="optimization_job_not_found",
                    )
                if row.base_rule_version_id != job.base_rule_version_id:
                    raise ConflictError(
                        "optimization input identity is immutable",
                        code="optimization_job_input_changed",
                    )
                _assert_a5_readiness_transaction(session, job)
                _apply_job(row, job)
                session.flush()
                return job
        except SQLAlchemyError as exc:
            raise RetryableError(
                "optimization storage is unavailable",
                code="optimization_storage_unavailable",
            ) from exc

    def get_attempt(
        self, optimization_job_id: str, attempt_number: int
    ) -> AttemptDto | None:
        try:
            with self._sessions() as session:
                if session.get(OptimizationJob, optimization_job_id) is None:
                    raise NotFoundError(
                        "optimization job does not exist",
                        code="optimization_job_not_found",
                    )
                row = session.scalar(
                    select(OptimizationAttempt).where(
                        OptimizationAttempt.optimization_job_id == optimization_job_id,
                        OptimizationAttempt.attempt_number == attempt_number,
                    )
                )
                return None if row is None else _attempt_dto(row)
        except SQLAlchemyError as exc:
            raise RetryableError(
                "optimization storage is unavailable",
                code="optimization_storage_unavailable",
            ) from exc

    def save_attempt(self, attempt: AttemptDto) -> AttemptDto:
        try:
            with self._sessions.begin() as session:
                if session.get(OptimizationJob, attempt.optimization_job_id) is None:
                    raise NotFoundError(
                        "optimization job does not exist",
                        code="optimization_job_not_found",
                    )
                row = session.scalar(
                    select(OptimizationAttempt)
                    .where(
                        OptimizationAttempt.optimization_job_id
                        == attempt.optimization_job_id,
                        OptimizationAttempt.attempt_number == attempt.attempt_number,
                    )
                    .with_for_update()
                )
                if row is None:
                    session.add(_attempt_row(attempt))
                else:
                    current = _attempt_dto(row)
                    _require_monotonic(current, attempt)
                    _apply_attempt(row, attempt)
                session.flush()
                return attempt
        except IntegrityError as exc:
            raise ConflictError(
                "optimization attempt conflicts with its checkpoint",
                code="optimization_attempt_conflict",
            ) from exc
        except SQLAlchemyError as exc:
            raise RetryableError(
                "optimization storage is unavailable",
                code="optimization_storage_unavailable",
            ) from exc

    def list_attempts(
        self, optimization_job_id: str
    ) -> tuple[AttemptDto, ...]:
        try:
            with self._sessions() as session:
                if session.get(OptimizationJob, optimization_job_id) is None:
                    raise NotFoundError(
                        "optimization job does not exist",
                        code="optimization_job_not_found",
                    )
                rows = session.scalars(
                    select(OptimizationAttempt)
                    .where(
                        OptimizationAttempt.optimization_job_id == optimization_job_id
                    )
                    .order_by(OptimizationAttempt.attempt_number)
                )
                return tuple(_attempt_dto(row) for row in rows)
        except SQLAlchemyError as exc:
            raise RetryableError(
                "optimization storage is unavailable",
                code="optimization_storage_unavailable",
            ) from exc


def _job_row(job: JobDto) -> OptimizationJob:
    row = OptimizationJob(
        id=job.optimization_job_id,
        base_rule_version_id=job.base_rule_version_id,
        dataset_version_id=job.dataset_version_id,
        max_rounds=job.max_rounds,
        created_at=job.created_at,
        updated_at=job.updated_at,
        schema_version=str(job.schema_version),
    )
    _apply_job(row, job)
    return row


def _apply_job(row: OptimizationJob, job: JobDto) -> None:
    row.candidate_rule_version_id = job.candidate_rule_version_id
    row.status = job.status.value
    row.current_round = job.current_round
    row.candidates_per_round = job.candidates_per_round
    row.required_stability_runs = job.required_stability_runs
    row.hashes_json = job.hashes.model_dump(mode="json")
    row.provenance_json = job.provenance.model_dump(mode="json")
    row.samples_json = [item.model_dump(mode="json") for item in job.samples]
    row.failure_trajectory_json = [
        item.model_dump(mode="json") for item in job.failure_trajectory
    ]
    row.readiness_json = job.readiness.model_dump(mode="json")
    row.graph_trace_json = [item.model_dump(mode="json") for item in job.graph_trace]
    row.error_json = (
        job.failure_trajectory[-1].model_dump(mode="json")
        if job.failure_trajectory
        else None
    )
    row.last_checkpoint_sha256 = job.last_checkpoint_sha256
    row.completed_at = job.completed_at
    row.updated_at = job.updated_at


def _job_dto(row: OptimizationJob) -> JobDto:
    legacy_blocked = row.readiness_json is None
    readiness = (
        OptimizationReadiness(
            status=OptimizationReadinessStatus.BLOCKED,
            claims_allowed=False,
            blockers=(
                "legacy pre-A5 optimization has no verified A3/A4 evidence",
            ),
            assessed_at=_aware(row.created_at),
        )
        if legacy_blocked
        else OptimizationReadiness.model_validate(row.readiness_json)
    )
    return JobDto(
        optimization_job_id=row.id,
        base_rule_version_id=row.base_rule_version_id,
        dataset_version_id=row.dataset_version_id,
        status=(OptimizationStatusDto.BLOCKED if legacy_blocked else row.status),
        max_rounds=row.max_rounds,
        candidates_per_round=row.candidates_per_round,
        required_stability_runs=row.required_stability_runs,
        current_round=row.current_round,
        samples=tuple(
            OptimizationSample.model_validate(item) for item in row.samples_json or []
        ),
        hashes=ExecutionHashes.model_validate(row.hashes_json),
        provenance=OptimizationProvenance.model_validate(row.provenance_json),
        readiness=readiness,
        candidate_rule_version_id=(
            None if legacy_blocked else row.candidate_rule_version_id
        ),
        last_checkpoint_sha256=row.last_checkpoint_sha256,
        failure_trajectory=tuple(
            AttemptFailure.model_validate(item)
            for item in row.failure_trajectory_json or []
        ),
        graph_trace=tuple(
            OptimizationTraceEvent.model_validate(item)
            for item in row.graph_trace_json or []
        ),
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        completed_at=(
            _aware(row.completed_at or row.updated_at)
            if legacy_blocked
            else _aware(row.completed_at) if row.completed_at else None
        ),
    )


def _attempt_row(attempt: AttemptDto) -> OptimizationAttempt:
    row = OptimizationAttempt(
        id=attempt.attempt_id,
        optimization_job_id=attempt.optimization_job_id,
        attempt_number=attempt.attempt_number,
        root_cause="PENDING",
        rationale="checkpoint",
        evaluation_result_json={},
        created_at=attempt.started_at,
        updated_at=attempt.updated_at,
        schema_version=str(attempt.schema_version),
    )
    _apply_attempt(row, attempt)
    return row


def _apply_attempt(row: OptimizationAttempt, attempt: AttemptDto) -> None:
    row.status = attempt.status.value
    row.candidate_rule_version_id = attempt.candidate_rule_version_id
    row.root_cause = (
        attempt.root_cause.root_cause.value if attempt.root_cause else "PENDING"
    )
    row.rationale = (
        attempt.root_cause.rationale if attempt.root_cause else "checkpoint"
    )
    row.root_cause_json = (
        attempt.root_cause.model_dump(mode="json") if attempt.root_cause else None
    )
    row.candidates_json = [item.model_dump(mode="json") for item in attempt.candidates]
    row.evaluations_json = [
        item.model_dump(mode="json") for item in attempt.evaluations
    ]
    row.evaluation_result_json = {"evaluations": row.evaluations_json}
    row.accepted = attempt.selected_candidate_id is not None
    row.rejection_reason = attempt.failure.message if attempt.failure else None
    row.failure_json = attempt.failure.model_dump(mode="json") if attempt.failure else None
    row.checkpoint_sha256 = attempt.checkpoint_sha256
    row.completed_at = attempt.completed_at
    row.updated_at = attempt.updated_at


def _attempt_dto(row: OptimizationAttempt) -> AttemptDto:
    return AttemptDto(
        attempt_id=row.id,
        optimization_job_id=row.optimization_job_id,
        attempt_number=row.attempt_number,
        status=row.status,
        root_cause=(
            RootCauseDecision.model_validate(row.root_cause_json)
            if row.root_cause_json
            else None
        ),
        candidates=tuple(
            OptimizationCandidate.model_validate(item)
            for item in row.candidates_json or []
        ),
        evaluations=tuple(
            JointRegressionResult.model_validate(item)
            for item in row.evaluations_json or []
        ),
        selected_candidate_id=(
            next(
                (
                    item["candidate_id"]
                    for item in row.evaluations_json or []
                    if item.get("target_gate_passed")
                    and item.get("protection_gate_passed")
                    and item.get("stability_gate_passed")
                ),
                None,
            )
            if row.accepted
            else None
        ),
        candidate_rule_version_id=row.candidate_rule_version_id,
        failure=(AttemptFailure.model_validate(row.failure_json) if row.failure_json else None),
        started_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        completed_at=_aware(row.completed_at) if row.completed_at else None,
        checkpoint_sha256=row.checkpoint_sha256,
    )


def _require_monotonic(current: AttemptDto, candidate: AttemptDto) -> None:
    if current.attempt_id != candidate.attempt_id:
        raise ConflictError(
            "attempt identity is immutable", code="optimization_attempt_changed"
        )
    if current.root_cause is not None and candidate.root_cause != current.root_cause:
        raise ConflictError(
            "root cause checkpoint is immutable", code="optimization_root_cause_changed"
        )
    if current.candidates and candidate.candidates != current.candidates:
        raise ConflictError(
            "candidate checkpoint is immutable", code="optimization_candidates_changed"
        )
    if candidate.evaluations[: len(current.evaluations)] != current.evaluations:
        raise ConflictError(
            "completed evaluations are immutable",
            code="optimization_evaluation_changed",
        )


def _assert_a5_readiness_transaction(session: Session, job: JobDto) -> None:
    if job.status in {
        OptimizationStatusDto.NOT_READY,
        OptimizationStatusDto.BLOCKED,
        OptimizationStatusDto.CANCELLED,
        OptimizationStatusDto.OPTIMIZATION_FAILED,
    }:
        return
    readiness = job.readiness
    run = (
        session.get(EvaluationRun, readiness.a4_evaluation_run_id)
        if readiness.a4_evaluation_run_id
        else None
    )
    dataset = session.get(DatasetVersion, job.dataset_version_id)
    report = run.report_json if run is not None else None
    manifest = dataset.manifest_json if dataset is not None else None
    provenance = dataset.provenance_json if dataset is not None else None
    failure_samples = (
        {
            item.get("sample_id"): item
            for item in report.get("failure_samples", [])
            if isinstance(item, dict) and item.get("sample_id")
        }
        if isinstance(report, dict)
        else {}
    )
    target_samples = tuple(
        item for item in job.samples if item.role is SampleRole.TARGET
    )
    target_evidence_matches = bool(target_samples) and all(
        item.sample_id in failure_samples
        and item.evidence_sha256
        in tuple(failure_samples[item.sample_id].get("evidence_sha256s", ()))
        for item in target_samples
    )
    try:
        persisted_run = (
            A4EvaluationRun.model_validate(run.run_json) if run is not None else None
        )
        persisted_report = (
            A4EvaluationReport.model_validate(report)
            if isinstance(report, dict)
            else None
        )
        persisted_dataset = (
            AnnotationDatasetVersion.model_validate(manifest)
            if isinstance(manifest, dict)
            else None
        )
    except (ValidationError, ValueError, TypeError):
        persisted_run = None
        persisted_report = None
        persisted_dataset = None
    valid = (
        readiness.claims_allowed
        and run is not None
        and dataset is not None
        and run.status == "COMPLETED"
        and run.purpose == "CANDIDATE_DIAGNOSTIC"
        and run.dataset_split == "OPTIMIZATION"
        and run.source_type == "real"
        and run.provenance_status == "verified"
        and run.claims_allowed
        and run.dataset_version_id == job.dataset_version_id
        and run.dataset_manifest_sha256 == readiness.dataset_manifest_sha256
        and run.binding_sha256 == readiness.a4_binding_sha256
        and run.result_sha256 == readiness.a4_result_sha256
        and run.report_sha256 == readiness.a4_report_sha256
        and run.run_sha256 == readiness.a4_run_sha256
        and persisted_run is not None
        and persisted_report is not None
        and persisted_dataset is not None
        and persisted_run.run_id == persisted_report.run_id
        and persisted_run.binding == persisted_report.binding
        and persisted_run.result_sha256 == persisted_report.result_sha256
        and persisted_run.dataset == persisted_report.dataset
        and persisted_dataset.dataset_version_id == persisted_run.dataset.dataset_version_id
        and persisted_dataset.manifest_sha256 == persisted_run.dataset.manifest_sha256
        and isinstance(report, dict)
        and report.get("claims_allowed") is True
        and report.get("status") == "verified"
        and report.get("source_type") == "real"
        and report.get("report_sha256") == readiness.a4_report_sha256
        and dataset.status == "FROZEN"
        and dataset.manifest_hash == readiness.dataset_manifest_sha256
        and isinstance(manifest, dict)
        and manifest.get("manifest_kind") == "a3_annotation_dataset"
        and isinstance(manifest.get("required_human_cases"), int)
        and manifest["required_human_cases"] > 0
        and manifest["required_human_cases"] == len(manifest.get("samples") or ())
        and isinstance(provenance, dict)
        and provenance.get("status") == "verified"
        and provenance.get("claims_allowed") is True
        and target_evidence_matches
    )
    if not valid:
        raise PermanentError(
            "persisted A3/A4 evidence does not authorize optimization execution",
            code="optimization_readiness_bypass_forbidden",
        )
