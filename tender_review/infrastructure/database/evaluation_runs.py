from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.evaluation.public import (
    EvaluationReport,
    EvaluationRun,
    EvaluationRunStatus,
    FrozenThresholdPolicy,
)
from tender_review.shared.errors import ConflictError, NotFoundError, RetryableError

from .models import EvaluationRun as EvaluationRunRecord
from .models import EvaluationThresholdPolicy


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class SqlAlchemyEvaluationRunRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def add(self, run: EvaluationRun, report: EvaluationReport) -> EvaluationRun:
        try:
            with self._sessions.begin() as session:
                session.add(self._row(run, report))
                session.flush()
                return run
        except IntegrityError as exc:
            raise ConflictError("evaluation run already exists", code="evaluation_run_conflict") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    def get(self, run_id: str) -> EvaluationRun:
        try:
            with self._sessions() as session:
                row = session.get(EvaluationRunRecord, run_id)
                if row is None:
                    raise NotFoundError("evaluation run does not exist", code="evaluation_run_not_found")
                return self._to_run(row)
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    def get_report(self, run_id: str) -> EvaluationReport:
        try:
            with self._sessions() as session:
                row = session.get(EvaluationRunRecord, run_id)
                if row is None:
                    raise NotFoundError("evaluation report does not exist", code="evaluation_report_not_found")
                return EvaluationReport.model_validate(row.report_json)
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    def list(self, limit: int = 100) -> tuple[EvaluationRun, ...]:
        try:
            with self._sessions() as session:
                rows = session.scalars(
                    select(EvaluationRunRecord)
                    .order_by(EvaluationRunRecord.created_at.desc())
                    .limit(max(1, min(limit, 500)))
                )
                result: list[EvaluationRun] = []
                for row in rows:
                    try:
                        result.append(self._to_run(row))
                    except ValueError:
                        # Pre-A4 rows are deliberately non-claimable and do not satisfy
                        # the new immutable run contract.
                        continue
                return tuple(result)
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    def complete(self, run: EvaluationRun, report: EvaluationReport) -> EvaluationRun:
        try:
            with self._sessions.begin() as session:
                row = session.scalar(
                    select(EvaluationRunRecord)
                    .where(EvaluationRunRecord.id == run.run_id)
                    .with_for_update()
                )
                if row is None:
                    raise NotFoundError("evaluation run does not exist", code="evaluation_run_not_found")
                current = self._to_run(row)
                if current.status not in {EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING}:
                    raise ConflictError("evaluation run is immutable", code="evaluation_run_immutable")
                if current.binding != run.binding:
                    raise ConflictError("evaluation binding changed", code="evaluation_binding_changed")
                self._apply(row, run, report)
                session.flush()
                return run
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    def add_policy(self, policy: FrozenThresholdPolicy) -> FrozenThresholdPolicy:
        try:
            with self._sessions.begin() as session:
                session.add(EvaluationThresholdPolicy(
                    id=policy.policy_id,
                    baseline_run_id=policy.baseline_run_id,
                    baseline_report_sha256=policy.baseline_report_sha256,
                    approved_by=policy.approved_by,
                    frozen_at=policy.frozen_at,
                    policy_sha256=policy.policy_sha256,
                    policy_json=policy.model_dump(mode="json"),
                ))
                session.flush()
                return policy
        except IntegrityError as exc:
            raise ConflictError("threshold policy already exists", code="threshold_policy_conflict") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    def get_policy(self, policy_id: str) -> FrozenThresholdPolicy:
        try:
            with self._sessions() as session:
                row = session.get(EvaluationThresholdPolicy, policy_id)
                if row is None:
                    raise NotFoundError("threshold policy does not exist", code="threshold_policy_not_found")
                return FrozenThresholdPolicy.model_validate(row.policy_json)
        except SQLAlchemyError as exc:
            raise RetryableError("evaluation storage is unavailable", code="evaluation_storage_unavailable") from exc

    @staticmethod
    def _row(run: EvaluationRun, report: EvaluationReport) -> EvaluationRunRecord:
        row = EvaluationRunRecord(
            id=run.run_id,
            dataset_version_id=run.binding.dataset_version_id,
            rule_version_id=run.binding.rule_version_id,
            model_config_id=run.model_config_id,
            report_artifact_id=None,
            status=run.status.value,
            retriever_version=run.retriever_version,
            code_version=run.evaluator_version,
            metrics_json=report.metrics.model_dump(mode="json"),
            started_at=run.started_at,
            completed_at=run.completed_at,
        )
        SqlAlchemyEvaluationRunRepository._apply(row, run, report)
        return row

    @staticmethod
    def _apply(row: EvaluationRunRecord, run: EvaluationRun, report: EvaluationReport) -> None:
        row.status = run.status.value
        row.purpose = run.purpose.value
        row.dataset_split = run.binding.dataset_split.value
        row.source_type = run.source_type.value
        row.provenance_status = run.provenance_status
        row.claims_allowed = run.claims_allowed
        row.dataset_manifest_sha256 = run.binding.dataset_manifest_sha256
        row.rule_version_sha256 = run.binding.rule_version_sha256
        row.input_sha256 = run.binding.input_sha256
        row.config_sha256 = run.binding.config_sha256
        row.code_sha256 = run.binding.code_sha256
        row.model_sha256 = run.binding.model_sha256
        row.prompt_sha256 = run.binding.prompt_sha256
        row.binding_sha256 = run.binding.binding_sha256
        row.result_sha256 = run.result_sha256
        row.report_sha256 = run.report_sha256
        row.run_sha256 = run.run_sha256
        row.evaluator_version = run.evaluator_version
        row.reproducibility_command = run.reproducibility_command
        row.blockers_json = list(run.blockers)
        row.dataset_snapshot_json = run.dataset.model_dump(mode="json")
        row.binding_json = run.binding.model_dump(mode="json")
        row.run_json = run.model_dump(mode="json")
        row.report_json = report.model_dump(mode="json")
        row.metrics_json = report.metrics.model_dump(mode="json")
        row.started_at = run.started_at
        row.completed_at = run.completed_at

    @staticmethod
    def _to_run(row: EvaluationRunRecord) -> EvaluationRun:
        payload = dict(row.run_json)
        if payload.get("started_at") is not None:
            payload["started_at"] = _aware(row.started_at)
        if payload.get("completed_at") is not None:
            payload["completed_at"] = _aware(row.completed_at)
        return EvaluationRun.model_validate(payload)
