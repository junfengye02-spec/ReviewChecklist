from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from pydantic import ValidationError

from tender_review.evaluation.public import (
    AnnotationDatasetVersion,
    EvaluationReport,
    EvaluationRun as EvaluationRunDto,
    EvaluationRunStatus,
    EvaluationSourceType,
)
from tender_review.rule_management.public import (
    EvaluationGate,
    RuleProvenance,
    RuleSet as RuleSetDto,
    RuleVersion as RuleVersionDto,
)
from tender_review.shared.errors import (
    ConflictError,
    NotFoundError,
    PermanentError,
    RetryableError,
)

from .models import DatasetVersion, EvaluationRun, RuleSet, RuleVersion


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class SqlAlchemyRuleVersionRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def get_rule_set(self, rule_set_id: str) -> RuleSetDto:
        try:
            with self._sessions() as session:
                row = session.get(RuleSet, rule_set_id)
                if row is None:
                    raise NotFoundError("rule set does not exist", code="rule_set_not_found")
                return self._to_set(row)
        except SQLAlchemyError as exc:
            raise RetryableError("rule storage is unavailable", code="rule_storage_unavailable") from exc

    def create_rule_set(self, rule_set: RuleSetDto) -> RuleSetDto:
        try:
            with self._sessions.begin() as session:
                session.add(RuleSet(
                    id=rule_set.rule_set_id,
                    rule_key=rule_set.rule_key,
                    name=rule_set.name,
                    description=rule_set.description,
                    current_version_id=rule_set.current_version_id,
                    created_at=rule_set.created_at,
                    updated_at=rule_set.created_at,
                    schema_version=str(rule_set.schema_version),
                ))
                session.flush()
                return rule_set
        except IntegrityError as exc:
            raise ConflictError("rule set already exists", code="rule_set_conflict") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("rule storage is unavailable", code="rule_storage_unavailable") from exc

    def get_version(self, rule_version_id: str) -> RuleVersionDto:
        try:
            with self._sessions() as session:
                row = session.get(RuleVersion, rule_version_id)
                if row is None:
                    raise NotFoundError("rule version does not exist", code="rule_version_not_found")
                return self._to_version(row)
        except SQLAlchemyError as exc:
            raise RetryableError("rule storage is unavailable", code="rule_storage_unavailable") from exc

    def list_versions(self, rule_set_id: str) -> tuple[RuleVersionDto, ...]:
        try:
            with self._sessions() as session:
                if session.get(RuleSet, rule_set_id) is None:
                    raise NotFoundError("rule set does not exist", code="rule_set_not_found")
                rows = session.scalars(
                    select(RuleVersion)
                    .where(RuleVersion.rule_set_id == rule_set_id)
                    .order_by(RuleVersion.version_number)
                )
                return tuple(self._to_version(row) for row in rows)
        except SQLAlchemyError as exc:
            raise RetryableError("rule storage is unavailable", code="rule_storage_unavailable") from exc

    def add_version(self, version: RuleVersionDto) -> RuleVersionDto:
        try:
            with self._sessions.begin() as session:
                session.add(self._row(version))
                session.flush()
                return version
        except IntegrityError as exc:
            raise ConflictError("duplicate rule version", code="rule_version_duplicate") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("rule storage is unavailable", code="rule_storage_unavailable") from exc

    def save_gate(self, version: RuleVersionDto, gate: EvaluationGate) -> RuleVersionDto:
        try:
            with self._sessions.begin() as session:
                row = session.scalar(
                    select(RuleVersion).where(RuleVersion.id == version.rule_version_id).with_for_update()
                )
                if row is None:
                    raise NotFoundError("rule version does not exist", code="rule_version_not_found")
                if row.content_hash != version.content_sha256:
                    raise ConflictError("rule content is immutable", code="rule_content_changed")
                row.status = version.status.value
                row.evaluation_gate_json = gate.model_dump(mode="json")
                session.flush()
                return version.model_copy(update={"evaluation_gate": gate})
        except SQLAlchemyError as exc:
            raise RetryableError("rule storage is unavailable", code="rule_storage_unavailable") from exc

    def publish(self, rule_set: RuleSetDto, version: RuleVersionDto) -> RuleVersionDto:
        try:
            with self._sessions.begin() as session:
                set_row = session.scalar(select(RuleSet).where(RuleSet.id == rule_set.rule_set_id).with_for_update())
                version_row = session.scalar(select(RuleVersion).where(RuleVersion.id == version.rule_version_id).with_for_update())
                if set_row is None or version_row is None:
                    raise NotFoundError("rule set or version does not exist", code="rule_version_not_found")
                self._assert_release_transaction(session, version_row, version)
                if set_row.current_version_id and set_row.current_version_id != version.rule_version_id:
                    previous = session.get(RuleVersion, set_row.current_version_id)
                    if previous is not None:
                        previous.status = "ROLLED_BACK"
                set_row.current_version_id = version.rule_version_id
                self._apply_publish(version_row, version)
                session.flush()
                return version
        except SQLAlchemyError as exc:
            raise RetryableError("rule publication storage is unavailable", code="rule_publish_storage_unavailable") from exc

    def rollback(
        self, rule_set: RuleSetDto, current: RuleVersionDto, target: RuleVersionDto
    ) -> RuleVersionDto:
        try:
            with self._sessions.begin() as session:
                set_row = session.scalar(select(RuleSet).where(RuleSet.id == rule_set.rule_set_id).with_for_update())
                current_row = session.get(RuleVersion, current.rule_version_id)
                target_row = session.get(RuleVersion, target.rule_version_id)
                if set_row is None or current_row is None or target_row is None:
                    raise NotFoundError("rollback resources do not exist", code="rule_rollback_not_found")
                current_row.status = current.status.value
                target_row.status = target.status.value
                set_row.current_version_id = target.rule_version_id
                session.flush()
                return target
        except SQLAlchemyError as exc:
            raise RetryableError("rule rollback storage is unavailable", code="rule_rollback_storage_unavailable") from exc

    @staticmethod
    def _row(version: RuleVersionDto) -> RuleVersion:
        import json
        return RuleVersion(
            id=version.rule_version_id,
            rule_set_id=version.rule_set_id,
            parent_version_id=version.parent_version_id,
            version_number=version.version_number,
            status=version.status.value,
            content_hash=version.content_sha256,
            content_json=json.loads(version.content_json),
            execution_config_json=json.loads(version.execution_config_json),
            change_summary=version.change_summary,
            provenance_json=version.provenance.model_dump(mode="json"),
            evaluation_gate_json=(version.evaluation_gate.model_dump(mode="json") if version.evaluation_gate else None),
            created_at=version.created_at,
            updated_at=version.created_at,
            published_at=version.published_at,
            published_by=version.published_by,
            schema_version=str(version.schema_version),
        )

    @staticmethod
    def _to_set(row: RuleSet) -> RuleSetDto:
        return RuleSetDto(
            rule_set_id=row.id,
            rule_key=row.rule_key,
            name=row.name,
            description=row.description,
            current_version_id=row.current_version_id,
            created_at=_aware(row.created_at),
        )

    @staticmethod
    def _to_version(row: RuleVersion) -> RuleVersionDto:
        import json
        return RuleVersionDto(
            rule_version_id=row.id,
            rule_set_id=row.rule_set_id,
            version_number=row.version_number,
            parent_version_id=row.parent_version_id,
            status=row.status,
            content_json=json.dumps(row.content_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            execution_config_json=json.dumps(row.execution_config_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            content_sha256=row.content_hash,
            change_summary=row.change_summary,
            provenance=RuleProvenance.model_validate(row.provenance_json),
            evaluation_gate=(EvaluationGate.model_validate(row.evaluation_gate_json) if row.evaluation_gate_json else None),
            created_at=_aware(row.created_at),
            published_at=_aware(row.published_at) if row.published_at else None,
            published_by=row.published_by,
        )

    @staticmethod
    def _apply_publish(row: RuleVersion, version: RuleVersionDto) -> None:
        row.status = version.status.value
        row.published_at = version.published_at
        row.published_by = version.published_by

    @staticmethod
    def _assert_release_transaction(
        session: Session,
        version_row: RuleVersion,
        candidate: RuleVersionDto,
    ) -> None:
        try:
            gate = EvaluationGate.model_validate(version_row.evaluation_gate_json)
            if gate.evaluation_run_id is None or gate.report_sha256 is None:
                raise ValueError("persisted gate has no completed report")
            run_row = session.scalar(
                select(EvaluationRun)
                .where(EvaluationRun.id == gate.evaluation_run_id)
                .with_for_update()
            )
            dataset_row = session.scalar(
                select(DatasetVersion)
                .where(DatasetVersion.id == gate.dataset_version_id)
                .with_for_update()
            )
            if run_row is None or dataset_row is None:
                raise ValueError("persisted release evidence is missing")
            run = EvaluationRunDto.model_validate(run_row.run_json)
            report = EvaluationReport.model_validate(run_row.report_json)
            dataset = AnnotationDatasetVersion.model_validate(dataset_row.manifest_json)
        except (TypeError, ValueError, ValidationError) as exc:
            raise PermanentError(
                "persisted release evidence is invalid",
                code="release_gate_persistence_invalid",
            ) from exc

        valid = (
            version_row.status == "WAITING_APPROVAL"
            and candidate.status.value == "PUBLISHED"
            and gate.status.value == "PASSED"
            and not gate.provisional
            and gate.claims_allowed
            and gate.rule_version_id == version_row.id == run.binding.rule_version_id
            and gate.dataset_version_id == run.binding.dataset_version_id == dataset.dataset_version_id
            and gate.report_sha256 == run.report_sha256 == report.report_sha256
            and run.status is EvaluationRunStatus.COMPLETED
            and run.purpose.value == "RELEASE_GATE"
            and run.source_type is EvaluationSourceType.REAL
            and run.provenance_status == "verified"
            and run.claims_allowed
            and report.source_type is EvaluationSourceType.REAL
            and report.status == "verified"
            and report.claims_allowed
            and report.release_gate.passed
            and dataset.status.value == "FROZEN"
            and dataset.provenance.status == "verified"
            and dataset.provenance.claims_allowed
            and dataset.manifest_sha256 == run.binding.dataset_manifest_sha256
            and run.binding.rule_version_sha256 == version_row.content_hash
            and run_row.binding_sha256 == run.binding.binding_sha256
            and run_row.result_sha256 == run.result_sha256 == report.result_sha256
            and run_row.report_sha256 == report.report_sha256
            and run_row.run_sha256 == run.run_sha256
        )
        if not valid:
            raise PermanentError(
                "database transaction rejected non-eligible publication",
                code="release_gate_transaction_rejected",
            )
