from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.evaluation.public import (
    AnnotationDatasetVersion,
    AnnotationSampleStatus,
    DatasetStatus,
)
from tender_review.shared.errors import ConflictError, NotFoundError, RetryableError

from .models import DatasetAnnotationSampleRecord, DatasetVersion


class SqlAlchemyAnnotationDatasetRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def add_version(self, version: AnnotationDatasetVersion) -> AnnotationDatasetVersion:
        try:
            with self._sessions.begin() as session:
                session.add(
                    DatasetVersion(
                        id=version.dataset_version_id,
                        dataset_name=version.dataset_name,
                        version_number=version.version_number,
                        parent_version_id=version.parent_version_id,
                        manifest_hash=version.manifest_sha256,
                        source_type="REAL",
                        status=version.status.value,
                        change_summary=version.change_summary,
                        provenance_json=version.provenance.model_dump(mode="json"),
                        manifest_json=version.model_dump(mode="json"),
                        split_strategy_json={
                            item.document_sha256: item.split.value
                            for item in version.samples
                        },
                        manifest_artifact_id=None,
                        frozen_at=version.frozen_at,
                        created_at=version.created_at,
                        updated_at=version.created_at,
                        schema_version=str(version.schema_version),
                    )
                )
                session.add_all(self._sample_rows(version))
                session.flush()
                return version
        except IntegrityError as exc:
            raise ConflictError(
                "duplicate or invalid annotation dataset version",
                code="annotation_dataset_conflict",
            ) from exc
        except SQLAlchemyError as exc:
            raise RetryableError(
                "annotation dataset storage is unavailable",
                code="annotation_dataset_storage_unavailable",
            ) from exc

    def get_version(self, dataset_version_id: str) -> AnnotationDatasetVersion:
        try:
            with self._sessions() as session:
                row = session.get(DatasetVersion, dataset_version_id)
                if row is None or not _is_annotation_manifest(row.manifest_json):
                    raise NotFoundError(
                        "annotation dataset version does not exist",
                        code="annotation_dataset_not_found",
                    )
                return AnnotationDatasetVersion.model_validate(row.manifest_json)
        except SQLAlchemyError as exc:
            raise RetryableError(
                "annotation dataset storage is unavailable",
                code="annotation_dataset_storage_unavailable",
            ) from exc

    def list_versions(
        self,
        dataset_name: str | None = None,
        status: DatasetStatus | None = None,
        sample_status: AnnotationSampleStatus | None = None,
    ) -> tuple[AnnotationDatasetVersion, ...]:
        try:
            with self._sessions() as session:
                statement = select(DatasetVersion).order_by(
                    DatasetVersion.dataset_name, DatasetVersion.version_number
                )
                if dataset_name is not None:
                    statement = statement.where(DatasetVersion.dataset_name == dataset_name)
                if status is not None:
                    statement = statement.where(DatasetVersion.status == status.value)
                if sample_status is not None:
                    matching_ids = select(
                        DatasetAnnotationSampleRecord.dataset_version_id
                    ).where(DatasetAnnotationSampleRecord.status == sample_status.value)
                    statement = statement.where(DatasetVersion.id.in_(matching_ids))
                rows = session.scalars(statement)
                return tuple(
                    AnnotationDatasetVersion.model_validate(row.manifest_json)
                    for row in rows
                    if _is_annotation_manifest(row.manifest_json)
                )
        except SQLAlchemyError as exc:
            raise RetryableError(
                "annotation dataset storage is unavailable",
                code="annotation_dataset_storage_unavailable",
            ) from exc

    def replace_version(
        self,
        version: AnnotationDatasetVersion,
        *,
        expected_manifest_sha256: str,
    ) -> AnnotationDatasetVersion:
        try:
            with self._sessions.begin() as session:
                row = session.get(DatasetVersion, version.dataset_version_id)
                if row is None or not _is_annotation_manifest(row.manifest_json):
                    raise NotFoundError(
                        "annotation dataset version does not exist",
                        code="annotation_dataset_not_found",
                    )
                if row.manifest_hash != expected_manifest_sha256:
                    raise ConflictError(
                        "annotation dataset changed concurrently",
                        code="annotation_dataset_stale_write",
                    )
                if row.status == DatasetStatus.FROZEN.value:
                    raise ConflictError(
                        "frozen annotation datasets are immutable",
                        code="annotation_dataset_frozen",
                    )
                row.manifest_hash = version.manifest_sha256
                row.status = version.status.value
                row.provenance_json = version.provenance.model_dump(mode="json")
                row.manifest_json = version.model_dump(mode="json")
                row.frozen_at = version.frozen_at

                existing = {
                    item.sample_key: item
                    for item in session.scalars(
                        select(DatasetAnnotationSampleRecord).where(
                            DatasetAnnotationSampleRecord.dataset_version_id
                            == version.dataset_version_id
                        )
                    )
                }
                for sample in version.samples:
                    projection = existing.get(sample.sample_id)
                    if projection is None:
                        raise ConflictError(
                            "annotation sample projection is missing",
                            code="annotation_sample_projection_missing",
                        )
                    _update_sample_row(projection, sample)
                session.flush()
                return version
        except (ConflictError, NotFoundError):
            raise
        except IntegrityError as exc:
            raise ConflictError(
                "invalid annotation dataset update",
                code="annotation_dataset_conflict",
            ) from exc
        except SQLAlchemyError as exc:
            raise RetryableError(
                "annotation dataset storage is unavailable",
                code="annotation_dataset_storage_unavailable",
            ) from exc

    @staticmethod
    def _sample_rows(
        version: AnnotationDatasetVersion,
    ) -> tuple[DatasetAnnotationSampleRecord, ...]:
        rows: list[DatasetAnnotationSampleRecord] = []
        for sample in version.samples:
            row = DatasetAnnotationSampleRecord(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"tender-review:a3:{version.dataset_version_id}:{sample.sample_id}",
                    )
                ),
                dataset_version_id=version.dataset_version_id,
                sample_key=sample.sample_id,
                finding_id=sample.finding_id,
                document_snapshot_id=sample.document_snapshot_id,
                source_pdf_reference=sample.source_pdf_reference,
                document_sha256=sample.document_sha256,
                source_case_sha256=sample.source_case_sha256,
                rule_version_id=sample.rule_version_id,
                rule_sha256=sample.rule_sha256,
                query_id=sample.query_id,
                question_label=sample.question_label,
                split=sample.split.value,
                status=sample.status.value,
                evidence_catalog_sha256=sample.evidence_catalog_sha256,
                final_label_sha256=sample.final_label_sha256,
                label_version=sample.label_version,
                sample_sha256=sample.sample_sha256,
                sample_json=sample.model_dump(mode="json"),
                created_at=version.created_at,
                updated_at=version.created_at,
                schema_version=str(sample.schema_version),
            )
            _update_sample_row(row, sample)
            rows.append(row)
        return tuple(rows)


def _is_annotation_manifest(value: object) -> bool:
    return isinstance(value, dict) and value.get("manifest_kind") == "a3_annotation_dataset"


def _update_sample_row(row: DatasetAnnotationSampleRecord, sample: object) -> None:
    row.status = sample.status.value
    row.annotation_human_decision_id = (
        sample.annotation.human_decision_id if sample.annotation else None
    )
    row.review_human_decision_id = (
        sample.review.human_decision_id if sample.review else None
    )
    row.adjudication_human_decision_id = (
        sample.adjudication.human_decision_id if sample.adjudication else None
    )
    row.annotator_id = sample.annotation.actor_id if sample.annotation else None
    row.annotated_at = sample.annotation.acted_at if sample.annotation else None
    row.reviewer_id = sample.review.actor_id if sample.review else None
    row.reviewed_at = sample.review.acted_at if sample.review else None
    row.adjudicator_id = sample.adjudication.actor_id if sample.adjudication else None
    row.adjudicated_at = sample.adjudication.acted_at if sample.adjudication else None
    row.final_label_sha256 = sample.final_label_sha256
    row.label_version = sample.label_version
    row.sample_sha256 = sample.sample_sha256
    row.sample_json = sample.model_dump(mode="json")
