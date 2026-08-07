from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.evaluation.public import DatasetSourceType, DatasetVersion as DatasetVersionDto
from tender_review.shared.errors import (
    ConflictError,
    NotFoundError,
    PermanentError,
    RetryableError,
)

from .models import DatasetVersion, EvaluationCase


class SqlAlchemyDatasetVersionRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def get_version(self, dataset_version_id: str) -> DatasetVersionDto:
        try:
            with self._sessions() as session:
                row = session.get(DatasetVersion, dataset_version_id)
                if row is None:
                    raise NotFoundError("dataset version does not exist", code="dataset_version_not_found")
                if (row.manifest_json or {}).get("manifest_kind") == "a3_annotation_dataset":
                    raise PermanentError(
                        "A3 annotation datasets are not evaluation-ready until A4 projection",
                        code="annotation_dataset_not_evaluation_ready",
                    )
                return DatasetVersionDto.model_validate(row.manifest_json)
        except SQLAlchemyError as exc:
            raise RetryableError("dataset storage is unavailable", code="dataset_storage_unavailable") from exc

    def list_versions(self, dataset_name: str) -> tuple[DatasetVersionDto, ...]:
        try:
            with self._sessions() as session:
                rows = session.scalars(
                    select(DatasetVersion)
                    .where(DatasetVersion.dataset_name == dataset_name)
                    .order_by(DatasetVersion.version_number)
                )
                return tuple(
                    DatasetVersionDto.model_validate(row.manifest_json)
                    for row in rows
                    if (row.manifest_json or {}).get("manifest_kind")
                    != "a3_annotation_dataset"
                )
        except SQLAlchemyError as exc:
            raise RetryableError("dataset storage is unavailable", code="dataset_storage_unavailable") from exc

    def add_version(self, version: DatasetVersionDto) -> DatasetVersionDto:
        source_types = {item.source_type for item in version.samples}
        if source_types == {DatasetSourceType.REAL}:
            aggregate_source = "REAL"
        elif source_types == {DatasetSourceType.SYNTHETIC}:
            aggregate_source = "SYNTHETIC"
        else:
            aggregate_source = "MIXED"
        try:
            with self._sessions.begin() as session:
                session.add(DatasetVersion(
                    id=version.dataset_version_id,
                    dataset_name=version.dataset_name,
                    version_number=version.version_number,
                    parent_version_id=version.parent_version_id,
                    manifest_hash=version.manifest_sha256,
                    source_type=aggregate_source,
                    status=version.status.value,
                    change_summary=version.change_summary,
                    provenance_json=version.provenance.model_dump(mode="json"),
                    manifest_json=version.model_dump(mode="json"),
                    split_strategy_json={
                        item.document_id: item.split.value for item in version.documents
                    },
                    manifest_artifact_id=None,
                    frozen_at=version.frozen_at,
                    created_at=version.created_at,
                    updated_at=version.created_at,
                    schema_version=str(version.schema_version),
                ))
                for sample in version.samples:
                    source_type = {
                        DatasetSourceType.REAL: "REAL",
                        DatasetSourceType.SYNTHETIC: "SYNTHETIC",
                        DatasetSourceType.EXTERNAL_PLATFORM: "EXTERNAL_PLATFORM",
                        DatasetSourceType.PROVISIONAL: "EXTERNAL_PLATFORM",
                    }[sample.source_type]
                    session.add(EvaluationCase(
                        id=str(uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"tender-review:{version.dataset_version_id}:{sample.sample_id}",
                        )),
                        dataset_version_id=version.dataset_version_id,
                        document_snapshot_id=sample.document_id,
                        case_key=sample.sample_id,
                        review_item=sample.finding_id or "provisional",
                        split=sample.split.value,
                        source_type=source_type,
                        expected_compliant=json.loads(sample.label_json).get("compliant"),
                        expected_finding_json=json.loads(sample.label_json),
                        expected_evidence_json={},
                        finding_id=sample.finding_id,
                        human_decision_id=sample.human_decision_id,
                        document_sha256=sample.document_sha256,
                        label_version=sample.label_version,
                        label_status=sample.provenance_status,
                        review_input_sha256=sample.review_input_sha256,
                        evidence_sha256=sample.evidence_sha256,
                        sample_sha256=sample.sample_sha256,
                        schema_version=str(sample.schema_version),
                    ))
                session.flush()
                return version
        except IntegrityError as exc:
            raise ConflictError("duplicate or invalid dataset version", code="dataset_version_conflict") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("dataset storage is unavailable", code="dataset_storage_unavailable") from exc
