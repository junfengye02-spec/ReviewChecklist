from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

from tender_review.shared.clock import Clock
from tender_review.shared.contracts import ContractModel
from tender_review.shared.errors import ConflictError, NotFoundError, PermanentError
from tender_review.shared.ids import IdGenerator
from tender_review.findings.public import DecisionOutcome


SHA256_PATTERN = r"^[0-9a-f]{64}$"


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def dataset_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _parse_canonical_object(value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError(f"{name} must be a canonical JSON object")
    return parsed


class DatasetSplit(str, Enum):
    OPTIMIZATION = "OPTIMIZATION"
    VALIDATION = "VALIDATION"
    FROZEN_TEST = "FROZEN_TEST"


class DatasetStatus(str, Enum):
    DRAFT = "DRAFT"
    PROVISIONAL = "PROVISIONAL"
    VERIFIED = "VERIFIED"
    FROZEN = "FROZEN"


class DatasetSourceType(str, Enum):
    REAL = "REAL"
    EXTERNAL_PLATFORM = "EXTERNAL_PLATFORM"
    SYNTHETIC = "SYNTHETIC"
    PROVISIONAL = "PROVISIONAL"


class DatasetProvenance(ContractModel):
    status: Literal["provisional", "verified"]
    claims_allowed: bool
    source_description: str = Field(min_length=1, max_length=8000)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_dataset_version_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("source_description")
    @classmethod
    def source_description_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_description must not be blank")
        return value

    @model_validator(mode="after")
    def provisional_boundary_is_preserved(self) -> Self:
        if self.status == "provisional" and self.claims_allowed:
            raise ValueError("provisional dataset provenance cannot allow claims")
        return self


class DatasetDocument(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    split: DatasetSplit
    source_type: DatasetSourceType


class DatasetSample(ContractModel):
    sample_id: str = Field(min_length=1, max_length=128)
    finding_id: str | None = Field(default=None, min_length=1, max_length=128)
    human_decision_id: str | None = Field(default=None, min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    split: DatasetSplit
    source_type: DatasetSourceType
    provenance_status: Literal["provisional", "verified"]
    label_version: str = Field(min_length=1, max_length=128)
    label_json: str = Field(min_length=2)
    review_input_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    sample_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def label_and_source_are_consistent(self) -> Self:
        label = _parse_canonical_object(self.label_json, "label_json")
        if self.source_type is DatasetSourceType.REAL and (
            self.finding_id is None or self.human_decision_id is None
        ):
            raise ValueError("REAL samples require finding and human decision provenance")
        if self.source_type is not DatasetSourceType.REAL and self.human_decision_id is not None:
            raise ValueError("non-real samples cannot claim a human decision")
        payload = self.model_dump(mode="json", exclude={"sample_sha256"})
        payload["label_json"] = label
        if self.sample_sha256 != dataset_sha256(payload):
            raise ValueError("sample_sha256 does not match sample content")
        return self


class DatasetVersion(ContractModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    dataset_name: str = Field(min_length=1, max_length=255)
    version_number: int = Field(ge=1)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    status: DatasetStatus
    change_summary: str = Field(min_length=1, max_length=8000)
    provenance: DatasetProvenance
    documents: tuple[DatasetDocument, ...] = Field(min_length=1)
    samples: tuple[DatasetSample, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime
    frozen_at: datetime | None = None

    @field_validator("change_summary")
    @classmethod
    def change_summary_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("change_summary must not be blank")
        return value

    @model_validator(mode="after")
    def manifest_is_immutable_and_leak_free(self) -> Self:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.version_number == 1 and self.parent_version_id is not None:
            raise ValueError("first dataset version cannot have a parent")
        if self.version_number > 1 and self.parent_version_id is None:
            raise ValueError("later dataset versions require a parent")
        document_map: dict[str, DatasetDocument] = {}
        for document in self.documents:
            existing = document_map.get(document.document_id)
            if existing is not None and existing != document:
                raise ValueError("one source document cannot cross dataset splits")
            document_map[document.document_id] = document
        if len(document_map) != len(self.documents):
            raise ValueError("dataset document manifest contains duplicates")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("dataset sample IDs must be unique")
        for sample in self.samples:
            document = document_map.get(sample.document_id)
            if document is None:
                raise ValueError("sample references a document outside the manifest")
            if (
                document.document_sha256 != sample.document_sha256
                or document.split is not sample.split
                or document.source_type is not sample.source_type
            ):
                raise ValueError("sample conflicts with document-level split or source")
        contains_unverified = any(
            item.source_type is not DatasetSourceType.REAL
            or item.human_decision_id is None
            or item.provenance_status == "provisional"
            for item in self.samples
        )
        if contains_unverified and self.status not in {
            DatasetStatus.DRAFT,
            DatasetStatus.PROVISIONAL,
        }:
            raise ValueError("unverified samples can only form draft/provisional datasets")
        if self.provenance.status == "provisional" and self.status not in {
            DatasetStatus.DRAFT,
            DatasetStatus.PROVISIONAL,
        }:
            raise ValueError("provisional provenance cannot become a real frozen dataset")
        if self.status is DatasetStatus.FROZEN:
            if self.frozen_at is None or not self.provenance.claims_allowed:
                raise ValueError("frozen dataset requires verified claimable provenance")
        elif self.frozen_at is not None:
            raise ValueError("only FROZEN datasets may have frozen_at")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != dataset_sha256(payload):
            raise ValueError("manifest_sha256 does not match dataset version")
        return self


class DatasetSampleInput(ContractModel):
    sample_id: str
    finding_id: str | None = None
    human_decision_id: str | None = None
    document_id: str
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    split: DatasetSplit
    source_type: DatasetSourceType
    provenance_status: Literal["provisional", "verified"]
    label_version: str
    label_json: str
    review_input_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)


class CreateDatasetVersion(ContractModel):
    dataset_name: str = Field(min_length=1, max_length=255)
    parent_version_id: str | None = None
    requested_status: DatasetStatus
    change_summary: str = Field(min_length=1, max_length=8000)
    provenance: DatasetProvenance
    samples: tuple[DatasetSampleInput, ...] = Field(min_length=1)

    @field_validator("change_summary")
    @classmethod
    def change_summary_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("change_summary must not be blank")
        return value


@runtime_checkable
class DatasetVersionRepository(Protocol):
    def get_version(self, dataset_version_id: str) -> DatasetVersion: ...

    def list_versions(self, dataset_name: str) -> tuple[DatasetVersion, ...]: ...

    def add_version(self, version: DatasetVersion) -> DatasetVersion: ...


class InMemoryDatasetVersionRepository:
    def __init__(self) -> None:
        self._versions: dict[str, DatasetVersion] = {}
        self._lock = RLock()

    def get_version(self, dataset_version_id: str) -> DatasetVersion:
        with self._lock:
            try:
                return self._versions[dataset_version_id]
            except KeyError as exc:
                raise NotFoundError("dataset version does not exist", code="dataset_version_not_found") from exc

    def list_versions(self, dataset_name: str) -> tuple[DatasetVersion, ...]:
        with self._lock:
            return tuple(sorted(
                (item for item in self._versions.values() if item.dataset_name == dataset_name),
                key=lambda item: item.version_number,
            ))

    def add_version(self, version: DatasetVersion) -> DatasetVersion:
        with self._lock:
            if version.dataset_version_id in self._versions:
                raise ConflictError("dataset version already exists", code="dataset_version_conflict")
            if any(
                item.dataset_name == version.dataset_name
                and (item.version_number == version.version_number or item.manifest_sha256 == version.manifest_sha256)
                for item in self._versions.values()
            ):
                raise ConflictError("duplicate dataset version", code="dataset_version_duplicate")
            self._versions[version.dataset_version_id] = version
            return version


class DatasetVersionService:
    def __init__(self, repository: DatasetVersionRepository, ids: IdGenerator, clock: Clock) -> None:
        self._repository = repository
        self._ids = ids
        self._clock = clock

    def create_version(self, command: CreateDatasetVersion) -> DatasetVersion:
        versions = self._repository.list_versions(command.dataset_name)
        expected_parent = versions[-1].dataset_version_id if versions else None
        if command.parent_version_id != expected_parent:
            raise ConflictError(
                "parent_version_id must reference the latest dataset version",
                code="dataset_parent_not_latest",
                details={"expected_parent_version_id": expected_parent},
            )
        now = self._clock.now()
        samples: list[DatasetSample] = []
        documents: dict[str, DatasetDocument] = {}
        for value in command.samples:
            label = _parse_canonical_object(value.label_json, "label_json")
            sample_payload = value.model_dump(mode="json")
            sample_payload["label_json"] = label
            sample = DatasetSample(
                **value.model_dump(),
                sample_sha256=dataset_sha256(sample_payload),
            )
            samples.append(sample)
            document = DatasetDocument(
                document_id=value.document_id,
                document_sha256=value.document_sha256,
                split=value.split,
                source_type=value.source_type,
            )
            existing = documents.get(value.document_id)
            if existing is not None and existing != document:
                raise PermanentError(
                    "all samples from one source document must use one split",
                    code="dataset_document_leakage",
                    details={"document_id": value.document_id},
                )
            documents[value.document_id] = document
        contains_unverified = any(
            item.source_type is not DatasetSourceType.REAL
            or item.human_decision_id is None
            or item.provenance_status == "provisional"
            for item in samples
        )
        if contains_unverified and command.requested_status not in {
            DatasetStatus.DRAFT,
            DatasetStatus.PROVISIONAL,
        }:
            raise PermanentError(
                "without real human labels the dataset must remain draft/provisional",
                code="dataset_real_freeze_forbidden",
            )
        payload = {
            "schema_version": 1,
            "dataset_version_id": self._ids.new(),
            "dataset_name": command.dataset_name,
            "version_number": len(versions) + 1,
            "parent_version_id": command.parent_version_id,
            "status": command.requested_status,
            "change_summary": command.change_summary,
            "provenance": command.provenance.model_dump(mode="json"),
            "documents": [item.model_dump(mode="json") for item in sorted(documents.values(), key=lambda item: item.document_id)],
            "samples": [item.model_dump(mode="json") for item in sorted(samples, key=lambda item: item.sample_id)],
            "created_at": now,
            "frozen_at": now if command.requested_status is DatasetStatus.FROZEN else None,
        }
        version = DatasetVersion(**payload, manifest_sha256=dataset_sha256(payload))
        return self._repository.add_version(version)


def deterministic_document_splits(
    document_ids: tuple[str, ...],
    *,
    optimization_percent: int = 60,
    validation_percent: int = 20,
) -> dict[str, DatasetSplit]:
    if not 0 <= optimization_percent <= 100 or not 0 <= validation_percent <= 100:
        raise ValueError("split percentages must be between 0 and 100")
    if optimization_percent + validation_percent > 100:
        raise ValueError("split percentages must not exceed 100")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document IDs must be unique")
    result: dict[str, DatasetSplit] = {}
    for document_id in sorted(document_ids):
        bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < optimization_percent:
            split = DatasetSplit.OPTIMIZATION
        elif bucket < optimization_percent + validation_percent:
            split = DatasetSplit.VALIDATION
        else:
            split = DatasetSplit.FROZEN_TEST
        result[document_id] = split
    return result


def samples_from_human_decision(
    outcome: DecisionOutcome,
    *,
    split: DatasetSplit,
) -> tuple[DatasetSampleInput, ...]:
    """Project a real submitted decision into immutable document-scoped samples."""

    finding = outcome.finding
    decision = outcome.decision
    label = {
        "decision": decision.decision.value,
        "conclusion": finding.conclusion,
        "message": finding.message,
        "revision": decision.revision.model_dump(mode="json") if decision.revision else None,
    }
    label_json = canonical_json(label)
    return tuple(
        DatasetSampleInput(
            sample_id=f"{decision.decision_id}:{document.document_id}",
            finding_id=finding.finding_id,
            human_decision_id=decision.decision_id,
            document_id=document.document_id,
            document_sha256=document.document_sha256,
            split=split,
            source_type=DatasetSourceType.REAL,
            provenance_status=finding.provenance.status,
            label_version=decision.decision_sha256,
            label_json=label_json,
            review_input_sha256=decision.review_input_sha256,
            evidence_sha256=decision.evidence_sha256,
        )
        for document in finding.documents
    )
