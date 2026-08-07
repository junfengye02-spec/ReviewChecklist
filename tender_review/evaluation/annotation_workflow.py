from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from tender_review.findings.public import FindingRepository, HumanDecision
from tender_review.rule_management.public import RuleVersionRepository
from tender_review.shared.clock import Clock
from tender_review.shared.contracts import ContractModel
from tender_review.shared.errors import ConflictError, NotFoundError, PermanentError
from tender_review.shared.ids import IdGenerator

from .dataset_versioning import DatasetSplit, DatasetStatus, dataset_sha256


SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value.strip()


def _named_human(value: str) -> str:
    normalized = _not_blank(value)
    first_token = normalized.casefold()
    for separator in (":", "/", "_", "-"):
        first_token = first_token.split(separator, 1)[0]
    if first_token in {
        "ai",
        "assistant",
        "anonymous",
        "bot",
        "fake",
        "model",
        "provisional",
        "service",
        "synthetic",
        "system",
    }:
        raise ValueError("a named human identity is required")
    return normalized


class AnnotationSampleStatus(str, Enum):
    PENDING_ANNOTATION = "PENDING_ANNOTATION"
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFLICT = "CONFLICT"
    VERIFIED = "VERIFIED"
    FROZEN = "FROZEN"


class AnnotationEvidenceChunk(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: tuple[str, ...] = Field(default=(), max_length=32)
    excerpt: str = Field(min_length=1, max_length=8000)
    text_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("chunk_id", "excerpt")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def reference_is_locatable_and_untampered(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        if any(not part.strip() for part in self.section_path):
            raise ValueError("section_path must not contain blank values")
        if self.text_sha256 != hashlib.sha256(self.excerpt.encode("utf-8")).hexdigest():
            raise ValueError("text_sha256 does not match excerpt")
        return self


class ChunkRelevanceLabel(ContractModel):
    no_answer: bool
    relevant_chunk_ids: tuple[str, ...] = ()
    label_schema_version: Literal["chunk-relevance.v1"] = "chunk-relevance.v1"

    @model_validator(mode="after")
    def relevance_is_consistent(self) -> Self:
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids must be unique")
        if self.no_answer and self.relevant_chunk_ids:
            raise ValueError("no-answer labels cannot contain relevant chunks")
        if not self.no_answer and not self.relevant_chunk_ids:
            raise ValueError("answerable labels require at least one relevant chunk")
        return self


class HumanLabelRecord(ContractModel):
    human_decision_id: str = Field(min_length=1, max_length=128)
    human_decision_sha256: str = Field(pattern=SHA256_PATTERN)
    human_decision_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    actor_id: str = Field(min_length=1, max_length=255)
    acted_at: datetime
    label: ChunkRelevanceLabel
    label_sha256: str = Field(pattern=SHA256_PATTERN)
    relevant_evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("actor_id")
    @classmethod
    def actor_is_named_human(cls, value: str) -> str:
        return _named_human(value)

    @model_validator(mode="after")
    def record_is_hashed(self) -> Self:
        if self.acted_at.tzinfo is None:
            raise ValueError("acted_at must be timezone-aware")
        if self.label_sha256 != dataset_sha256(self.label.model_dump(mode="json")):
            raise ValueError("label_sha256 does not match label")
        return self


class AnnotationSampleInput(ContractModel):
    sample_id: str = Field(min_length=1, max_length=128)
    finding_id: str = Field(min_length=1, max_length=128)
    document_snapshot_id: str = Field(min_length=1, max_length=128)
    source_pdf_reference: str = Field(min_length=1, max_length=2048)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    source_case_sha256: str = Field(pattern=SHA256_PATTERN)
    rule_version_id: str = Field(min_length=1, max_length=128)
    rule_sha256: str = Field(pattern=SHA256_PATTERN)
    query_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8000)
    question_label: str = Field(min_length=1, max_length=512)
    split: DatasetSplit
    candidate_chunks: tuple[AnnotationEvidenceChunk, ...] = Field(min_length=1)

    @field_validator(
        "sample_id",
        "finding_id",
        "document_snapshot_id",
        "source_pdf_reference",
        "rule_version_id",
        "query_id",
        "query",
        "question_label",
    )
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def chunks_are_unique(self) -> Self:
        chunk_ids = tuple(item.chunk_id for item in self.candidate_chunks)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("candidate chunk IDs must be unique")
        return self


class DatasetAnnotationSample(AnnotationSampleInput):
    status: AnnotationSampleStatus = AnnotationSampleStatus.PENDING_ANNOTATION
    evidence_catalog_sha256: str = Field(pattern=SHA256_PATTERN)
    annotation: HumanLabelRecord | None = None
    review: HumanLabelRecord | None = None
    adjudication: HumanLabelRecord | None = None
    final_label_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    label_version: str | None = Field(default=None, min_length=1, max_length=128)
    sample_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def state_and_hashes_are_consistent(self) -> Self:
        chunk_payload = [
            item.model_dump(mode="json")
            for item in sorted(self.candidate_chunks, key=lambda item: item.chunk_id)
        ]
        if self.evidence_catalog_sha256 != dataset_sha256(chunk_payload):
            raise ValueError("evidence_catalog_sha256 does not match candidate chunks")
        records = tuple(
            item for item in (self.annotation, self.review, self.adjudication) if item
        )
        actor_ids = tuple(item.actor_id for item in records)
        decision_ids = tuple(item.human_decision_id for item in records)
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("annotation, review, and adjudication require different people")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("each human action requires a distinct HumanDecision")

        final_hash: str | None = None
        if self.status is AnnotationSampleStatus.PENDING_ANNOTATION:
            if records:
                raise ValueError("pending annotation samples cannot contain human records")
        elif self.status is AnnotationSampleStatus.PENDING_REVIEW:
            if self.annotation is None or self.review is not None or self.adjudication is not None:
                raise ValueError("pending review requires only an annotation")
        elif self.status is AnnotationSampleStatus.CONFLICT:
            if self.annotation is None or self.review is None or self.adjudication is not None:
                raise ValueError("conflict requires annotation and independent review")
            if self.annotation.label_sha256 == self.review.label_sha256:
                raise ValueError("matching labels cannot enter conflict")
        elif self.status in {
            AnnotationSampleStatus.VERIFIED,
            AnnotationSampleStatus.FROZEN,
        }:
            if self.annotation is None or self.review is None:
                raise ValueError("verified samples require annotation and independent review")
            if self.annotation.label_sha256 == self.review.label_sha256:
                if self.adjudication is not None:
                    raise ValueError("matching labels do not require adjudication")
                final_hash = self.annotation.label_sha256
            else:
                if self.adjudication is None:
                    raise ValueError("conflicting labels require independent adjudication")
                final_hash = self.adjudication.label_sha256

        if final_hash != self.final_label_sha256:
            raise ValueError("final_label_sha256 does not match workflow state")
        expected_version = (
            f"chunk-relevance.v1:{final_hash}" if final_hash is not None else None
        )
        if self.label_version != expected_version:
            raise ValueError("label_version does not match final label")
        payload = self.model_dump(mode="json", exclude={"sample_sha256"})
        if self.sample_sha256 != dataset_sha256(payload):
            raise ValueError("sample_sha256 does not match sample content")
        return self


class AnnotationDatasetProvenance(ContractModel):
    source_description: str = Field(min_length=1, max_length=8000)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal["provisional", "verified"] = "provisional"
    claims_allowed: bool = False

    @field_validator("source_description")
    @classmethod
    def description_is_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def claim_boundary_is_consistent(self) -> Self:
        if self.claims_allowed and self.status != "verified":
            raise ValueError("only verified annotation provenance may allow claims")
        return self


class AnnotationDatasetVersion(ContractModel):
    manifest_kind: Literal["a3_annotation_dataset"] = "a3_annotation_dataset"
    dataset_version_id: str = Field(min_length=1, max_length=128)
    dataset_name: str = Field(min_length=1, max_length=255)
    version_number: int = Field(ge=1)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    status: DatasetStatus
    change_summary: str = Field(min_length=1, max_length=8000)
    required_human_cases: int = Field(ge=1)
    provenance: AnnotationDatasetProvenance
    samples: tuple[DatasetAnnotationSample, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime
    frozen_at: datetime | None = None

    @field_validator("dataset_name", "change_summary")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def manifest_is_consistent(self) -> Self:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.version_number == 1 and self.parent_version_id is not None:
            raise ValueError("first dataset version cannot have a parent")
        if self.version_number > 1 and self.parent_version_id is None:
            raise ValueError("later dataset versions require a parent")
        if self.required_human_cases != len(self.samples):
            raise ValueError("required_human_cases must equal the version sample count")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("dataset sample IDs must be unique")
        human_decision_ids = tuple(
            record.human_decision_id
            for sample in self.samples
            for record in (sample.annotation, sample.review, sample.adjudication)
            if record is not None
        )
        if len(human_decision_ids) != len(set(human_decision_ids)):
            raise ValueError("one HumanDecision cannot label multiple dataset samples")
        document_splits: dict[str, DatasetSplit] = {}
        for sample in self.samples:
            existing = document_splits.get(sample.document_sha256)
            if existing is not None and existing is not sample.split:
                raise ValueError("one document hash cannot cross dataset splits")
            document_splits[sample.document_sha256] = sample.split

        statuses = {item.status for item in self.samples}
        if self.status is DatasetStatus.DRAFT:
            if AnnotationSampleStatus.FROZEN in statuses:
                raise ValueError("draft datasets cannot contain frozen samples")
            if self.frozen_at is not None:
                raise ValueError("draft datasets cannot have frozen_at")
        elif self.status is DatasetStatus.VERIFIED:
            if statuses != {AnnotationSampleStatus.VERIFIED}:
                raise ValueError("VERIFIED datasets may contain only VERIFIED samples")
            if self.provenance.status != "verified" or self.frozen_at is not None:
                raise ValueError("VERIFIED dataset provenance is inconsistent")
        elif self.status is DatasetStatus.FROZEN:
            if statuses != {AnnotationSampleStatus.FROZEN}:
                raise ValueError("FROZEN datasets may contain only FROZEN samples")
            if self.provenance.status != "verified" or self.frozen_at is None:
                raise ValueError("FROZEN dataset provenance is inconsistent")
            if DatasetSplit.FROZEN_TEST not in {item.split for item in self.samples}:
                raise ValueError("FROZEN datasets require a frozen-test split")
            expected_claims = self.required_human_cases == len(self.samples)
            if self.provenance.claims_allowed != expected_claims:
                raise ValueError(
                    "FROZEN claims require every sample to be independently verified"
                )
        else:
            raise ValueError("A3 annotation datasets use DRAFT, VERIFIED, or FROZEN")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != dataset_sha256(payload):
            raise ValueError("manifest_sha256 does not match dataset version")
        return self


class CreateAnnotationDatasetVersion(ContractModel):
    dataset_name: str = Field(min_length=1, max_length=255)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    change_summary: str = Field(min_length=1, max_length=8000)
    source_description: str = Field(min_length=1, max_length=8000)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_work_package_sha256: str = Field(pattern=SHA256_PATTERN)
    samples: tuple[AnnotationSampleInput, ...] = Field(min_length=1)


class SubmitAnnotationLabel(ContractModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    sample_id: str = Field(min_length=1, max_length=128)
    actor_id: str = Field(min_length=1, max_length=255)
    human_decision_id: str = Field(min_length=1, max_length=128)
    label: ChunkRelevanceLabel

    @field_validator("actor_id")
    @classmethod
    def actor_is_named_human(cls, value: str) -> str:
        return _named_human(value)


class CreateAnnotationDatasetRevision(ContractModel):
    parent_version_id: str = Field(min_length=1, max_length=128)
    change_summary: str = Field(min_length=1, max_length=8000)
    reset_sample_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reset_ids_are_unique(self) -> Self:
        if len(self.reset_sample_ids) != len(set(self.reset_sample_ids)):
            raise ValueError("reset_sample_ids must be unique")
        return self


@runtime_checkable
class AnnotationDatasetRepository(Protocol):
    def add_version(self, version: AnnotationDatasetVersion) -> AnnotationDatasetVersion: ...

    def get_version(self, dataset_version_id: str) -> AnnotationDatasetVersion: ...

    def list_versions(
        self,
        dataset_name: str | None = None,
        status: DatasetStatus | None = None,
        sample_status: AnnotationSampleStatus | None = None,
    ) -> tuple[AnnotationDatasetVersion, ...]: ...

    def replace_version(
        self,
        version: AnnotationDatasetVersion,
        *,
        expected_manifest_sha256: str,
    ) -> AnnotationDatasetVersion: ...


class InMemoryAnnotationDatasetRepository:
    def __init__(self) -> None:
        self._versions: dict[str, AnnotationDatasetVersion] = {}
        self._lock = RLock()

    def add_version(self, version: AnnotationDatasetVersion) -> AnnotationDatasetVersion:
        with self._lock:
            if version.dataset_version_id in self._versions:
                raise ConflictError("dataset version already exists", code="dataset_version_conflict")
            if any(
                item.dataset_name == version.dataset_name
                and item.version_number == version.version_number
                for item in self._versions.values()
            ):
                raise ConflictError("dataset version number already exists", code="dataset_version_duplicate")
            self._versions[version.dataset_version_id] = version
            return version

    def get_version(self, dataset_version_id: str) -> AnnotationDatasetVersion:
        with self._lock:
            try:
                return self._versions[dataset_version_id]
            except KeyError as exc:
                raise NotFoundError(
                    "annotation dataset version does not exist",
                    code="annotation_dataset_not_found",
                ) from exc

    def list_versions(
        self,
        dataset_name: str | None = None,
        status: DatasetStatus | None = None,
        sample_status: AnnotationSampleStatus | None = None,
    ) -> tuple[AnnotationDatasetVersion, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        item
                        for item in self._versions.values()
                        if (dataset_name is None or item.dataset_name == dataset_name)
                        and (status is None or item.status is status)
                        and (
                            sample_status is None
                            or any(sample.status is sample_status for sample in item.samples)
                        )
                    ),
                    key=lambda item: (item.dataset_name, item.version_number),
                )
            )

    def replace_version(
        self,
        version: AnnotationDatasetVersion,
        *,
        expected_manifest_sha256: str,
    ) -> AnnotationDatasetVersion:
        with self._lock:
            current = self.get_version(version.dataset_version_id)
            if current.manifest_sha256 != expected_manifest_sha256:
                raise ConflictError(
                    "annotation dataset changed concurrently",
                    code="annotation_dataset_stale_write",
                )
            if current.status is DatasetStatus.FROZEN:
                raise ConflictError(
                    "frozen annotation datasets are immutable",
                    code="annotation_dataset_frozen",
                )
            self._versions[version.dataset_version_id] = version
            return version


@runtime_checkable
class HumanDecisionResolver(Protocol):
    def get_decision(self, finding_id: str, decision_id: str) -> HumanDecision: ...


@runtime_checkable
class AnnotationReferenceValidator(Protocol):
    def validate_sample(self, sample: AnnotationSampleInput) -> None: ...


@runtime_checkable
class DocumentSnapshotResolver(Protocol):
    def get_snapshot(self, snapshot_id: str) -> Any: ...


class RepositoryHumanDecisionResolver:
    def __init__(self, repository: FindingRepository) -> None:
        self._repository = repository

    def get_decision(self, finding_id: str, decision_id: str) -> HumanDecision:
        for decision in self._repository.list_decisions(finding_id):
            if decision.decision_id == decision_id:
                return decision
        raise NotFoundError(
            "HumanDecision does not exist for the sample finding",
            code="annotation_human_decision_not_found",
        )


class RepositoryAnnotationReferenceValidator:
    def __init__(
        self,
        documents: DocumentSnapshotResolver,
        findings: FindingRepository,
        rules: RuleVersionRepository,
    ) -> None:
        self._documents = documents
        self._findings = findings
        self._rules = rules

    def validate_sample(self, sample: AnnotationSampleInput) -> None:
        snapshot = self._documents.get_snapshot(sample.document_snapshot_id)
        if snapshot.object.sha256 != sample.document_sha256:
            raise PermanentError(
                "document_sha256 does not match the source PDF snapshot",
                code="annotation_document_hash_mismatch",
            )
        if snapshot.object.object_key != sample.source_pdf_reference:
            raise PermanentError(
                "source_pdf_reference must be the snapshot content-addressed object key",
                code="annotation_pdf_reference_mismatch",
            )
        finding = self._findings.get_finding(sample.finding_id)
        finding_documents = {
            item.document_id: item.document_sha256 for item in finding.documents
        }
        if finding_documents.get(sample.document_snapshot_id) != sample.document_sha256:
            raise PermanentError(
                "sample document is not present in the referenced Finding",
                code="annotation_finding_document_mismatch",
            )
        if finding.rule_version_id != sample.rule_version_id:
            raise PermanentError(
                "sample rule version does not match the referenced Finding",
                code="annotation_finding_rule_mismatch",
            )
        rule = self._rules.get_version(sample.rule_version_id)
        if rule.content_sha256 != sample.rule_sha256:
            raise PermanentError(
                "rule_sha256 does not match the immutable RuleVersion",
                code="annotation_rule_hash_mismatch",
            )


class AnnotationDatasetService:
    def __init__(
        self,
        repository: AnnotationDatasetRepository,
        decisions: HumanDecisionResolver,
        ids: IdGenerator,
        clock: Clock,
        references: AnnotationReferenceValidator | None = None,
    ) -> None:
        self._repository = repository
        self._decisions = decisions
        self._ids = ids
        self._clock = clock
        self._references = references

    def create_version(
        self, command: CreateAnnotationDatasetVersion
    ) -> AnnotationDatasetVersion:
        versions = self._repository.list_versions(command.dataset_name)
        expected_parent = versions[-1].dataset_version_id if versions else None
        if command.parent_version_id != expected_parent:
            raise ConflictError(
                "parent_version_id must reference the latest annotation dataset",
                code="annotation_dataset_parent_not_latest",
                details={"expected_parent_version_id": expected_parent},
            )
        self._validate_document_splits(command.samples, versions)
        if self._references is not None:
            for sample in command.samples:
                self._references.validate_sample(sample)
        samples = tuple(self._new_sample(item) for item in command.samples)
        payload = {
            "schema_version": 1,
            "manifest_kind": "a3_annotation_dataset",
            "dataset_version_id": self._ids.new(),
            "dataset_name": command.dataset_name,
            "version_number": len(versions) + 1,
            "parent_version_id": command.parent_version_id,
            "status": DatasetStatus.DRAFT,
            "change_summary": command.change_summary,
            "required_human_cases": len(samples),
            "provenance": AnnotationDatasetProvenance(
                source_description=command.source_description,
                source_manifest_sha256=command.source_manifest_sha256,
                source_work_package_sha256=command.source_work_package_sha256,
            ).model_dump(mode="json"),
            "samples": [item.model_dump(mode="json") for item in samples],
            "created_at": self._clock.now(),
            "frozen_at": None,
        }
        version = AnnotationDatasetVersion(
            **payload,
            manifest_sha256=dataset_sha256(payload),
        )
        return self._repository.add_version(version)

    def get_version(self, dataset_version_id: str) -> AnnotationDatasetVersion:
        return self._repository.get_version(dataset_version_id)

    def list_versions(
        self,
        *,
        dataset_name: str | None = None,
        status: DatasetStatus | None = None,
        sample_status: AnnotationSampleStatus | None = None,
    ) -> tuple[AnnotationDatasetVersion, ...]:
        return self._repository.list_versions(dataset_name, status, sample_status)

    def list_samples(
        self,
        dataset_version_id: str,
        *,
        status: AnnotationSampleStatus | None = None,
    ) -> tuple[DatasetAnnotationSample, ...]:
        version = self._repository.get_version(dataset_version_id)
        return tuple(
            item for item in version.samples if status is None or item.status is status
        )

    def submit_annotation(self, command: SubmitAnnotationLabel) -> AnnotationDatasetVersion:
        version, sample = self._editable_sample(
            command, required=AnnotationSampleStatus.PENDING_ANNOTATION
        )
        record = self._record(sample, command)
        updated = self._rehash_sample(
            sample,
            status=AnnotationSampleStatus.PENDING_REVIEW,
            annotation=record,
        )
        return self._replace_sample(version, updated)

    def submit_review(self, command: SubmitAnnotationLabel) -> AnnotationDatasetVersion:
        version, sample = self._editable_sample(
            command, required=AnnotationSampleStatus.PENDING_REVIEW
        )
        record = self._record(sample, command)
        if sample.annotation is not None and record.actor_id == sample.annotation.actor_id:
            raise PermanentError(
                "reviewer must be independent from the annotator",
                code="annotation_self_review_forbidden",
            )
        matching = sample.annotation is not None and (
            record.label_sha256 == sample.annotation.label_sha256
        )
        updated = self._rehash_sample(
            sample,
            status=(
                AnnotationSampleStatus.VERIFIED
                if matching
                else AnnotationSampleStatus.CONFLICT
            ),
            review=record,
            final_label_sha256=record.label_sha256 if matching else None,
            label_version=(
                f"chunk-relevance.v1:{record.label_sha256}" if matching else None
            ),
        )
        return self._replace_sample(version, updated)

    def adjudicate(self, command: SubmitAnnotationLabel) -> AnnotationDatasetVersion:
        version, sample = self._editable_sample(
            command, required=AnnotationSampleStatus.CONFLICT
        )
        record = self._record(sample, command)
        prior_actors = {
            item.actor_id for item in (sample.annotation, sample.review) if item is not None
        }
        if record.actor_id in prior_actors:
            raise PermanentError(
                "adjudicator must be independent from annotation and review",
                code="annotation_adjudicator_not_independent",
            )
        updated = self._rehash_sample(
            sample,
            status=AnnotationSampleStatus.VERIFIED,
            adjudication=record,
            final_label_sha256=record.label_sha256,
            label_version=f"chunk-relevance.v1:{record.label_sha256}",
        )
        return self._replace_sample(version, updated)

    def freeze(self, dataset_version_id: str) -> AnnotationDatasetVersion:
        version = self._repository.get_version(dataset_version_id)
        if version.status is DatasetStatus.FROZEN:
            return version
        if version.status is not DatasetStatus.VERIFIED:
            raise PermanentError(
                "all required samples must be independently verified before freezing",
                code="annotation_dataset_not_verified",
            )
        if DatasetSplit.FROZEN_TEST not in {item.split for item in version.samples}:
            raise PermanentError(
                "a frozen dataset requires at least one frozen-test document",
                code="annotation_dataset_missing_frozen_test",
            )
        samples = tuple(
            self._rehash_sample(item, status=AnnotationSampleStatus.FROZEN)
            for item in version.samples
        )
        updated = self._rehash_version(
            version,
            status=DatasetStatus.FROZEN,
            samples=samples,
            frozen_at=self._clock.now(),
            provenance=version.provenance.model_copy(
                update={"status": "verified", "claims_allowed": bool(samples)}
            ),
        )
        return self._repository.replace_version(
            updated, expected_manifest_sha256=version.manifest_sha256
        )

    def create_revision(
        self, command: CreateAnnotationDatasetRevision
    ) -> AnnotationDatasetVersion:
        parent = self._repository.get_version(command.parent_version_id)
        if parent.status is not DatasetStatus.FROZEN:
            raise ConflictError(
                "only a frozen annotation dataset can be revised",
                code="annotation_dataset_revision_requires_frozen_parent",
            )
        versions = self._repository.list_versions(parent.dataset_name)
        if versions[-1].dataset_version_id != parent.dataset_version_id:
            raise ConflictError(
                "a revision must use the latest annotation dataset version",
                code="annotation_dataset_parent_not_latest",
            )
        sample_ids = {item.sample_id for item in parent.samples}
        unknown = set(command.reset_sample_ids) - sample_ids
        if unknown:
            raise PermanentError(
                "reset_sample_ids contains unknown samples",
                code="annotation_dataset_unknown_samples",
                details={"sample_ids": sorted(unknown)},
            )
        reset = set(command.reset_sample_ids)
        samples = tuple(
            self._reset_sample(item)
            if item.sample_id in reset
            else self._rehash_sample(
                item,
                status=AnnotationSampleStatus.VERIFIED,
            )
            for item in parent.samples
        )
        now = self._clock.now()
        payload = {
            "schema_version": 1,
            "manifest_kind": "a3_annotation_dataset",
            "dataset_version_id": self._ids.new(),
            "dataset_name": parent.dataset_name,
            "version_number": parent.version_number + 1,
            "parent_version_id": parent.dataset_version_id,
            "status": DatasetStatus.DRAFT,
            "change_summary": command.change_summary,
            "required_human_cases": parent.required_human_cases,
            "provenance": parent.provenance.model_copy(
                update={"status": "provisional", "claims_allowed": False}
            ).model_dump(mode="json"),
            "samples": [item.model_dump(mode="json") for item in samples],
            "created_at": now,
            "frozen_at": None,
        }
        version = AnnotationDatasetVersion(
            **payload,
            manifest_sha256=dataset_sha256(payload),
        )
        return self._repository.add_version(version)

    def _new_sample(self, value: AnnotationSampleInput) -> DatasetAnnotationSample:
        payload = value.model_dump(mode="json")
        payload.update(
            {
                "status": AnnotationSampleStatus.PENDING_ANNOTATION,
                "evidence_catalog_sha256": dataset_sha256(
                    [
                        item.model_dump(mode="json")
                        for item in sorted(
                            value.candidate_chunks, key=lambda item: item.chunk_id
                        )
                    ]
                ),
                "annotation": None,
                "review": None,
                "adjudication": None,
                "final_label_sha256": None,
                "label_version": None,
            }
        )
        return DatasetAnnotationSample(
            **payload,
            sample_sha256=dataset_sha256(payload),
        )

    def _reset_sample(self, sample: DatasetAnnotationSample) -> DatasetAnnotationSample:
        return self._rehash_sample(
            sample,
            status=AnnotationSampleStatus.PENDING_ANNOTATION,
            annotation=None,
            review=None,
            adjudication=None,
            final_label_sha256=None,
            label_version=None,
        )

    def _editable_sample(
        self,
        command: SubmitAnnotationLabel,
        *,
        required: AnnotationSampleStatus,
    ) -> tuple[AnnotationDatasetVersion, DatasetAnnotationSample]:
        version = self._repository.get_version(command.dataset_version_id)
        if version.status is DatasetStatus.FROZEN:
            raise ConflictError(
                "frozen annotation datasets are immutable; create a new version",
                code="annotation_dataset_frozen",
            )
        sample = next(
            (item for item in version.samples if item.sample_id == command.sample_id),
            None,
        )
        if sample is None:
            raise NotFoundError(
                "annotation sample does not exist",
                code="annotation_sample_not_found",
            )
        if sample.status is not required:
            raise ConflictError(
                "annotation action is invalid for the current sample state",
                code="annotation_state_conflict",
                details={"status": sample.status.value, "required": required.value},
            )
        return version, sample

    def _record(
        self,
        sample: DatasetAnnotationSample,
        command: SubmitAnnotationLabel,
    ) -> HumanLabelRecord:
        decision = self._decisions.get_decision(
            sample.finding_id, command.human_decision_id
        )
        if decision.reviewer_id != command.actor_id:
            raise PermanentError(
                "HumanDecision reviewer does not match the submitted actor",
                code="annotation_human_decision_actor_mismatch",
            )
        known_chunks = {item.chunk_id: item for item in sample.candidate_chunks}
        unknown = set(command.label.relevant_chunk_ids) - set(known_chunks)
        if unknown:
            raise PermanentError(
                "label references chunks outside the sample evidence catalog",
                code="annotation_chunk_reference_invalid",
                details={"chunk_ids": sorted(unknown)},
            )
        evidence = [
            known_chunks[chunk_id].model_dump(mode="json")
            for chunk_id in sorted(command.label.relevant_chunk_ids)
        ]
        return HumanLabelRecord(
            human_decision_id=decision.decision_id,
            human_decision_sha256=decision.decision_sha256,
            human_decision_evidence_sha256=decision.evidence_sha256,
            actor_id=decision.reviewer_id,
            acted_at=decision.decided_at,
            label=command.label,
            label_sha256=dataset_sha256(command.label.model_dump(mode="json")),
            relevant_evidence_sha256=dataset_sha256(evidence),
        )

    def _replace_sample(
        self,
        version: AnnotationDatasetVersion,
        updated_sample: DatasetAnnotationSample,
    ) -> AnnotationDatasetVersion:
        samples = tuple(
            updated_sample if item.sample_id == updated_sample.sample_id else item
            for item in version.samples
        )
        all_verified = all(
            item.status is AnnotationSampleStatus.VERIFIED for item in samples
        )
        updated = self._rehash_version(
            version,
            status=DatasetStatus.VERIFIED if all_verified else DatasetStatus.DRAFT,
            samples=samples,
            provenance=version.provenance.model_copy(
                update={"status": "verified" if all_verified else "provisional"}
            ),
        )
        return self._repository.replace_version(
            updated, expected_manifest_sha256=version.manifest_sha256
        )

    @staticmethod
    def _rehash_sample(
        sample: DatasetAnnotationSample,
        **changes: object,
    ) -> DatasetAnnotationSample:
        payload = sample.model_dump(mode="json", exclude={"sample_sha256"})
        for name, value in changes.items():
            payload[name] = (
                value.model_dump(mode="json")
                if isinstance(value, ContractModel)
                else value
            )
        return DatasetAnnotationSample(
            **payload,
            sample_sha256=dataset_sha256(payload),
        )

    @staticmethod
    def _rehash_version(
        version: AnnotationDatasetVersion,
        **changes: object,
    ) -> AnnotationDatasetVersion:
        payload = version.model_dump(mode="json", exclude={"manifest_sha256"})
        for name, value in changes.items():
            if isinstance(value, ContractModel):
                payload[name] = value.model_dump(mode="json")
            elif name == "samples" and isinstance(value, tuple):
                payload[name] = [item.model_dump(mode="json") for item in value]
            else:
                payload[name] = value
        return AnnotationDatasetVersion(
            **payload,
            manifest_sha256=dataset_sha256(payload),
        )

    @staticmethod
    def _validate_document_splits(
        samples: tuple[AnnotationSampleInput, ...],
        versions: tuple[AnnotationDatasetVersion, ...],
    ) -> None:
        assignments: dict[str, DatasetSplit] = {}
        for version in versions:
            for sample in version.samples:
                assignments[sample.document_sha256] = sample.split
        for sample in samples:
            existing = assignments.get(sample.document_sha256)
            if existing is not None and existing is not sample.split:
                raise PermanentError(
                    "one document hash cannot cross optimization, validation, and frozen-test",
                    code="annotation_document_leakage",
                    details={"document_sha256": sample.document_sha256},
                )
            assignments[sample.document_sha256] = sample.split
