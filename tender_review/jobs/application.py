from __future__ import annotations

import hashlib
import json

from tender_review.shared.clock import Clock
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import ConflictError
from tender_review.shared.ids import IdGenerator
from tender_review.retrieval.public import SearchResult
from tender_review.review.public import (
    DateRule,
    NumericRangeRule,
    ReviewInputProvenance,
    ReviewRequest,
    SetRule,
    TextPresenceRule,
)

from .domain import (
    advance_stage,
    can_rerun,
    cancel_job,
    complete_job,
    queue_retry,
    record_failure,
    start_job,
    wait_for_human,
)
from .models import (
    CreateReviewJobCommand,
    IdempotencyRecord,
    IdempotentReviewJob,
    JobCheckpoint,
    JobFailure,
    JobLifecycle,
    ReviewExecutionSpec,
    ReviewJob,
    ReviewStage,
    build_review_execution_spec,
    clone_review_execution_spec,
)
from .ports import ReviewJobRepository


def normalized_request_hash(command: CreateReviewJobCommand) -> str:
    """Hash the canonical JSON representation after DTO-level normalization."""

    payload = json.dumps(
        command.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execution_fingerprint(command: CreateReviewJobCommand) -> str:
    """Fingerprint inputs that determine a review execution's result."""

    inputs = (
        command.execution_spec.model_dump(mode="json")
        if command.execution_spec is not None
        else {
            "document_sha256": command.document_sha256,
            "model_config_hash": command.model_config_hash,
            "rule_version_hash": command.rule_version_hash,
        }
    )
    payload = json.dumps(
        inputs,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReviewJobService:
    """Application service for explicit review-job commands and queries."""

    def __init__(
        self,
        *,
        repository: ReviewJobRepository,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._ids = ids
        self._clock = clock

    def create(
        self,
        command: CreateReviewJobCommand,
        *,
        caller_id: str,
        scope: str,
        idempotency_key: str,
    ) -> IdempotentReviewJob:
        normalized_caller = _require_value(caller_id, "caller_id")
        normalized_scope = _require_value(scope, "scope")
        normalized_key = _require_value(idempotency_key, "idempotency_key")
        now = self._clock.now()
        job_id = self._ids.new()
        execution_spec = (
            build_review_execution_spec(job_id, command.execution_spec)
            if command.execution_spec is not None
            else None
        )
        job = ReviewJob(
            id=job_id,
            document_snapshot_id=command.document_snapshot_id,
            rule_version_id=command.rule_version_id,
            model_config_id=command.model_config_id,
            input_fingerprint=execution_fingerprint(command),
            execution_spec_sha256=(
                execution_spec.input_sha256 if execution_spec is not None else None
            ),
            max_attempts=command.max_attempts,
            available_at=now,
            created_at=now,
            updated_at=now,
        )
        record = IdempotencyRecord(
            id=self._ids.new(),
            caller_id=normalized_caller,
            scope=normalized_scope,
            idempotency_key=normalized_key,
            request_hash=normalized_request_hash(command),
            resource_id=job.id,
            created_at=now,
        )
        return self._repository.create_review_job(job, record, execution_spec)

    def get(self, job_id: str) -> ReviewJob:
        return self._repository.get_review_job(_require_value(job_id, "job_id"))

    def get_execution_spec(self, job_id: str) -> ReviewExecutionSpec:
        return self._repository.get_review_execution_spec(
            _require_value(job_id, "job_id")
        )

    def cancel(self, job_id: str) -> ReviewJob:
        job = self.get(job_id)
        if job.status is JobLifecycle.CANCELLED:
            return job
        durable_cancel = getattr(self._repository, "cancel_review_job", None)
        if callable(durable_cancel):
            return durable_cancel(job.id)
        return self._save(cancel_job(job, now=self._clock.now()))

    def rerun(self, job_id: str) -> ReviewJob:
        original = self.get(job_id)
        if not can_rerun(original):
            raise ConflictError(
                "Only terminal review jobs can be explicitly rerun",
                code="review_job_rerun_not_allowed",
                details={"status": original.status.value},
            )
        now = self._clock.now()
        original_spec = (
            self.get_execution_spec(original.id)
            if original.execution_spec_sha256 is not None
            else None
        )
        rerun_id = self._ids.new()
        rerun_spec = (
            clone_review_execution_spec(original_spec, rerun_id)
            if original_spec is not None
            else None
        )
        rerun = ReviewJob(
            id=rerun_id,
            document_snapshot_id=original.document_snapshot_id,
            rule_version_id=original.rule_version_id,
            model_config_id=original.model_config_id,
            input_fingerprint=original.input_fingerprint,
            execution_spec_sha256=(
                rerun_spec.input_sha256 if rerun_spec is not None else None
            ),
            rerun_of=original.id,
            max_attempts=original.max_attempts,
            available_at=now,
            created_at=now,
            updated_at=now,
        )
        return self._repository.create_review_job(
            rerun, execution_spec=rerun_spec
        ).job

    def start(self, job_id: str) -> ReviewJob:
        return self._save(start_job(self.get(job_id), now=self._clock.now()))

    def advance_stage(self, job_id: str, stage: ReviewStage) -> ReviewJob:
        return self._save(advance_stage(self.get(job_id), stage, now=self._clock.now()))

    def wait_for_human(self, job_id: str) -> ReviewJob:
        return self._save(wait_for_human(self.get(job_id), now=self._clock.now()))

    def complete(self, job_id: str) -> ReviewJob:
        return self._save(complete_job(self.get(job_id), now=self._clock.now()))

    def retry(self, job_id: str) -> ReviewJob:
        return self._save(queue_retry(self.get(job_id), now=self._clock.now()))

    def fail(self, job_id: str, failure: JobFailure) -> ReviewJob:
        return self._save(record_failure(self.get(job_id), failure, now=self._clock.now()))

    def save_checkpoint(self, checkpoint: JobCheckpoint) -> JobCheckpoint:
        return self._repository.save_checkpoint(checkpoint)

    def list_checkpoints(self, job_id: str) -> tuple[JobCheckpoint, ...]:
        return self._repository.list_checkpoints(_require_value(job_id, "job_id"))

    def _save(self, job: ReviewJob) -> ReviewJob:
        return self._repository.save_review_job(job)


def _require_value(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ConflictError(f"{field_name} must not be blank", code=f"{field_name}_blank")
    return normalized


ReviewRule = DateRule | SetRule | NumericRangeRule | TextPresenceRule


class ReviewExecutionSpecParser:
    """Handler-facing bridge from a verified spec to the existing ReviewRequest."""

    def parse(
        self,
        spec: ReviewExecutionSpec,
        *,
        rule: ReviewRule,
        resolved_rule_version_id: str,
        resolved_rule_version_hash: str,
        resolved_dataset_version_id: str,
        resolved_dataset_version_hash: str,
        provenance_status: str,
        claims_allowed: bool,
        retrieval_results_sha256: str,
        call: CallContext,
        retrieval_result: SearchResult | None = None,
    ) -> ReviewRequest:
        if (
            resolved_rule_version_id != spec.rule_version_id
            or resolved_rule_version_hash != spec.rule_version_hash
        ):
            raise ConflictError(
                "Resolved rule version conflicts with the review execution spec",
                code="review_execution_rule_conflict",
            )
        if (
            resolved_dataset_version_id != spec.dataset_version_id
            or resolved_dataset_version_hash != spec.dataset_version_hash
        ):
            raise ConflictError(
                "Resolved dataset version conflicts with the review execution spec",
                code="review_execution_dataset_conflict",
            )
        if provenance_status not in {"provisional", "verified"}:
            raise ConflictError(
                "Dataset provenance status is not supported",
                code="review_execution_provenance_invalid",
            )
        source_kind = (
            "verified_retrieval"
            if provenance_status == "verified"
            else "provisional_retrieval"
        )
        provenance = ReviewInputProvenance(
            source_kind=source_kind,
            status=provenance_status,
            claims_allowed=claims_allowed,
            dataset_version_id=spec.dataset_version_id,
            input_sha256=spec.input_sha256,
            results_sha256=retrieval_results_sha256,
            variant=spec.retrieval_variant,
        )
        return ReviewRequest(
            review_job_id=spec.job_id,
            query=spec.query,
            document_ids=(spec.document_snapshot_id,),
            rule=rule,
            provenance=provenance,
            call=call,
            retrieval_result=retrieval_result,
        )
