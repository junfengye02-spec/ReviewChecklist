from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from pydantic import TypeAdapter, ValidationError

from tender_review.documents import ArtifactStore
from tender_review.documents.lifecycle import (
    ArtifactRecord,
    ArtifactType,
    DocumentLifecycleRepository,
)
from tender_review.evaluation.public import DatasetVersionRepository
from tender_review.findings.public import DocumentIdentity, FindingRepository
from tender_review.jobs.public import (
    CheckpointValue,
    JobFailure,
    JobHandlerOutcome,
    JobHandlerStatus,
    JobMessage,
    JobResult,
    ReviewExecutionSpec,
    ReviewExecutionSpecParser,
    ReviewJobRepository,
    ReviewStage,
)
from tender_review.retrieval.public import (
    ArtifactBackedHybridRetriever,
    ArtifactSearchResult,
    EmbeddingProvider,
    RetrievalIndexLoadError,
    RetrievalIndexLoader,
    RetrievalProvenance,
    SearchRequest,
    SearchResult,
)
from tender_review.review.public import (
    LangGraphReviewWorkflow,
    EXTRACT_STRUCTURED_FIELDS,
    PERSIST_FINDING,
    RETRIEVE_EVIDENCE,
    ReviewGraphNode,
    ReviewGraphState,
    ReviewLifecycle,
    ReviewProcessingStage,
    ReviewRule,
    approval_finding_from_review_state,
)
from tender_review.rule_management.public import RuleVersionRepository
from tender_review.shared.clock import Clock
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import (
    ErrorCategory,
    PermanentError,
    ServiceError,
)
from tender_review.shared.faults import DisabledFaultInjector, FaultInjector
from tender_review.shared.observability import (
    CorrelationContext,
    log_event,
    record_metric,
)
from tender_review.stage8.public import (
    ActorKind,
    AuditResult,
    AuditService,
    ReportSourceType,
)

from .runner import WorkerExecutionContext


_RULE_ADAPTER = TypeAdapter(ReviewRule)

_STAGE_MAP = {
    ReviewProcessingStage.RETRIEVING: ReviewStage.RETRIEVING,
    ReviewProcessingStage.VERIFYING_EVIDENCE: ReviewStage.VERIFYING,
    ReviewProcessingStage.EXTRACTING: ReviewStage.EXTRACTING,
    ReviewProcessingStage.COMPARING: ReviewStage.COMPARING,
    ReviewProcessingStage.REPORTING: ReviewStage.REPORTING,
}

FAULT_AFTER_RETRIEVAL_CHECKPOINT = "review.after_retrieval_checkpoint"
FAULT_AFTER_EXTRACTION_CHECKPOINT = "review.after_extraction_checkpoint"
FAULT_AFTER_REPORT_CHECKPOINT = "review.after_report_checkpoint"
_FAULT_NODES = {
    FAULT_AFTER_RETRIEVAL_CHECKPOINT: RETRIEVE_EVIDENCE,
    FAULT_AFTER_EXTRACTION_CHECKPOINT: EXTRACT_STRUCTURED_FIELDS,
    FAULT_AFTER_REPORT_CHECKPOINT: PERSIST_FINDING,
}


def retrieval_results_sha256(result: SearchResult) -> str:
    """Hash deterministic result fields without runtime latency measurements."""

    payload = result.model_dump(mode="json")
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("latency_ms", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ApprovalFindingPersister:
    """Idempotently project terminal review states into approval storage."""

    def __init__(
        self,
        *,
        jobs: ReviewJobRepository,
        documents: DocumentLifecycleRepository,
        findings: FindingRepository,
    ) -> None:
        self._jobs = jobs
        self._documents = documents
        self._findings = findings

    def __call__(self, state: ReviewGraphState) -> str:
        job = self._jobs.get_review_job(state.review_job_id)
        spec = self._jobs.get_review_execution_spec(state.review_job_id)
        snapshot = self._documents.get_snapshot(spec.document_snapshot_id)
        finding = approval_finding_from_review_state(
            state,
            rule_version_id=spec.rule_version_id,
            documents=(
                DocumentIdentity(
                    document_id=snapshot.id,
                    document_sha256=snapshot.object.sha256,
                ),
            ),
            created_at=job.created_at,
        )
        return self._findings.add_finding(finding).finding_id


class ReviewJobHandler:
    """Production orchestration from a signed execution spec to one graph thread."""

    job_type = "review"

    def __init__(
        self,
        *,
        jobs: ReviewJobRepository,
        documents: DocumentLifecycleRepository,
        rules: RuleVersionRepository,
        datasets: DatasetVersionRepository,
        artifact_store: ArtifactStore,
        embedding_provider: EmbeddingProvider,
        index_loader: RetrievalIndexLoader,
        workflow: LangGraphReviewWorkflow,
        findings: ApprovalFindingPersister,
        clock: Clock,
        model_config_id: str,
        model_config_hash: str,
        call_timeout_seconds: float,
        call_max_attempts: int,
        parser: ReviewExecutionSpecParser | None = None,
        fault_injector: FaultInjector | None = None,
        audit: AuditService | None = None,
    ) -> None:
        if not model_config_id.strip():
            raise ValueError("model_config_id must not be blank")
        if len(model_config_hash) != 64:
            raise ValueError("model_config_hash must be a SHA-256 digest")
        self._jobs = jobs
        self._documents = documents
        self._rules = rules
        self._datasets = datasets
        self._artifact_store = artifact_store
        self._embedding_provider = embedding_provider
        self._index_loader = index_loader
        self._workflow = workflow
        self._findings = findings
        self._clock = clock
        self._model_config_id = model_config_id
        self._model_config_hash = model_config_hash
        self._call_timeout_seconds = call_timeout_seconds
        self._call_max_attempts = call_max_attempts
        self._parser = parser or ReviewExecutionSpecParser()
        self._faults = fault_injector or DisabledFaultInjector()
        self._audit = audit
        self._logger = logging.getLogger("tender_review.worker.review")

    def __call__(
        self, job: JobMessage, context: WorkerExecutionContext
    ) -> JobHandlerOutcome:
        try:
            return self._execute(job, context)
        except ServiceError as exc:
            log_event(
                self._logger,
                logging.WARNING,
                event="review.handler_failed",
                message="Review job handler returned a typed failure",
                context=CorrelationContext(
                    job_id=job.job_id,
                    thread_id=job.job_id,
                    call_id=f"review:{job.job_id}",
                ),
                error_code=exc.code,
                error_category=exc.category.value,
                retryable=exc.retryable,
            )
            status = (
                JobHandlerStatus.CANCELLED
                if exc.category is ErrorCategory.CANCELLED
                else JobHandlerStatus.FAILED
            )
            return JobHandlerOutcome(
                status=status,
                failure=JobFailure(
                    code=exc.code,
                    message=exc.message,
                    category=exc.category,
                    retryable=exc.retryable,
                    stage=self._failure_stage(context),
                ),
            )
        except (TypeError, ValueError) as exc:
            del exc
            return JobHandlerOutcome(
                status=JobHandlerStatus.FAILED,
                failure=JobFailure(
                    code="review_execution_input_invalid",
                    message="Review execution input failed validation",
                    category=ErrorCategory.PERMANENT,
                    retryable=False,
                    stage=self._failure_stage(context),
                ),
            )

    def _execute(
        self, message: JobMessage, context: WorkerExecutionContext
    ) -> JobHandlerOutcome:
        if message.job_type != self.job_type:
            raise PermanentError(
                "ReviewJobHandler received an unsupported job type",
                code="review_job_type_invalid",
            )
        job = self._jobs.get_review_job(message.job_id)
        if job.execution_spec_sha256 is None:
            raise PermanentError(
                "Review jobs created without an execution spec cannot be executed",
                code="review_execution_spec_required",
            )
        spec = self._load_verified_spec(message.job_id)
        call = CallContext(
            call_id=f"review:{spec.job_id}",
            timeout_seconds=self._call_timeout_seconds,
            max_attempts=self._call_max_attempts,
        )
        correlation = self._correlation(spec, call.call_id)
        log_event(
            self._logger,
            logging.INFO,
            event="review.handler_started",
            message="Review job handler started",
            context=correlation,
            attempt=message.attempt,
            recovery_count=max(0, message.attempt - 1),
        )
        snapshot = self._documents.get_snapshot(spec.document_snapshot_id)
        self._require_equal(
            snapshot.object.sha256,
            spec.document_sha256,
            code="review_execution_document_conflict",
            message="Document snapshot hash conflicts with the execution spec",
        )
        rule_version = self._rules.get_version(spec.rule_version_id)
        self._require_equal(
            rule_version.content_sha256,
            spec.rule_version_hash,
            code="review_execution_rule_conflict",
            message="Rule version hash conflicts with the execution spec",
        )
        rule = self._parse_rule(rule_version.content_json)
        dataset = self._datasets.get_version(spec.dataset_version_id)
        self._require_equal(
            dataset.manifest_sha256,
            spec.dataset_version_hash,
            code="review_execution_dataset_conflict",
            message="Dataset version hash conflicts with the execution spec",
        )
        self._validate_model_identity(spec)
        artifacts = self._load_artifacts(spec)
        retriever = self._build_retriever(spec, artifacts, call)
        state_reader = getattr(self._workflow, "latest_state", None)
        durable_state = state_reader(spec.job_id) if callable(state_reader) else None
        if durable_state is not None and durable_state.retrieval_result is not None:
            result = self._restore_retrieval_result(
                retriever, durable_state.retrieval_result
            )
            retrieval_duration_ms = None
            retrieval_metric_status = "not_collected"
            retrieval_metric_source = "langgraph_checkpoint"
            log_event(
                self._logger,
                logging.INFO,
                event="review.retrieval_recovered",
                message="Review retrieval result restored from a durable checkpoint",
                context=correlation,
            )
        else:
            retrieval_started = time.perf_counter()
            result = retriever.search(
                SearchRequest(
                    query=spec.query,
                    document_ids=(spec.document_snapshot_id,),
                    limit=retriever.index.manifest.top_k,
                    call=call,
                )
            )
            retrieval_duration_ms = max(
                0.0, (time.perf_counter() - retrieval_started) * 1000.0
            )
            retrieval_metric_status = "collected"
            retrieval_metric_source = "process_monotonic"
        if not isinstance(result, ArtifactSearchResult):
            raise PermanentError(
                "Artifact-backed retrieval returned an unverified result type",
                code="review_retrieval_result_invalid",
            )
        record_metric(
            self._logger,
            name="review_retrieval_duration",
            value=retrieval_duration_ms,
            unit="ms",
            source=retrieval_metric_source,
            status=retrieval_metric_status,
            context=correlation,
        )
        request = self._parser.parse(
            spec,
            rule=rule,
            resolved_rule_version_id=rule_version.rule_version_id,
            resolved_rule_version_hash=rule_version.content_sha256,
            resolved_dataset_version_id=dataset.dataset_version_id,
            resolved_dataset_version_hash=dataset.manifest_sha256,
            provenance_status=result.provenance.status,
            claims_allowed=result.provenance.claims_allowed,
            retrieval_results_sha256=retrieval_results_sha256(result),
            call=call,
            retrieval_result=result,
        )
        fault_point = next(
            (point for point in _FAULT_NODES if self._faults.is_armed(point)),
            None,
        )
        interrupt_after = (_FAULT_NODES[fault_point],) if fault_point else ()
        state = self._workflow.run(
            request,
            interrupt_after=interrupt_after,
            correlation=correlation,
        )
        pointer = self._workflow.latest_checkpoint(spec.job_id)
        if pointer is None:
            raise PermanentError(
                "LangGraph completed without a durable checkpoint",
                code="review_graph_checkpoint_missing",
            )
        trace_reader = getattr(self._workflow, "checkpoint_trace", None)
        pointers = tuple(trace_reader(spec.job_id)) if callable(trace_reader) else ()
        if not pointers:
            pointers = (pointer,)
        node_records_reader = getattr(self._workflow, "node_records", None)
        node_records = (
            tuple(node_records_reader(spec.job_id))
            if callable(node_records_reader)
            else ()
        )
        for trace_pointer in pointers[:-1]:
            self._save_pointer(context, spec, trace_pointer)
        self._save_observability(
            context,
            spec,
            state,
            pointer,
            node_records=node_records,
            recovery_count=max(0, message.attempt - 1),
        )
        self._save_pointer(context, spec, pointers[-1])
        correlation = correlation.with_checkpoint(pointer.checkpoint_id)
        log_event(
            self._logger,
            logging.INFO,
            event="review.checkpoint_synced",
            message="Review graph checkpoint synced to the leased job trace",
            context=correlation,
            graph_node=pointer.node.value,
            lifecycle=pointer.lifecycle.value,
        )
        if fault_point is not None:
            self._record_audit(
                spec,
                state,
                pointer.checkpoint_id,
                action="review.graph.interrupted",
                result=AuditResult.FAILED,
            )
            log_event(
                self._logger,
                logging.WARNING,
                event="review.fault_injected",
                message="Configured reliability drill interrupted the review graph",
                context=correlation,
                fault_point=fault_point,
            )
            self._faults.trip(fault_point)
        finding_id = None
        if state.node in {
            ReviewGraphNode.DONE,
            ReviewGraphNode.NEED_MORE_EVIDENCE,
            ReviewGraphNode.HUMAN_HANDOFF,
        }:
            finding_id = self._findings(state)
        self._emit_call_metrics(state, correlation)
        audit_action = {
            ReviewLifecycle.COMPLETED: "review.graph.completed",
            ReviewLifecycle.NEED_MORE_EVIDENCE: "review.graph.needs_more_evidence",
            ReviewLifecycle.WAITING_HUMAN: "review.graph.waiting_human",
            ReviewLifecycle.FAILED: "review.graph.failed",
        }[state.lifecycle]
        self._record_audit(
            spec,
            state,
            pointer.checkpoint_id,
            action=audit_action,
            result=(
                AuditResult.FAILED
                if state.lifecycle is ReviewLifecycle.FAILED
                else AuditResult.SUCCEEDED
            ),
        )
        log_event(
            self._logger,
            logging.INFO,
            event="review.handler_completed",
            message="Review job handler reached a durable graph outcome",
            context=correlation,
            graph_node=state.node.value,
            lifecycle=state.lifecycle.value,
            finding_id=finding_id,
        )
        return self._outcome(state, finding_id)

    def _load_verified_spec(self, job_id: str) -> ReviewExecutionSpec:
        spec = self._jobs.get_review_execution_spec(job_id)
        try:
            verified = ReviewExecutionSpec.model_validate(spec.model_dump(mode="json"))
        except ValidationError as exc:
            raise PermanentError(
                "Stored review execution spec failed integrity validation",
                code="review_execution_spec_tampered",
            ) from exc
        self._jobs.verify_review_execution_spec(verified)
        return verified

    def _validate_model_identity(self, spec: ReviewExecutionSpec) -> None:
        if (
            spec.model_config_id != self._model_config_id
            or spec.model_config_hash != self._model_config_hash
        ):
            raise PermanentError(
                "Production model configuration conflicts with the execution spec",
                code="review_execution_model_config_conflict",
            )

    def _load_artifacts(
        self, spec: ReviewExecutionSpec
    ) -> dict[str, ArtifactRecord]:
        artifacts = {
            "retriever": self._documents.get_artifact(
                spec.retriever_artifact.artifact_id
            ),
            "index": self._documents.get_artifact(spec.index_artifact.artifact_id),
            "chunk": self._documents.get_artifact(spec.chunk_artifact.artifact_id),
        }
        expected = {
            "retriever": (spec.retriever_artifact, ArtifactType.INDEX),
            "index": (spec.index_artifact, ArtifactType.INDEX),
            "chunk": (spec.chunk_artifact, ArtifactType.PARSED_JSON),
        }
        for role, artifact in artifacts.items():
            reference, artifact_type = expected[role]
            actual = (
                artifact.id,
                artifact.document_snapshot_id,
                artifact.artifact_type,
                artifact.object.bucket,
                artifact.object.object_key,
                artifact.object.sha256,
            )
            wanted = (
                reference.artifact_id,
                spec.document_snapshot_id,
                artifact_type,
                reference.bucket,
                reference.object_key,
                reference.sha256,
            )
            if actual != wanted:
                raise PermanentError(
                    f"{role} artifact conflicts with the execution spec",
                    code="review_execution_artifact_conflict",
                    details={"artifact_role": role},
                )
        return artifacts

    def _build_retriever(
        self,
        spec: ReviewExecutionSpec,
        artifacts: dict[str, ArtifactRecord],
        call: CallContext,
    ) -> ArtifactBackedHybridRetriever:
        try:
            retriever = ArtifactBackedHybridRetriever.from_manifest(
                artifact_store=self._artifact_store,
                manifest_key=artifacts["retriever"].object.object_key,
                embedding_provider=self._embedding_provider,
                call=call,
                loader=self._index_loader,
            )
        except RetrievalIndexLoadError as exc:
            raise PermanentError(
                "Retrieval artifacts failed integrity validation",
                code="review_retrieval_artifact_invalid",
            ) from exc
        manifest = retriever.index.manifest
        actual = (
            retriever.index.manifest_key,
            retriever.index.manifest_sha256,
            manifest.vector_index.key,
            manifest.vector_index.sha256,
            manifest.chunk_catalog.key,
            manifest.chunk_catalog.sha256,
        )
        expected = (
            spec.retriever_artifact.object_key,
            spec.retriever_artifact.sha256,
            spec.index_artifact.object_key,
            spec.index_artifact.sha256,
            spec.chunk_artifact.object_key,
            spec.chunk_artifact.sha256,
        )
        if actual != expected:
            raise PermanentError(
                "Retriever manifest references conflict with the execution spec",
                code="review_retrieval_manifest_conflict",
            )
        return retriever

    @staticmethod
    def _restore_retrieval_result(
        retriever: ArtifactBackedHybridRetriever,
        result: SearchResult,
    ) -> ArtifactSearchResult:
        manifest = retriever.index.manifest
        return ArtifactSearchResult(
            retriever=result.retriever,
            hits=result.hits,
            provenance=RetrievalProvenance(
                retriever_version=manifest.retriever_version,
                embedding_model=manifest.embedding_model,
                embedding_dimensions=manifest.embedding_dimensions,
                chunk_config=manifest.chunk_config,
                top_k=manifest.top_k,
                candidate_limit=manifest.candidate_limit,
                manifest_sha256=retriever.index.manifest_sha256,
                chunk_catalog_sha256=manifest.chunk_catalog.sha256,
                index_sha256=manifest.vector_index.sha256,
                latency_ms=0.0,
            ),
        )

    @staticmethod
    def _parse_rule(content_json: str) -> ReviewRule:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            payload = json.loads(content_json, object_pairs_hook=reject_duplicates)
            return _RULE_ADAPTER.validate_python(payload)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            raise PermanentError(
                "RuleVersion.content_json is not a supported ReviewRule",
                code="review_execution_rule_invalid",
            ) from exc

    def _save_pointer(
        self,
        context: WorkerExecutionContext,
        spec: ReviewExecutionSpec,
        pointer: Any,
    ) -> None:
        stage = _STAGE_MAP.get(pointer.stage, ReviewStage.RETRIEVING)
        values = (
            CheckpointValue(key="langgraph_thread_id", value=spec.job_id),
            CheckpointValue(
                key="langgraph_checkpoint_id", value=pointer.checkpoint_id
            ),
            CheckpointValue(key="lifecycle", value=pointer.lifecycle.value),
            CheckpointValue(
                key="retriever_artifact_id",
                value=spec.retriever_artifact.artifact_id,
            ),
            CheckpointValue(
                key="index_artifact_id", value=spec.index_artifact.artifact_id
            ),
            CheckpointValue(
                key="chunk_artifact_id", value=spec.chunk_artifact.artifact_id
            ),
        )
        context.save_checkpoint(
            node_name=f"langgraph:{pointer.node.value.lower()}",
            stage=stage.value,
            state_json={"values": [value.model_dump(mode="json") for value in values]},
            output_artifact_id=spec.retriever_artifact.artifact_id,
        )

    def _save_observability(
        self,
        context: WorkerExecutionContext,
        spec: ReviewExecutionSpec,
        state: ReviewGraphState,
        pointer: Any,
        *,
        node_records: tuple[dict[str, object], ...],
        recovery_count: int,
    ) -> None:
        retry_count = sum(record.attempt > 1 for record in state.call_records)
        values = [
            CheckpointValue(key="langgraph_thread_id", value=spec.job_id),
            CheckpointValue(
                key="langgraph_checkpoint_id", value=pointer.checkpoint_id
            ),
            CheckpointValue(key="call_id", value=f"review:{spec.job_id}"),
            CheckpointValue(key="rule_version", value=spec.rule_version_id),
            CheckpointValue(key="dataset_version", value=spec.dataset_version_id),
            CheckpointValue(key="model_config", value=spec.model_config_id),
            CheckpointValue(key="recovery_count", value=str(recovery_count)),
            CheckpointValue(key="retry_count", value=str(retry_count)),
            CheckpointValue(
                key="model_token_status",
                value="not_collected:provider_metrics_are_log_only",
            ),
            CheckpointValue(
                key="model_cost_status",
                value="not_collected:pricing_not_configured",
            ),
            CheckpointValue(
                key="metrics_source",
                value="langgraph_internal_channel+provider_adapter_logs",
            ),
        ]
        for index, record in enumerate(node_records, start=1):
            node_name = str(record.get("node_name", "unknown"))
            duration_ms = float(record.get("duration_ms", 0.0))
            values.append(
                CheckpointValue(
                    key=f"node_duration_ms:{index}:{node_name}",
                    value=f"{duration_ms:.3f}",
                )
            )
        context.save_checkpoint(
            node_name="observability",
            stage=_STAGE_MAP.get(pointer.stage, ReviewStage.RETRIEVING).value,
            state_json={"values": [value.model_dump(mode="json") for value in values]},
            output_artifact_id=None,
        )

    @staticmethod
    def _correlation(spec: ReviewExecutionSpec, call_id: str) -> CorrelationContext:
        return CorrelationContext(
            job_id=spec.job_id,
            thread_id=spec.job_id,
            call_id=call_id,
            rule_version=spec.rule_version_id,
            dataset_version=spec.dataset_version_id,
            model_config=spec.model_config_id,
        )

    @staticmethod
    def _failure_stage(context: WorkerExecutionContext) -> ReviewStage:
        loader = getattr(context, "latest_checkpoint", None)
        checkpoint = loader() if callable(loader) else None
        stage = getattr(checkpoint, "stage", None)
        try:
            return ReviewStage(getattr(stage, "value", stage))
        except (TypeError, ValueError):
            return ReviewStage.RETRIEVING

    def _emit_call_metrics(
        self, state: ReviewGraphState, context: CorrelationContext
    ) -> None:
        for record in state.call_records:
            call_context = CorrelationContext(
                **{
                    **context.fields(),
                    "call_id": record.call_id,
                }
            )
            log_event(
                self._logger,
                logging.INFO,
                event="review.model_or_tool_call",
                message="Review external call completed",
                context=call_context,
                operation=record.operation,
                attempt=record.attempt,
                outcome=record.outcome,
                retryable=record.retryable,
                error_code=record.error_code,
            )
            record_metric(
                self._logger,
                name="external_call_duration",
                value=record.duration_ms,
                unit="ms",
                source="process_monotonic",
                context=call_context,
                operation=record.operation,
                attempt=record.attempt,
                outcome=record.outcome,
            )

    def _record_audit(
        self,
        spec: ReviewExecutionSpec,
        state: ReviewGraphState,
        checkpoint_id: str,
        *,
        action: str,
        result: AuditResult,
    ) -> None:
        if self._audit is None:
            return
        try:
            self._audit.record(
                actor_kind=ActorKind.AI,
                actor_id="review-workflow",
                action=action,
                resource_type="review_job",
                resource_id=spec.job_id,
                source_type=(
                    ReportSourceType.REAL
                    if state.provenance.status == "verified"
                    else ReportSourceType.PROVISIONAL
                ),
                provenance_status=state.provenance.status,
                claims_allowed=state.provenance.claims_allowed,
                call_id=f"review:{spec.job_id}",
                request_id=spec.job_id,
                result=result,
                artifact_sha256s=(
                    spec.input_sha256,
                    state.provenance.results_sha256,
                ),
                job_id=spec.job_id,
                thread_id=spec.job_id,
                checkpoint_id=checkpoint_id,
                rule_version=spec.rule_version_id,
                dataset_version=spec.dataset_version_id,
                model_config=spec.model_config_id,
            )
        except Exception as exc:
            log_event(
                self._logger,
                logging.WARNING,
                event="review.audit_failed",
                message="Review audit event could not be recorded",
                context=self._correlation(
                    spec, f"review:{spec.job_id}"
                ).with_checkpoint(checkpoint_id),
                error_type=type(exc).__name__,
            )

    @staticmethod
    def _outcome(
        state: ReviewGraphState, finding_id: str | None
    ) -> JobHandlerOutcome:
        reference = f"finding:{finding_id}" if finding_id is not None else None
        if state.lifecycle is ReviewLifecycle.COMPLETED:
            return JobHandlerOutcome(
                status=JobHandlerStatus.COMPLETED,
                result=JobResult(
                    output_reference=reference,
                    summary="review completed",
                ),
            )
        if state.lifecycle in {
            ReviewLifecycle.NEED_MORE_EVIDENCE,
            ReviewLifecycle.WAITING_HUMAN,
        }:
            return JobHandlerOutcome(
                status=JobHandlerStatus.WAITING_HUMAN,
                result=JobResult(
                    output_reference=reference,
                    summary=state.reason or state.lifecycle.value,
                ),
            )
        failure = state.failure
        if failure is None:
            raise PermanentError(
                "Failed review graph state has no typed failure",
                code="review_graph_failure_missing",
            )
        status = (
            JobHandlerStatus.CANCELLED
            if failure.category is ErrorCategory.CANCELLED
            else JobHandlerStatus.FAILED
        )
        return JobHandlerOutcome(
            status=status,
            failure=JobFailure(
                code=failure.code,
                message=failure.message,
                category=failure.category,
                retryable=failure.retryable,
                stage=_STAGE_MAP.get(state.stage, ReviewStage.RETRIEVING),
            ),
        )

    @staticmethod
    def _require_equal(
        actual: str, expected: str, *, code: str, message: str
    ) -> None:
        if actual != expected:
            raise PermanentError(message, code=code)
