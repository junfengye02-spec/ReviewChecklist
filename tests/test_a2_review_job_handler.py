from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from sqlalchemy import create_engine

from tender_review.documents import InMemoryArtifactStore
from tender_review.documents.models import ArtifactWrite
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.document_lifecycle import (
    SqlAlchemyDocumentLifecycleRepository,
)
from tender_review.infrastructure.database.finding_records import (
    SqlAlchemyFindingRepository,
)
from tender_review.infrastructure.database.langgraph_checkpoints import (
    LANGGRAPH_CHECKPOINT_METADATA,
    SqlAlchemyCheckpointSaver,
)
from tender_review.infrastructure.database.models import (
    DatasetVersion,
    DocumentArtifact,
    DocumentSnapshot,
    ModelConfig,
    RuleSet,
    RuleVersion,
)
from tender_review.jobs.adapters import MySqlJobRepository
from tender_review.jobs.public import (
    JobHandlerOutcome,
    JobHandlerStatus,
    JobLifecycle,
    JobMessage,
    JobResult,
    ReviewExecutionSpecDraft,
    ReviewJob,
    build_review_execution_spec,
)
from tender_review.retrieval import (
    ArtifactSearchResult,
    FakeEmbeddingProvider,
    RetrievalChunkConfig,
    RetrievalIndexLoader,
    RetrievalProvenance,
    SearchHit,
)
from tender_review.review.public import (
    FakeLlmProvider,
    LangGraphReviewWorkflow,
    SingleReviewWorkflow,
)
from tender_review.rule_management.public import canonical_json
from tender_review.shared.clock import FixedClock
from tender_review.shared.errors import ConflictError
from tender_review.shared.faults import OneShotFaultInjector
from tender_review.shared.ids import SequentialIdGenerator
from tender_review.stage8.public import AuditService, InMemoryAuditEventSink
from tender_review.worker import (
    ApprovalFindingPersister,
    ReviewJobHandler,
    Worker,
    retrieval_results_sha256,
)
from tender_review.worker.review_handler import (
    FAULT_AFTER_EXTRACTION_CHECKPOINT,
    FAULT_AFTER_REPORT_CHECKPOINT,
    FAULT_AFTER_RETRIEVAL_CHECKPOINT,
)


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
MODEL_ID = "model-1"
MODEL_HASH = "4" * 64
DATASET_ID = "dataset-1"
DATASET_HASH = "3" * 64
DOCUMENT_ID = "document-1"
DOCUMENT_HASH = "1" * 64
RULE_ID = "rule-1"
BUCKET = "review-artifacts"
RULE = {
    "schema_version": 1,
    "review_item_id": "authorization",
    "field_name": "authorization_text",
    "tool_name": "text_presence",
    "required_terms": ["signed", "authorization"],
    "mode": "all",
    "case_sensitive": False,
}
RULE_HASH = hashlib.sha256(
    canonical_json({"content": RULE, "execution_config": {}}).encode("utf-8")
).hexdigest()


def _encoded(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _put(store: InMemoryArtifactStore, value: dict[str, object]):
    return store.put(
        ArtifactWrite(
            key="content-addressed",
            content=_encoded(value),
            media_type="application/json",
        )
    )


class _RuleRepository:
    def __init__(self, *, content_json: str = canonical_json(RULE)) -> None:
        self.content_json = content_json

    def get_version(self, rule_version_id: str):
        return SimpleNamespace(
            rule_version_id=rule_version_id,
            content_sha256=RULE_HASH,
            content_json=self.content_json,
        )


class _DatasetRepository:
    def get_version(self, dataset_version_id: str):
        return SimpleNamespace(
            dataset_version_id=dataset_version_id,
            manifest_sha256=DATASET_HASH,
        )


class _ExecutionContext:
    def __init__(self) -> None:
        self.heartbeats = 0
        self.checkpoints: list[dict[str, object]] = []

    def heartbeat(self):
        self.heartbeats += 1

    def save_checkpoint(self, **values):
        self.checkpoints.append(values)


class HandlerFixture:
    def __init__(
        self,
        *,
        chunks_document_id: str = DOCUMENT_ID,
        fault_injector=None,
        audit=None,
    ) -> None:
        self.directory = TemporaryDirectory()
        path = Path(self.directory.name) / "a2-handler.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        LANGGRAPH_CHECKPOINT_METADATA.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.store = InMemoryArtifactStore()
        chunk_text = "signed authorization"
        chunk_config = {
            "strategy_name": "structural",
            "strategy_version": "1",
            "config_sha256": "9" * 64,
        }
        catalog = _put(
            self.store,
            {
                "schema_version": 1,
                "artifact_type": "retrieval_chunk_catalog",
                "format_version": "retrieval-chunk-catalog-v1",
                "chunk_config": chunk_config,
                "chunk_count": 1,
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": chunks_document_id,
                        "page_start": 7,
                        "page_end": 7,
                        "section_path": ["Authorization"],
                        "text": chunk_text,
                        "text_sha256": hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            },
        )
        vector = _put(
            self.store,
            {
                "schema_version": 1,
                "artifact_type": "retrieval_vector_index",
                "format_version": "retrieval-vector-index-v1",
                "chunk_catalog_sha256": catalog.sha256,
                "chunk_config": chunk_config,
                "embedding_model": "embedding-v1",
                "dimensions": 2,
                "vector_count": 1,
                "vectors": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": chunks_document_id,
                        "values": [1.0, 0.0],
                    }
                ],
            },
        )
        manifest = _put(
            self.store,
            {
                "schema_version": 1,
                "artifact_type": "hybrid_retrieval_index",
                "format_version": "retrieval-index-manifest-v1",
                "retriever_version": "artifact-backed-hybrid-v1",
                "chunk_catalog": {
                    "key": catalog.key,
                    "sha256": catalog.sha256,
                    "size_bytes": catalog.size_bytes,
                    "media_type": "application/json",
                },
                "vector_index": {
                    "key": vector.key,
                    "sha256": vector.sha256,
                    "size_bytes": vector.size_bytes,
                    "media_type": "application/json",
                },
                "chunk_config": chunk_config,
                "embedding_model": "embedding-v1",
                "embedding_dimensions": 2,
                "top_k": 1,
                "candidate_limit": 1,
                "bm25": {"kind": "bm25", "k1": 1.2, "b": 0.75},
                "rrf": {"k": 60},
                "status": "provisional",
                "claims_allowed": False,
            },
        )
        self.references = {
            "retriever": ("manifest-artifact", manifest, "index"),
            "index": ("index-artifact", vector, "index"),
            "chunk": ("chunk-artifact", catalog, "parsed_json"),
        }
        with self.sessions.begin() as session:
            session.add(
                DocumentSnapshot(
                    id=DOCUMENT_ID,
                    sha256=DOCUMENT_HASH,
                    object_key=f"sha256/{DOCUMENT_HASH[:2]}/{DOCUMENT_HASH}",
                    source_system="test",
                    source_document_id="source-1",
                    file_name="tender.pdf",
                    size_bytes=1,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            session.add(
                ModelConfig(
                    id=MODEL_ID,
                    provider="openai-compatible",
                    model_name="review-v1",
                    prompt_version="v1",
                    config_hash=MODEL_HASH,
                    parameters_json={},
                )
            )
            session.add(RuleSet(id="rule-set-1", rule_key="rule", name="Rule"))
            session.add(
                RuleVersion(
                    id=RULE_ID,
                    rule_set_id="rule-set-1",
                    version_number=1,
                    status="DRAFT",
                    content_hash=RULE_HASH,
                    content_json=RULE,
                    execution_config_json={},
                    change_summary="initial",
                    provenance_json={
                        "schema_version": 1,
                        "source_type": "manual",
                        "status": "verified",
                        "claims_allowed": True,
                    },
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            session.add(
                DatasetVersion(
                    id=DATASET_ID,
                    dataset_name="dataset",
                    version_number=1,
                    manifest_hash=DATASET_HASH,
                    source_type="SYNTHETIC",
                    status="PROVISIONAL",
                    split_strategy_json={},
                    manifest_json={},
                )
            )
            for artifact_id, reference, artifact_type in self.references.values():
                session.add(
                    DocumentArtifact(
                        id=artifact_id,
                        document_snapshot_id=DOCUMENT_ID,
                        artifact_type=artifact_type,
                        bucket=BUCKET,
                        object_key=reference.key,
                        sha256=reference.sha256,
                        size_bytes=reference.size_bytes,
                        media_type="application/json",
                        metadata_json={},
                    )
                )
        self.jobs = MySqlJobRepository(self.sessions, now_provider=lambda: NOW)
        self.documents = SqlAlchemyDocumentLifecycleRepository(
            self.sessions, snapshot_bucket=BUCKET
        )
        self.findings = SqlAlchemyFindingRepository(self.sessions)
        self.llm = FakeLlmProvider((self._extraction_json(),))
        self.embedding = FakeEmbeddingProvider(
            dimensions=2,
            model="embedding-v1",
            vectors={"Does the tender contain signed authorization?": (1.0, 0.0)},
        )
        self.workflow = LangGraphReviewWorkflow(
            SingleReviewWorkflow(
                self.llm,
                id_generator=SequentialIdGenerator(("finding-1",)),
            ),
            checkpointer=SqlAlchemyCheckpointSaver(self.sessions),
        )
        self.persister = ApprovalFindingPersister(
            jobs=self.jobs,
            documents=self.documents,
            findings=self.findings,
        )
        self.handler = ReviewJobHandler(
            jobs=self.jobs,
            documents=self.documents,
            rules=_RuleRepository(),  # type: ignore[arg-type]
            datasets=_DatasetRepository(),  # type: ignore[arg-type]
            artifact_store=self.store,
            embedding_provider=self.embedding,
            index_loader=RetrievalIndexLoader(self.store),
            workflow=self.workflow,
            findings=self.persister,
            clock=FixedClock(NOW),
            model_config_id=MODEL_ID,
            model_config_hash=MODEL_HASH,
            call_timeout_seconds=5,
            call_max_attempts=1,
            fault_injector=fault_injector,
            audit=audit,
        )

    def close(self) -> None:
        self.engine.dispose()
        self.directory.cleanup()

    def create_job(self, job_id: str = "job-1") -> JobMessage:
        def execution_reference(role: str):
            artifact_id, reference, _ = self.references[role]
            return {
                "artifact_id": artifact_id,
                "bucket": BUCKET,
                "object_key": reference.key,
                "sha256": reference.sha256,
            }

        draft = ReviewExecutionSpecDraft(
            document_snapshot_id=DOCUMENT_ID,
            document_sha256=DOCUMENT_HASH,
            rule_version_id=RULE_ID,
            rule_version_hash=RULE_HASH,
            dataset_version_id=DATASET_ID,
            dataset_version_hash=DATASET_HASH,
            model_config_id=MODEL_ID,
            model_config_hash=MODEL_HASH,
            query="Does the tender contain signed authorization?",
            retrieval_variant="hybrid-rrf-v1",
            retriever_artifact=execution_reference("retriever"),
            index_artifact=execution_reference("index"),
            chunk_artifact=execution_reference("chunk"),
        )
        spec = build_review_execution_spec(job_id, draft)
        self.jobs.create_review_job(
            ReviewJob(
                id=job_id,
                document_snapshot_id=DOCUMENT_ID,
                rule_version_id=RULE_ID,
                model_config_id=MODEL_ID,
                input_fingerprint="f" * 64,
                execution_spec_sha256=spec.input_sha256,
                available_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            ),
            execution_spec=spec,
        )
        return JobMessage(
            job_id=job_id,
            job_type="review",
            input_reference=DOCUMENT_ID,
        )

    @staticmethod
    def _extraction_json() -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "review_item_id": "authorization",
                "fields": [
                    {
                        "schema_version": 1,
                        "field_name": "authorization_text",
                        "value_type": "text",
                        "value": "signed authorization",
                        "sources": [
                            {
                                "schema_version": 1,
                                "source_id": "source-1",
                                "document_id": DOCUMENT_ID,
                                "chunk_id": "chunk-1",
                                "page_number": 7,
                                "section_path": ["Authorization"],
                                "excerpt": "signed authorization",
                            }
                        ],
                    }
                ],
            }
        )


class ReviewJobHandlerIntegrationTests(unittest.TestCase):
    def test_single_process_fake_fault_drills_resume_without_duplicate_side_effects(
        self,
    ) -> None:
        fault_points = (
            FAULT_AFTER_RETRIEVAL_CHECKPOINT,
            FAULT_AFTER_EXTRACTION_CHECKPOINT,
            FAULT_AFTER_REPORT_CHECKPOINT,
        )
        for fault_point in fault_points:
            with self.subTest(fault_point=fault_point):
                audit_sink = InMemoryAuditEventSink()
                audit = AuditService(
                    audit_sink,
                    SequentialIdGenerator(prefix="audit"),
                    FixedClock(NOW),
                )
                fixture = HandlerFixture(
                    fault_injector=OneShotFaultInjector(fault_point),
                    audit=audit,
                )
                try:
                    message = fixture.create_job(f"job-{fault_point.rsplit('.', 1)[-1]}")
                    context = _ExecutionContext()

                    interrupted = fixture.handler(message, context)  # type: ignore[arg-type]
                    recovered = fixture.handler(
                        message.model_copy(update={"attempt": 2}),
                        context,  # type: ignore[arg-type]
                    )

                    self.assertEqual(interrupted.status, JobHandlerStatus.FAILED)
                    self.assertTrue(interrupted.failure.retryable)
                    self.assertEqual(interrupted.failure.code, "injected_fault")
                    self.assertEqual(recovered.status, JobHandlerStatus.COMPLETED)
                    self.assertEqual(len(fixture.embedding.calls), 1)
                    self.assertEqual(len(fixture.llm.calls), 1)
                    self.assertEqual(
                        len(fixture.findings.list_findings(message.job_id)), 1
                    )

                    observability = [
                        checkpoint
                        for checkpoint in context.checkpoints
                        if checkpoint["node_name"] == "observability"
                    ]
                    self.assertGreaterEqual(len(observability), 2)
                    values = {
                        item["key"]: item["value"]
                        for item in observability[-1]["state_json"]["values"]
                    }
                    self.assertEqual(values["recovery_count"], "1")
                    self.assertEqual(values["rule_version"], RULE_ID)
                    self.assertEqual(values["dataset_version"], DATASET_ID)
                    self.assertEqual(values["model_config"], MODEL_ID)
                    self.assertEqual(
                        values["metrics_source"],
                        "langgraph_internal_channel+provider_adapter_logs",
                    )
                    self.assertEqual(
                        values["model_token_status"],
                        "not_collected:provider_metrics_are_log_only",
                    )
                    self.assertNotIn("prompt_tokens", values)
                    self.assertNotIn("completion_tokens", values)
                    self.assertTrue(
                        any(key.startswith("node_duration_ms:") for key in values)
                    )
                    self.assertEqual(context.checkpoints[-1]["node_name"], "langgraph:done")

                    events = audit.list_events()
                    self.assertEqual(
                        {event.action for event in events},
                        {"review.graph.interrupted", "review.graph.completed"},
                    )
                    for event in events:
                        serialized = event.model_dump(mode="json", by_alias=True)
                        self.assertEqual(serialized["job_id"], message.job_id)
                        self.assertEqual(serialized["thread_id"], message.job_id)
                        self.assertTrue(serialized["checkpoint_id"])
                        self.assertEqual(serialized["call_id"], f"review:{message.job_id}")
                        self.assertEqual(serialized["rule_version"], RULE_ID)
                        self.assertEqual(serialized["dataset_version"], DATASET_ID)
                        self.assertEqual(serialized["model_config"], MODEL_ID)
                        self.assertNotIn(
                            "signed authorization",
                            json.dumps(serialized, ensure_ascii=False),
                        )
                finally:
                    fixture.close()

    def test_real_langgraph_resume_is_idempotent_and_worker_checkpoint_is_pointer_only(
        self,
    ) -> None:
        fixture = HandlerFixture()
        try:
            message = fixture.create_job()
            context = _ExecutionContext()

            first = fixture.handler(message, context)  # type: ignore[arg-type]
            second = fixture.handler(message, context)  # type: ignore[arg-type]

            self.assertEqual(first.status, JobHandlerStatus.COMPLETED)
            self.assertEqual(second, first)
            self.assertEqual(len(fixture.llm.calls), 1)
            self.assertEqual(len(fixture.findings.list_findings(message.job_id)), 1)
            self.assertEqual(
                fixture.workflow.latest_checkpoint(message.job_id).thread_id,
                message.job_id,
            )
            values = context.checkpoints[-1]["state_json"]["values"]
            keys = {item["key"] for item in values}
            self.assertEqual(
                keys,
                {
                    "langgraph_thread_id",
                    "langgraph_checkpoint_id",
                    "lifecycle",
                    "retriever_artifact_id",
                    "index_artifact_id",
                    "chunk_artifact_id",
                },
            )
            self.assertNotIn("signed authorization", json.dumps(values))
        finally:
            fixture.close()

    def test_worker_uses_fenced_waiting_human_write_for_need_more_evidence(self) -> None:
        fixture = HandlerFixture(chunks_document_id="outside-document")
        try:
            message = fixture.create_job()
            worker = Worker(
                worker_id="worker-a2",
                repository=fixture.jobs,
                leases=fixture.jobs,
                handlers={"review": fixture.handler},
                clock=FixedClock(NOW),
                lease_seconds=30,
                poll_interval_seconds=0,
            )

            self.assertTrue(worker.run_once())

            durable = fixture.jobs.get_review_job(message.job_id)
            self.assertEqual(durable.status, JobLifecycle.WAITING_HUMAN)
            finding = fixture.findings.list_findings(message.job_id)[0]
            self.assertEqual(finding.workflow_state.value, "NEED_MORE_EVIDENCE")
            with self.assertRaises(ConflictError) as raised:
                fixture.jobs.mark_waiting_human(
                    message.job_id, durable.lease_token, JobResult(summary="stale")
                )
            self.assertEqual(raised.exception.code, "stale_lease")
        finally:
            fixture.close()

    def test_legacy_job_and_model_identity_conflict_fail_explicitly(self) -> None:
        fixture = HandlerFixture()
        try:
            message = fixture.create_job()
            fixture.handler._model_config_hash = "8" * 64
            outcome = fixture.handler(message, _ExecutionContext())  # type: ignore[arg-type]
            self.assertEqual(outcome.status, JobHandlerStatus.FAILED)
            self.assertEqual(
                outcome.failure.code, "review_execution_model_config_conflict"
            )

            fixture.jobs.create_review_job(
                ReviewJob(
                    id="legacy-job",
                    document_snapshot_id=DOCUMENT_ID,
                    rule_version_id=RULE_ID,
                    model_config_id=MODEL_ID,
                    input_fingerprint="e" * 64,
                    available_at=NOW,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            legacy = fixture.handler(
                JobMessage(
                    job_id="legacy-job",
                    job_type="review",
                    input_reference=DOCUMENT_ID,
                ),
                _ExecutionContext(),  # type: ignore[arg-type]
            )
            self.assertEqual(legacy.status, JobHandlerStatus.FAILED)
            self.assertEqual(legacy.failure.code, "review_execution_spec_required")
        finally:
            fixture.close()

    def test_rule_content_is_strictly_parsed_as_review_rule(self) -> None:
        fixture = HandlerFixture()
        try:
            message = fixture.create_job()
            fixture.handler._rules = _RuleRepository(
                content_json='{"rules":[{"pattern":"signed.*authorization"}]}'
            )
            outcome = fixture.handler(message, _ExecutionContext())  # type: ignore[arg-type]
            self.assertEqual(outcome.status, JobHandlerStatus.FAILED)
            self.assertEqual(outcome.failure.code, "review_execution_rule_invalid")
        finally:
            fixture.close()

    def test_retrieval_hash_excludes_latency_but_not_evidence(self) -> None:
        provenance = RetrievalProvenance(
            retriever_version="artifact-backed-hybrid-v1",
            embedding_model="embedding-v1",
            embedding_dimensions=2,
            chunk_config=RetrievalChunkConfig(
                strategy_name="structural",
                strategy_version="1",
                config_sha256="9" * 64,
            ),
            top_k=1,
            candidate_limit=1,
            manifest_sha256="a" * 64,
            chunk_catalog_sha256="b" * 64,
            index_sha256="c" * 64,
            latency_ms=1,
        )
        result = ArtifactSearchResult(
            retriever="artifact-hybrid:rrf",
            hits=(
                SearchHit(
                    chunk_id="chunk-1",
                    document_id=DOCUMENT_ID,
                    text="signed authorization",
                    section_path=("Authorization",),
                    page_start=7,
                    page_end=7,
                    score=1.0,
                    source="hybrid",
                    rank=1,
                ),
            ),
            provenance=provenance,
        )
        slower = result.model_copy(
            update={"provenance": provenance.model_copy(update={"latency_ms": 99})}
        )
        changed = result.model_copy(
            update={
                "hits": (
                    result.hits[0].model_copy(update={"text": "different evidence"}),
                )
            }
        )

        self.assertEqual(
            retrieval_results_sha256(result), retrieval_results_sha256(slower)
        )
        self.assertNotEqual(
            retrieval_results_sha256(result), retrieval_results_sha256(changed)
        )
        self.assertNotEqual(retrieval_results_sha256(result), "c" * 64)


class WorkerOutcomeTests(unittest.TestCase):
    def test_outcome_contract_covers_all_durable_terminal_branches(self) -> None:
        completed = JobHandlerOutcome(status="COMPLETED")
        waiting = JobHandlerOutcome(status="WAITING_HUMAN")
        failed = JobHandlerOutcome(
            status="FAILED",
            failure={
                "code": "model_failed",
                "message": "model failed",
                "category": "permanent",
                "retryable": False,
                "stage": "EXTRACTING",
            },
        )
        cancelled = JobHandlerOutcome(
            status="CANCELLED",
            failure={
                "code": "cancelled",
                "message": "cancelled",
                "category": "cancelled",
                "retryable": False,
                "stage": "RETRIEVING",
            },
        )
        self.assertEqual(
            [item.status.value for item in (completed, waiting, failed, cancelled)],
            ["COMPLETED", "WAITING_HUMAN", "FAILED", "CANCELLED"],
        )


if __name__ == "__main__":
    unittest.main()
