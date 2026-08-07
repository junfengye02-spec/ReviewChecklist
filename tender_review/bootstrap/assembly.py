from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping

from tender_review.documents.fakes import (
    FakeChunkingStrategy,
    FakeDocumentParser,
    FakeOcrProvider,
    InMemoryArtifactStore,
)
from tender_review.documents.ports import (
    ArtifactStore,
    ChunkingStrategy,
    DocumentParser,
    OcrProvider,
)
from tender_review.documents.application import (
    DocumentParsingJobHandler,
    DocumentService,
)
from tender_review.documents.lifecycle import (
    DocumentLifecycleRepository,
    DocumentLifecycleService,
    InMemoryDocumentLifecycleRepository,
)
from tender_review.documents.parsing.adapters.pymupdf import PyMuPDFStructuredParser
from tender_review.documents.parsing.application import DocumentParsingService
from tender_review.documents.parsing.chunking import StructuralChunker
from tender_review.documents.parsing.fakes import UnavailableOcrProvider
from tender_review.documents.storage import (
    ContentAddressedObjectStore,
    InMemoryContentAddressedStore,
)
from tender_review.jobs.fakes import FakeLeaseManager, InMemoryJobRepository
from tender_review.jobs.models import JobResult
from tender_review.jobs.ports import JobRepository, LeaseManager, ReviewJobRepository
from tender_review.jobs.public import ReviewJobService
from tender_review.findings.public import (
    FindingDecisionService,
    FindingRepository,
    InMemoryFindingRepository,
)
from tender_review.rule_management.public import (
    InMemoryRuleVersionRepository,
    RuleVersionRepository,
    RuleVersionService,
)
from tender_review.evaluation.public import (
    AnnotationEvaluationDatasetResolver,
    AnnotationDatasetRepository,
    AnnotationDatasetService,
    DatasetVersionRepository,
    DatasetVersionService,
    EvaluationRunRepository,
    EvaluationRunService,
    InMemoryAnnotationDatasetRepository,
    InMemoryDatasetVersionRepository,
    InMemoryEvaluationRunRepository,
    RepositoryAnnotationReferenceValidator,
    RepositoryHumanDecisionResolver,
)
from tender_review.optimization.public import (
    A4OptimizationReadinessVerifier,
    FakeCandidateGenerator,
    FakeRegressionEvaluator,
    InMemoryOptimizationRepository,
    OptimizationRepository,
    OptimizationService,
    RootCauseAnalyzer,
    RuleVersionCandidateStager,
)
from tender_review.optimization.unavailable import (
    UnavailableCandidateGenerator,
    UnavailableRegressionEvaluator,
)
from tender_review.stage8.demo import assemble_stage8
from tender_review.stage8.public import AuditService, Stage8QueryService
from tender_review.retrieval.fakes import (
    FakeEmbeddingProvider,
    FakeFusionStrategy,
    FakeRetriever,
)
from tender_review.retrieval.public import EmbeddingProvider, FusionStrategy, Retriever
from tender_review.review.fakes import FakeLlmProvider
from tender_review.review.ports import LlmProvider, ReviewTool
from tender_review.shared.clock import Clock, SystemClock
from tender_review.shared.config import AppSettings
from tender_review.shared.errors import PermanentError
from tender_review.shared.health import ReadinessCheck, StaticReadinessCheck
from tender_review.shared.ids import IdGenerator, UuidGenerator

if TYPE_CHECKING:
    from fastapi import FastAPI
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from sqlalchemy import Engine
    from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True)
class ApplicationContainer:
    settings: AppSettings
    clock: Clock
    ids: IdGenerator
    job_repository: JobRepository
    review_job_repository: ReviewJobRepository
    lease_manager: LeaseManager
    review_jobs: ReviewJobService
    documents: DocumentService
    finding_repository: FindingRepository
    finding_decisions: FindingDecisionService
    rule_version_repository: RuleVersionRepository
    rule_versions: RuleVersionService
    dataset_version_repository: DatasetVersionRepository
    dataset_versions: DatasetVersionService
    annotation_dataset_repository: AnnotationDatasetRepository
    annotation_datasets: AnnotationDatasetService
    evaluation_run_repository: EvaluationRunRepository
    evaluations: EvaluationRunService
    optimization_repository: OptimizationRepository
    optimizations: OptimizationService
    stage8: Stage8QueryService
    audit: AuditService
    document_lifecycle: DocumentLifecycleService
    document_repository: DocumentLifecycleRepository
    document_store: ContentAddressedObjectStore
    artifact_store: ArtifactStore
    document_parser: DocumentParser | None
    ocr_provider: OcrProvider | None
    chunking_strategy: ChunkingStrategy | None
    embedding_provider: EmbeddingProvider | None
    retriever: Retriever | None
    fusion_strategy: FusionStrategy | None
    llm_provider: LlmProvider | None
    checkpoint_saver: BaseCheckpointSaver[Any] | None
    retrieval_index_loader: Any | None
    review_workflow: Any | None
    review_job_handler: Any | None
    review_tools: Mapping[str, ReviewTool]
    worker_handlers: Mapping[str, Callable[..., JobResult]]
    readiness_checks: tuple[ReadinessCheck, ...]
    database_engine: Engine | None = None
    session_factory: sessionmaker[Session] | None = None

    def with_overrides(self, **values: Any) -> "ApplicationContainer":
        return replace(self, **values)

    def close(self) -> None:
        if self.database_engine is not None:
            self.database_engine.dispose()


def build_container(settings: AppSettings | None = None) -> ApplicationContainer:
    resolved = settings or AppSettings.from_env()
    if resolved.adapter_mode == "fake":
        if resolved.environment.strip().lower() in {"prod", "production", "staging"}:
            raise PermanentError(
                "Fake adapters are forbidden in production-like environments",
                code="fake_adapters_forbidden",
                details={"environment": resolved.environment},
            )
        return _build_fake_container(resolved)
    if resolved.adapter_mode == "production":
        return _build_production_container(resolved)
    raise PermanentError(
        "Unsupported adapter mode",
        code="adapter_mode_unavailable",
        details={"adapter_mode": resolved.adapter_mode},
    )


def _build_fake_container(settings: AppSettings) -> ApplicationContainer:
    clock = SystemClock()
    ids = UuidGenerator()
    repository = InMemoryJobRepository()
    document_store = InMemoryContentAddressedStore()
    document_repository = InMemoryDocumentLifecycleRepository(ids)
    document_lifecycle = DocumentLifecycleService(
        storage=document_store,
        repository=document_repository,
    )
    structured_parser = PyMuPDFStructuredParser()
    parsing = DocumentParsingService(
        parser=structured_parser,
        renderer=structured_parser,
        ocr_provider=UnavailableOcrProvider(),
    )
    documents = DocumentService(
        lifecycle=document_lifecycle,
        repository=document_repository,
        parser=parsing,
        chunker=StructuralChunker(),
    )
    parse_handler = DocumentParsingJobHandler(documents)
    finding_repository = InMemoryFindingRepository()
    rule_version_repository = InMemoryRuleVersionRepository()
    dataset_version_repository = InMemoryDatasetVersionRepository()
    annotation_dataset_repository = InMemoryAnnotationDatasetRepository()
    evaluation_run_repository = InMemoryEvaluationRunRepository()
    optimization_repository = InMemoryOptimizationRepository()
    evaluations = EvaluationRunService(
        evaluation_run_repository,
        AnnotationEvaluationDatasetResolver(annotation_dataset_repository),
        rule_version_repository,
        ids,
        clock,
    )
    rule_version_repository.set_release_gate_verifier(evaluations)
    rule_versions = RuleVersionService(rule_version_repository, ids, clock, evaluations)
    optimizations = OptimizationService(
        repository=optimization_repository,
        rule_versions=rule_version_repository,
        datasets=dataset_version_repository,
        ids=ids,
        clock=clock,
        root_causes=RootCauseAnalyzer(FakeLlmProvider()),
        candidates=FakeCandidateGenerator(),
        evaluator=FakeRegressionEvaluator(),
        stager=RuleVersionCandidateStager(rule_versions, rule_version_repository),
        readiness=A4OptimizationReadinessVerifier(
            annotation_dataset_repository,
            evaluation_run_repository,
            clock,
        ),
    )
    stage8 = assemble_stage8(
        settings=settings,
        ids=ids,
        clock=clock,
        review_jobs=repository,
        findings=finding_repository,
        rules=rule_version_repository,
        optimizations=optimization_repository,
    )
    return ApplicationContainer(
        settings=settings,
        clock=clock,
        ids=ids,
        job_repository=repository,
        review_job_repository=repository,
        lease_manager=FakeLeaseManager(),
        review_jobs=ReviewJobService(repository=repository, ids=ids, clock=clock),
        documents=documents,
        finding_repository=finding_repository,
        finding_decisions=FindingDecisionService(finding_repository, ids, clock),
        rule_version_repository=rule_version_repository,
        rule_versions=rule_versions,
        dataset_version_repository=dataset_version_repository,
        dataset_versions=DatasetVersionService(dataset_version_repository, ids, clock),
        annotation_dataset_repository=annotation_dataset_repository,
        annotation_datasets=AnnotationDatasetService(
            annotation_dataset_repository,
            RepositoryHumanDecisionResolver(finding_repository),
            ids,
            clock,
            RepositoryAnnotationReferenceValidator(
                document_repository,
                finding_repository,
                rule_version_repository,
            ),
        ),
        evaluation_run_repository=evaluation_run_repository,
        evaluations=evaluations,
        optimization_repository=optimization_repository,
        optimizations=optimizations,
        stage8=stage8.queries,
        audit=stage8.audit,
        document_lifecycle=document_lifecycle,
        document_repository=document_repository,
        document_store=document_store,
        artifact_store=InMemoryArtifactStore(),
        document_parser=FakeDocumentParser(),
        ocr_provider=FakeOcrProvider(),
        chunking_strategy=FakeChunkingStrategy(),
        embedding_provider=FakeEmbeddingProvider(),
        retriever=FakeRetriever(),
        fusion_strategy=FakeFusionStrategy(),
        llm_provider=FakeLlmProvider(),
        checkpoint_saver=None,
        retrieval_index_loader=None,
        review_workflow=None,
        review_job_handler=None,
        review_tools=MappingProxyType({}),
        worker_handlers=MappingProxyType({parse_handler.job_type: parse_handler}),
        readiness_checks=(
            StaticReadinessCheck(
                "dependencies",
                ready=True,
                detail="offline fake adapters assembled",
            ),
        ),
    )


def _build_production_container(settings: AppSettings) -> ApplicationContainer:
    llm_api_key = settings.llm_api_key.get_secret_value()
    embedding_api_key = settings.embedding_api_key.get_secret_value()
    missing = [
        name
        for name, value in (
            ("database_url", settings.database_url),
            ("minio_endpoint", settings.minio_endpoint),
            ("minio_access_key", settings.minio_access_key),
            ("minio_secret_key", settings.minio_secret_key),
            ("minio_bucket", settings.minio_bucket),
            ("llm_base_url", settings.llm_base_url),
            ("llm_api_key", llm_api_key),
            ("llm_model", settings.llm_model),
            ("embedding_base_url", settings.embedding_base_url),
            ("embedding_api_key", embedding_api_key),
            ("embedding_model", settings.embedding_model),
            ("model_config_id", settings.model_config_id),
        )
        if not value.strip()
    ]
    if missing:
        raise PermanentError(
            "Production adapter configuration is incomplete",
            code="production_configuration_invalid",
            details={"missing_fields": missing},
        )

    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        database_url = make_url(settings.database_url)
    except ArgumentError as exc:
        raise PermanentError(
            "Production database URL is invalid",
            code="production_database_url_invalid",
        ) from exc
    if (
        database_url.get_backend_name() != "mysql"
        or database_url.get_driver_name() != "pymysql"
    ):
        raise PermanentError(
            "Production requires a mysql+pymysql database URL",
            code="production_database_driver_invalid",
            details={"driver": database_url.drivername},
        )

    from tender_review.infrastructure.ai import (
        OpenAICompatibleEmbeddingProvider,
        OpenAICompatibleLlmProvider,
    )

    # These are standalone adapter caps. SingleReviewWorkflow gives each outer
    # attempt a child CallContext with max_attempts=1, so its root call remains
    # the sole total LLM attempt budget instead of multiplying both layers.
    try:
        llm_provider = OpenAICompatibleLlmProvider(
            api_key=llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_attempts,
            temperature=settings.llm_temperature,
        )
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            api_key=embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            base_url=settings.embedding_base_url,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_attempts=settings.embedding_max_attempts,
        )
    except ValueError as exc:
        raise PermanentError(
            "Production AI configuration is invalid",
            code="production_ai_configuration_invalid",
        ) from exc

    from tender_review.infrastructure.database import (
        DatabaseHealthAdapter,
        ModelConfigHealthAdapter,
        create_database_engine,
        create_session_factory,
    )
    from tender_review.infrastructure.database.document_lifecycle import (
        SqlAlchemyDocumentLifecycleRepository,
    )
    from tender_review.infrastructure.database.langgraph_checkpoints import (
        SqlAlchemyCheckpointSaver,
    )
    from tender_review.infrastructure.object_storage import MinioArtifactStore

    engine = create_database_engine(
        settings.database_url,
        echo=settings.database_echo,
        connect_timeout_seconds=settings.database_connect_timeout_seconds,
    )
    sessions = create_session_factory(engine)
    checkpoint_saver = SqlAlchemyCheckpointSaver(sessions)
    try:
        artifact_store = MinioArtifactStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
            region=settings.minio_region,
            timeout_seconds=settings.minio_timeout_seconds,
            max_attempts=settings.minio_max_attempts,
        )
    except ValueError as exc:
        engine.dispose()
        raise PermanentError(
            "Production MinIO configuration is invalid",
            code="production_minio_configuration_invalid",
        ) from exc
    clock = SystemClock()
    ids = UuidGenerator()
    from tender_review.jobs.adapters import MySqlJobRepository

    repository = MySqlJobRepository(sessions)
    document_repository = SqlAlchemyDocumentLifecycleRepository(
        sessions,
        snapshot_bucket=settings.minio_bucket,
    )
    document_lifecycle = DocumentLifecycleService(
        storage=artifact_store,
        repository=document_repository,
    )
    structured_parser = PyMuPDFStructuredParser()
    parsing = DocumentParsingService(
        parser=structured_parser,
        renderer=structured_parser,
        ocr_provider=UnavailableOcrProvider(),
    )
    documents = DocumentService(
        lifecycle=document_lifecycle,
        repository=document_repository,
        parser=parsing,
        chunker=StructuralChunker(),
    )
    parse_handler = DocumentParsingJobHandler(documents)
    # Stage 6 database adapters are assembled below after the shared session factory.
    from tender_review.infrastructure.database.dataset_versions import (
        SqlAlchemyDatasetVersionRepository,
    )
    from tender_review.infrastructure.database.annotation_datasets import (
        SqlAlchemyAnnotationDatasetRepository,
    )
    from tender_review.infrastructure.database.evaluation_runs import (
        SqlAlchemyEvaluationRunRepository,
    )
    from tender_review.infrastructure.database.finding_records import (
        SqlAlchemyFindingRepository,
    )
    from tender_review.infrastructure.database.rule_versions import (
        SqlAlchemyRuleVersionRepository,
    )
    from tender_review.infrastructure.database.optimization_jobs import (
        SqlAlchemyOptimizationRepository,
    )

    finding_repository = SqlAlchemyFindingRepository(sessions)
    rule_version_repository = SqlAlchemyRuleVersionRepository(sessions)
    dataset_version_repository = SqlAlchemyDatasetVersionRepository(sessions)
    annotation_dataset_repository = SqlAlchemyAnnotationDatasetRepository(sessions)
    evaluation_run_repository = SqlAlchemyEvaluationRunRepository(sessions)
    optimization_repository = SqlAlchemyOptimizationRepository(sessions)
    evaluations = EvaluationRunService(
        evaluation_run_repository,
        AnnotationEvaluationDatasetResolver(annotation_dataset_repository),
        rule_version_repository,
        ids,
        clock,
    )
    rule_versions = RuleVersionService(rule_version_repository, ids, clock, evaluations)
    optimizations = OptimizationService(
        repository=optimization_repository,
        rule_versions=rule_version_repository,
        datasets=dataset_version_repository,
        ids=ids,
        clock=clock,
        root_causes=RootCauseAnalyzer(llm_provider),
        candidates=UnavailableCandidateGenerator(),
        evaluator=UnavailableRegressionEvaluator(),
        stager=RuleVersionCandidateStager(rule_versions, rule_version_repository),
        readiness=A4OptimizationReadinessVerifier(
            annotation_dataset_repository,
            evaluation_run_repository,
            clock,
        ),
        checkpointer=checkpoint_saver,
    )
    stage8 = assemble_stage8(
        settings=settings,
        ids=ids,
        clock=clock,
        review_jobs=repository,
        findings=finding_repository,
        rules=rule_version_repository,
        optimizations=optimization_repository,
    )
    from tender_review.retrieval.public import RetrievalIndexLoader
    from tender_review.review.public import LangGraphReviewWorkflow, SingleReviewWorkflow
    from tender_review.worker.review_handler import (
        ApprovalFindingPersister,
        ReviewJobHandler,
    )

    retrieval_index_loader = RetrievalIndexLoader(artifact_store)
    finding_persister = ApprovalFindingPersister(
        jobs=repository,
        documents=document_repository,
        findings=finding_repository,
    )
    review_workflow = LangGraphReviewWorkflow(
        SingleReviewWorkflow(llm_provider, id_generator=ids),
        checkpointer=checkpoint_saver,
    )
    review_handler = ReviewJobHandler(
        jobs=repository,
        documents=document_repository,
        rules=rule_version_repository,
        datasets=dataset_version_repository,
        artifact_store=artifact_store,
        embedding_provider=embedding_provider,
        index_loader=retrieval_index_loader,
        workflow=review_workflow,
        findings=finding_persister,
        clock=clock,
        model_config_id=settings.model_config_id,
        model_config_hash=settings.model_config_hash,
        call_timeout_seconds=settings.llm_timeout_seconds,
        call_max_attempts=settings.llm_max_attempts,
        audit=stage8.audit,
    )
    return ApplicationContainer(
        settings=settings,
        clock=clock,
        ids=ids,
        job_repository=repository,
        review_job_repository=repository,
        lease_manager=repository,
        review_jobs=ReviewJobService(repository=repository, ids=ids, clock=clock),
        documents=documents,
        finding_repository=finding_repository,
        finding_decisions=FindingDecisionService(finding_repository, ids, clock),
        rule_version_repository=rule_version_repository,
        rule_versions=rule_versions,
        dataset_version_repository=dataset_version_repository,
        dataset_versions=DatasetVersionService(dataset_version_repository, ids, clock),
        annotation_dataset_repository=annotation_dataset_repository,
        annotation_datasets=AnnotationDatasetService(
            annotation_dataset_repository,
            RepositoryHumanDecisionResolver(finding_repository),
            ids,
            clock,
            RepositoryAnnotationReferenceValidator(
                document_repository,
                finding_repository,
                rule_version_repository,
            ),
        ),
        evaluation_run_repository=evaluation_run_repository,
        evaluations=evaluations,
        optimization_repository=optimization_repository,
        optimizations=optimizations,
        stage8=stage8.queries,
        audit=stage8.audit,
        document_lifecycle=document_lifecycle,
        document_repository=document_repository,
        document_store=artifact_store,
        artifact_store=artifact_store,
        document_parser=None,
        ocr_provider=None,
        chunking_strategy=None,
        embedding_provider=embedding_provider,
        retriever=None,
        fusion_strategy=None,
        llm_provider=llm_provider,
        checkpoint_saver=checkpoint_saver,
        retrieval_index_loader=retrieval_index_loader,
        review_workflow=review_workflow,
        review_job_handler=review_handler,
        review_tools=MappingProxyType({}),
        worker_handlers=MappingProxyType(
            {
                review_handler.job_type: review_handler,
                parse_handler.job_type: parse_handler,
            }
        ),
        readiness_checks=(
            DatabaseHealthAdapter(engine),
            artifact_store,
            ModelConfigHealthAdapter(
                sessions,
                model_config_id=settings.model_config_id,
                model_config_hash=settings.model_config_hash,
            ),
        ),
        database_engine=engine,
        session_factory=sessions,
    )


def create_api_app(settings: AppSettings | None = None) -> "FastAPI":
    from tender_review.api.app import create_app

    return create_app(build_container(settings))
    EvaluationRunRepository,
    EvaluationRunService,
