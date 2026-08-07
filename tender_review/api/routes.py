from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Header, Query, Request, Response, UploadFile, status

from tender_review.shared.errors import PermanentError, ServiceError
from tender_review.shared.health import CheckResult, ReadinessResult
from tender_review.findings.public import DecisionOutcome, Finding, HumanDecision
from tender_review.jobs.public import JobCheckpoint, ReviewExecutionSpec
from tender_review.rule_management.public import RuleVersion, RuleVersionDiff
from tender_review.optimization.public import OptimizationAttempt, OptimizationJob
from tender_review.performance.public import (
    A7AdmissionReport,
    RunStatus as A7RunStatus,
    create_unavailable_report as create_unavailable_a7_report,
    load_report as load_a7_report,
)
from tender_review.evaluation.public import (
    EvaluationReport as A4EvaluationReport,
    EvaluationRun as A4EvaluationRun,
    AnnotationDatasetVersion,
    AnnotationSampleStatus,
    DatasetAnnotationSample,
    DatasetStatus,
)
from tender_review.stage8.public import (
    ActorKind,
    AuditEvent,
    AuditResult,
    EvaluationReport,
    EvaluationRun,
    ReportSourceType,
    WorkbenchResourceIndex,
    stable_sha256 as audit_sha256,
)

from .dependencies import ApiContainer
from .schemas import (
    ApiIndexResponse,
    ComponentCheck,
    CreateAnnotationDatasetRequest,
    CreateAnnotationDatasetRevisionRequest,
    CreateEvaluationRunRequest,
    CreateReviewJobRequest,
    DocumentSnapshotResponse,
    EvaluateRuleVersionRequest,
    HumanDecisionRequest,
    LivenessResponse,
    ReadinessResponse,
    ReviewJobResponse,
    CreateRuleVersionRequest,
    PublishRuleVersionRequest,
    RollbackRuleSetRequest,
    SubmitAnnotationLabelRequest,
    OptimizeRuleVersionRequest,
    A5OptimizeRuleVersionRequest,
)


_CREATE_REVIEW_JOB_SCOPE = "POST:/api/v1/review-jobs"


def build_health_router(container: ApiContainer) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health/live", response_model=LivenessResponse)
    def live() -> LivenessResponse:
        return LivenessResponse(
            service=container.settings.service_name,
            version=container.settings.version,
        )

    @router.get("/health/ready", response_model=ReadinessResponse)
    def ready(response: Response) -> ReadinessResponse:
        results = tuple(_run_check(check) for check in container.readiness_checks)
        is_ready = all(result.ready for result in results)
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            service=container.settings.service_name,
            version=container.settings.version,
            checks={
                result.name: ComponentCheck(
                    status="ready" if result.ready else "not_ready",
                    detail=result.detail,
                )
                for result in results
            },
        )

    return router


def build_v1_router(container: ApiContainer) -> APIRouter:
    router = APIRouter(prefix=container.settings.api_prefix, tags=["api"])

    def index_response() -> ApiIndexResponse:
        return ApiIndexResponse(
            service=container.settings.service_name,
            version=container.settings.version,
        )

    router.add_api_route(
        "", index_response, methods=["GET"], response_model=ApiIndexResponse
    )

    @router.post(
        "/documents",
        response_model=DocumentSnapshotResponse,
        status_code=status.HTTP_201_CREATED,
        responses={409: {"description": "Source or content hash conflict"}},
        tags=["documents"],
    )
    async def upload_document(
        response: Response,
        file: UploadFile = File(...),
        source_system: str = Form(..., min_length=1, max_length=64),
        source_document_id: str = Form(..., min_length=1, max_length=255),
    ) -> DocumentSnapshotResponse:
        media_type = (file.content_type or "").lower().strip()
        if media_type != "application/pdf":
            raise PermanentError(
                "Only application/pdf uploads are supported",
                code="document_media_type_unsupported",
                details={"media_type": media_type or None},
            )
        content = bytearray()
        try:
            while chunk := await file.read(1024 * 1024):
                content.extend(chunk)
                if len(content) > container.settings.document_max_upload_bytes:
                    raise PermanentError(
                        "Uploaded document exceeds the configured size limit",
                        code="document_too_large",
                        details={
                            "max_bytes": container.settings.document_max_upload_bytes
                        },
                    )
        finally:
            await file.close()
        document = bytes(content)
        if not document:
            raise PermanentError(
                "Uploaded document is empty", code="document_empty"
            )
        if not document.startswith(b"%PDF-"):
            raise PermanentError(
                "Uploaded content is not a PDF file", code="document_pdf_invalid"
            )
        filename = (file.filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or len(filename) > 512:
            raise PermanentError(
                "Uploaded file name is invalid", code="document_file_name_invalid"
            )
        outcome = container.documents.upload(
            source_system=source_system,
            source_document_id=source_document_id,
            file_name=filename,
            content=document,
            media_type=media_type,
        )
        if not outcome.created:
            response.status_code = status.HTTP_200_OK
        return DocumentSnapshotResponse.from_snapshot(
            outcome.snapshot, created=outcome.created
        )
    router.add_api_route(
        "/",
        index_response,
        methods=["GET"],
        response_model=ApiIndexResponse,
    )

    @router.get(
        "/a7/admission-report",
        response_model=A7AdmissionReport,
        tags=["performance"],
    )
    def get_a7_admission_report() -> A7AdmissionReport:
        configured_path = container.settings.a7_admission_report_path.strip()
        if not configured_path:
            return create_unavailable_a7_report(
                status=A7RunStatus.NOT_RUN,
                blockers=(
                    "尚未配置经过验证的真实 A7 报告。",
                    "尚未运行真实 MySQL、MinIO、模型、PDF 和独立 Worker 压测。",
                    "未触发 RocketMQ 或 Redis 准入。",
                ),
            )
        try:
            return load_a7_report(
                configured_path,
                trusted_attestation_key=(
                    container.settings.a7_attestation_key.get_secret_value()
                ),
                trusted_attestation_key_id=container.settings.a7_attestation_key_id,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise PermanentError(
                "Configured A7 report failed integrity or provenance validation",
                code="a7_report_invalid",
                details={"error_type": type(exc).__name__},
            ) from exc

    @router.post(
        "/review-jobs",
        response_model=ReviewJobResponse,
        status_code=status.HTTP_201_CREATED,
        responses={409: {"description": "Idempotency key conflict"}},
        tags=["review-jobs"],
    )
    def create_review_job(
        request: CreateReviewJobRequest,
        response: Response,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=1, max_length=255
        ),
        caller_id: str | None = Header(
            default=None, alias="X-Caller-ID", max_length=255
        ),
    ) -> ReviewJobResponse:
        outcome = container.review_jobs.create(
            request.to_command(),
            caller_id=_normalized_header(caller_id, default="anonymous"),
            scope=_CREATE_REVIEW_JOB_SCOPE,
            idempotency_key=_normalized_header(idempotency_key),
        )
        if not outcome.created:
            response.status_code = status.HTTP_200_OK
        return ReviewJobResponse.from_job(outcome.job)

    @router.get(
        "/review-jobs/{job_id}",
        response_model=ReviewJobResponse,
        tags=["review-jobs"],
    )
    def get_review_job(job_id: str) -> ReviewJobResponse:
        return ReviewJobResponse.from_job(container.review_jobs.get(job_id))

    @router.get(
        "/review-jobs/{job_id}/execution-spec",
        response_model=ReviewExecutionSpec,
        tags=["review-jobs"],
    )
    def get_review_execution_spec(job_id: str) -> ReviewExecutionSpec:
        return container.review_jobs.get_execution_spec(job_id)

    @router.get(
        "/review-jobs/{job_id}/checkpoints",
        response_model=list[JobCheckpoint],
        tags=["review-jobs"],
    )
    def list_review_job_checkpoints(job_id: str) -> tuple[JobCheckpoint, ...]:
        return container.review_jobs.list_checkpoints(job_id)

    @router.get(
        "/review-jobs/{job_id}/findings",
        response_model=list[Finding],
        tags=["findings"],
    )
    def list_review_job_findings(job_id: str) -> tuple[Finding, ...]:
        return container.stage8.list_findings(job_id)

    @router.post(
        "/review-jobs/{job_id}/cancel",
        response_model=ReviewJobResponse,
        tags=["review-jobs"],
    )
    def cancel_review_job(job_id: str) -> ReviewJobResponse:
        return ReviewJobResponse.from_job(container.review_jobs.cancel(job_id))

    @router.post(
        "/review-jobs/{job_id}/rerun",
        response_model=ReviewJobResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["review-jobs"],
    )
    def rerun_review_job(job_id: str) -> ReviewJobResponse:
        return ReviewJobResponse.from_job(container.review_jobs.rerun(job_id))

    @router.post(
        "/findings/{finding_id}/decisions",
        response_model=DecisionOutcome,
        status_code=status.HTTP_201_CREATED,
        tags=["findings"],
    )
    def submit_finding_decision(
        finding_id: str,
        body: HumanDecisionRequest,
        request: Request,
        call_id: str | None = Header(default=None, alias="X-Call-ID", max_length=128),
    ) -> DecisionOutcome:
        finding = container.stage8.get_finding(finding_id)
        source_type = _finding_source_type(finding.provenance.source_kind)
        request_id, resolved_call_id = _request_identity(request, call_id)
        try:
            outcome = container.finding_decisions.submit(body.to_command(finding_id))
        except ServiceError:
            container.audit.record(
                actor_kind=ActorKind.HUMAN,
                actor_id=body.reviewer_id,
                action="finding.decision.submit",
                resource_type="finding",
                resource_id=finding_id,
                source_type=source_type,
                provenance_status=finding.provenance.status,
                claims_allowed=finding.provenance.claims_allowed,
                call_id=resolved_call_id,
                request_id=request_id,
                result=AuditResult.REJECTED,
                before_sha256=finding.finding_content_sha256,
                artifact_sha256s=(
                    finding.provenance.review_input_sha256,
                    finding.provenance.retrieval_results_sha256,
                ),
            )
            raise
        container.audit.record(
            actor_kind=ActorKind.HUMAN,
            actor_id=body.reviewer_id,
            action="finding.decision.submit",
            resource_type="finding",
            resource_id=finding_id,
            source_type=source_type,
            provenance_status=finding.provenance.status,
            claims_allowed=finding.provenance.claims_allowed,
            call_id=resolved_call_id,
            request_id=request_id,
            result=AuditResult.SUCCEEDED,
            before_sha256=finding.finding_content_sha256,
            after_sha256=outcome.decision.decision_sha256,
            artifact_sha256s=(
                finding.provenance.review_input_sha256,
                finding.provenance.retrieval_results_sha256,
            ),
        )
        return outcome

    @router.get(
        "/findings/{finding_id}", response_model=Finding, tags=["findings"]
    )
    def get_finding(finding_id: str) -> Finding:
        return container.stage8.get_finding(finding_id)

    @router.get(
        "/findings/{finding_id}/decisions",
        response_model=list[HumanDecision],
        tags=["findings"],
    )
    def list_finding_decisions(finding_id: str) -> tuple[HumanDecision, ...]:
        return container.stage8.list_finding_decisions(finding_id)

    @router.get(
        "/rule-sets/{rule_set_id}/versions",
        response_model=list[RuleVersion],
        tags=["rules"],
    )
    def list_rule_versions(rule_set_id: str) -> tuple[RuleVersion, ...]:
        return container.rule_versions.list_versions(rule_set_id)

    @router.post(
        "/rule-sets/{rule_set_id}/versions",
        response_model=RuleVersion,
        status_code=status.HTTP_201_CREATED,
        tags=["rules"],
    )
    def create_rule_version(
        rule_set_id: str, request: CreateRuleVersionRequest
    ) -> RuleVersion:
        return container.rule_versions.create_version(request.to_command(rule_set_id))

    @router.get(
        "/rule-versions/{version_id}/diff",
        response_model=RuleVersionDiff,
        tags=["rules"],
    )
    def diff_rule_version(version_id: str, against_version_id: str) -> RuleVersionDiff:
        return container.rule_versions.diff(against_version_id, version_id)

    @router.get(
        "/rule-versions/{version_id}",
        response_model=RuleVersion,
        tags=["rules"],
    )
    def get_rule_version(version_id: str) -> RuleVersion:
        return container.stage8.get_rule_version(version_id)

    @router.post(
        "/rule-versions/{version_id}/evaluate",
        response_model=RuleVersion,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["rules"],
    )
    def evaluate_rule_version(
        version_id: str, request: EvaluateRuleVersionRequest
    ) -> RuleVersion:
        return container.rule_versions.request_evaluation(
            version_id, request.dataset_version_id
        )

    @router.post(
        "/rule-versions/{version_id}/optimize",
        response_model=OptimizationJob,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["optimization"],
    )
    def optimize_rule_version(
        version_id: str, request: OptimizeRuleVersionRequest
    ) -> OptimizationJob:
        return container.optimizations.create(request.to_command(version_id))

    @router.post(
        "/a5/rule-versions/{version_id}/optimize",
        response_model=OptimizationJob,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["optimization"],
    )
    def optimize_rule_version_with_a4_evidence(
        version_id: str, request: A5OptimizeRuleVersionRequest
    ) -> OptimizationJob:
        return container.optimizations.create(request.to_command(version_id))

    @router.get(
        "/optimization-jobs/{optimization_job_id}",
        response_model=OptimizationJob,
        tags=["optimization"],
    )
    def get_optimization_job(optimization_job_id: str) -> OptimizationJob:
        return container.optimizations.get(optimization_job_id)

    @router.get(
        "/optimization-jobs/{optimization_job_id}/attempts",
        response_model=list[OptimizationAttempt],
        tags=["optimization"],
    )
    def list_optimization_attempts(
        optimization_job_id: str,
    ) -> tuple[OptimizationAttempt, ...]:
        return container.optimizations.list_attempts(optimization_job_id)

    @router.post(
        "/optimization-jobs/{optimization_job_id}/cancel",
        response_model=OptimizationJob,
        tags=["optimization"],
    )
    def cancel_optimization_job(optimization_job_id: str) -> OptimizationJob:
        return container.optimizations.cancel(optimization_job_id)

    @router.post(
        "/annotation-datasets",
        response_model=AnnotationDatasetVersion,
        status_code=status.HTTP_201_CREATED,
        tags=["annotation-datasets"],
    )
    def create_annotation_dataset(
        body: CreateAnnotationDatasetRequest,
    ) -> AnnotationDatasetVersion:
        return container.annotation_datasets.create_version(body.to_command())

    @router.get(
        "/annotation-datasets",
        response_model=list[AnnotationDatasetVersion],
        tags=["annotation-datasets"],
    )
    def list_annotation_datasets(
        dataset_name: str | None = None,
        dataset_status: DatasetStatus | None = Query(default=None, alias="status"),
        sample_status: AnnotationSampleStatus | None = None,
    ) -> tuple[AnnotationDatasetVersion, ...]:
        return container.annotation_datasets.list_versions(
            dataset_name=dataset_name,
            status=dataset_status,
            sample_status=sample_status,
        )

    @router.get(
        "/annotation-datasets/{dataset_version_id}",
        response_model=AnnotationDatasetVersion,
        tags=["annotation-datasets"],
    )
    def get_annotation_dataset(dataset_version_id: str) -> AnnotationDatasetVersion:
        return container.annotation_datasets.get_version(dataset_version_id)

    @router.get(
        "/annotation-datasets/{dataset_version_id}/samples",
        response_model=list[DatasetAnnotationSample],
        tags=["annotation-datasets"],
    )
    def list_annotation_samples(
        dataset_version_id: str,
        sample_status: AnnotationSampleStatus | None = Query(default=None, alias="status"),
    ) -> tuple[DatasetAnnotationSample, ...]:
        return container.annotation_datasets.list_samples(
            dataset_version_id, status=sample_status
        )

    @router.post(
        "/annotation-datasets/{dataset_version_id}/samples/{sample_id}/annotations",
        response_model=AnnotationDatasetVersion,
        tags=["annotation-datasets"],
    )
    def submit_annotation(
        dataset_version_id: str,
        sample_id: str,
        body: SubmitAnnotationLabelRequest,
    ) -> AnnotationDatasetVersion:
        return container.annotation_datasets.submit_annotation(
            body.to_command(dataset_version_id, sample_id)
        )

    @router.post(
        "/annotation-datasets/{dataset_version_id}/samples/{sample_id}/reviews",
        response_model=AnnotationDatasetVersion,
        tags=["annotation-datasets"],
    )
    def submit_annotation_review(
        dataset_version_id: str,
        sample_id: str,
        body: SubmitAnnotationLabelRequest,
    ) -> AnnotationDatasetVersion:
        return container.annotation_datasets.submit_review(
            body.to_command(dataset_version_id, sample_id)
        )

    @router.post(
        "/annotation-datasets/{dataset_version_id}/samples/{sample_id}/adjudications",
        response_model=AnnotationDatasetVersion,
        tags=["annotation-datasets"],
    )
    def adjudicate_annotation_conflict(
        dataset_version_id: str,
        sample_id: str,
        body: SubmitAnnotationLabelRequest,
    ) -> AnnotationDatasetVersion:
        return container.annotation_datasets.adjudicate(
            body.to_command(dataset_version_id, sample_id)
        )

    @router.post(
        "/annotation-datasets/{dataset_version_id}/freeze",
        response_model=AnnotationDatasetVersion,
        tags=["annotation-datasets"],
    )
    def freeze_annotation_dataset(
        dataset_version_id: str,
    ) -> AnnotationDatasetVersion:
        return container.annotation_datasets.freeze(dataset_version_id)

    @router.post(
        "/annotation-datasets/{dataset_version_id}/revisions",
        response_model=AnnotationDatasetVersion,
        status_code=status.HTTP_201_CREATED,
        tags=["annotation-datasets"],
        include_in_schema=False,
    )
    def create_annotation_dataset_revision(
        dataset_version_id: str,
        body: CreateAnnotationDatasetRevisionRequest,
    ) -> AnnotationDatasetVersion:
        return container.annotation_datasets.create_revision(
            body.to_command(dataset_version_id)
        )

    @router.post(
        "/a4/rule-versions/{version_id}/evaluation-runs",
        response_model=A4EvaluationRun,
        status_code=status.HTTP_201_CREATED,
        tags=["evaluation"],
    )
    def create_a4_evaluation_run(
        version_id: str,
        body: CreateEvaluationRunRequest,
    ) -> A4EvaluationRun:
        return container.evaluations.create(body.to_command(version_id))

    @router.get(
        "/a4/evaluation-runs",
        response_model=list[A4EvaluationRun],
        tags=["evaluation"],
    )
    def list_a4_evaluation_runs(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> tuple[A4EvaluationRun, ...]:
        return container.evaluations.list(limit)

    @router.get(
        "/a4/evaluation-runs/{run_id}",
        response_model=A4EvaluationRun,
        tags=["evaluation"],
    )
    def get_a4_evaluation_run(run_id: str) -> A4EvaluationRun:
        return container.evaluations.get(run_id)

    @router.get(
        "/a4/evaluation-runs/{run_id}/report",
        response_model=A4EvaluationReport,
        tags=["evaluation"],
    )
    def get_a4_evaluation_report(run_id: str) -> A4EvaluationReport:
        return container.evaluations.get_report(run_id)

    @router.post(
        "/rule-versions/{version_id}/publish",
        response_model=RuleVersion,
        tags=["rules"],
    )
    def publish_rule_version(
        version_id: str,
        body: PublishRuleVersionRequest,
        request: Request,
        call_id: str | None = Header(default=None, alias="X-Call-ID", max_length=128),
    ) -> RuleVersion:
        before = container.stage8.get_rule_version(version_id)
        request_id, resolved_call_id = _request_identity(request, call_id)
        try:
            gate = before.evaluation_gate
            if gate is None or gate.evaluation_run_id is None or gate.report_sha256 is None:
                raise PermanentError(
                    "persisted release gate is incomplete",
                    code="release_gate_persistence_invalid",
                )
            container.evaluations.assert_release_eligible(
                rule_version_id=before.rule_version_id,
                dataset_version_id=gate.dataset_version_id,
                evaluation_run_id=gate.evaluation_run_id,
                report_sha256=gate.report_sha256,
            )
            published = container.rule_versions.publish(body.to_command(version_id))
        except ServiceError:
            _audit_rule_mutation(
                container,
                version=before,
                actor_id=body.approver_id,
                action="rule_version.publish",
                request_id=request_id,
                call_id=resolved_call_id,
                result=AuditResult.REJECTED,
            )
            raise
        _audit_rule_mutation(
            container,
            version=published,
            actor_id=body.approver_id,
            action="rule_version.publish",
            request_id=request_id,
            call_id=resolved_call_id,
            result=AuditResult.SUCCEEDED,
            before_sha256=audit_sha256(before.model_dump(mode="json")),
        )
        return published

    @router.post(
        "/rule-sets/{rule_set_id}/rollback",
        response_model=RuleVersion,
        tags=["rules"],
    )
    def rollback_rule_set(
        rule_set_id: str,
        body: RollbackRuleSetRequest,
        request: Request,
        call_id: str | None = Header(default=None, alias="X-Call-ID", max_length=128),
    ) -> RuleVersion:
        target = container.stage8.get_rule_version(body.target_version_id)
        request_id, resolved_call_id = _request_identity(request, call_id)
        try:
            rolled_back = container.rule_versions.rollback(body.to_command(rule_set_id))
        except ServiceError:
            _audit_rule_mutation(
                container,
                version=target,
                actor_id=body.approver_id,
                action="rule_set.rollback",
                request_id=request_id,
                call_id=resolved_call_id,
                result=AuditResult.REJECTED,
            )
            raise
        _audit_rule_mutation(
            container,
            version=rolled_back,
            actor_id=body.approver_id,
            action="rule_set.rollback",
            request_id=request_id,
            call_id=resolved_call_id,
            result=AuditResult.SUCCEEDED,
        )
        return rolled_back

    @router.get(
        "/evaluation-runs/{run_id}",
        response_model=EvaluationRun,
        tags=["evaluation"],
    )
    def get_evaluation_run(run_id: str) -> EvaluationRun:
        return container.stage8.get_evaluation_run(run_id)

    @router.get(
        "/evaluation-runs/{run_id}/report",
        response_model=EvaluationReport,
        tags=["evaluation"],
    )
    def get_evaluation_report(run_id: str) -> EvaluationReport:
        return container.stage8.get_evaluation_report(run_id)

    @router.get(
        "/workbench",
        response_model=WorkbenchResourceIndex,
        tags=["workbench"],
    )
    def get_workbench_index() -> WorkbenchResourceIndex:
        return container.stage8.get_index()

    @router.get(
        "/audit-events", response_model=list[AuditEvent], tags=["audit"]
    )
    def list_audit_events(limit: int = 100) -> tuple[AuditEvent, ...]:
        return container.audit.list_events(limit)

    return router


def _normalized_header(value: str | None, *, default: str | None = None) -> str:
    normalized = (value or default or "").strip()
    if not normalized:
        raise PermanentError(
            "Required request header must not be blank",
            code="request_header_blank",
        )
    return normalized


def _run_check(check: object) -> CheckResult:
    name = str(getattr(check, "name", type(check).__name__))
    try:
        result = check.check()  # type: ignore[attr-defined]
        if not isinstance(result, ReadinessResult):
            raise TypeError("Readiness checks must return ReadinessResult")
        return CheckResult(name=result.name, ready=result.ready, detail=result.detail)
    except Exception as exc:
        logging.getLogger("tender_review.api").warning(
            "Readiness check failed",
            extra={
                "event": "health.readiness_failed",
                "check": name,
                "error": str(exc),
            },
        )
        return CheckResult(name=name, ready=False, detail="check failed")


def _request_identity(request: Request, call_id: str | None) -> tuple[str, str]:
    request_id = str(getattr(request.state, "request_id", "unknown"))
    normalized_call_id = (call_id or request_id).strip()
    return request_id, normalized_call_id


def _finding_source_type(source_kind: str) -> ReportSourceType:
    if source_kind == "verified_retrieval":
        return ReportSourceType.REAL
    return ReportSourceType.PROVISIONAL


def _rule_source_type(version: RuleVersion) -> ReportSourceType:
    if version.provenance.source_type == "synthetic":
        return ReportSourceType.SYNTHETIC
    if version.provenance.status == "verified":
        return ReportSourceType.REAL
    return ReportSourceType.PROVISIONAL


def _audit_rule_mutation(
    container: ApiContainer,
    *,
    version: RuleVersion,
    actor_id: str,
    action: str,
    request_id: str,
    call_id: str,
    result: AuditResult,
    before_sha256: str | None = None,
) -> None:
    container.audit.record(
        actor_kind=ActorKind.HUMAN,
        actor_id=actor_id,
        action=action,
        resource_type="rule_version",
        resource_id=version.rule_version_id,
        source_type=_rule_source_type(version),
        provenance_status=version.provenance.status,
        claims_allowed=version.provenance.claims_allowed,
        call_id=call_id,
        request_id=request_id,
        result=result,
        before_sha256=before_sha256
        or audit_sha256(version.model_dump(mode="json")),
        after_sha256=(
            audit_sha256(version.model_dump(mode="json"))
            if result is AuditResult.SUCCEEDED
            else None
        ),
        artifact_sha256s=(version.content_sha256,),
    )
