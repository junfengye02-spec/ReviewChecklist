from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tender_review.findings.public import (
    DocumentIdentity,
    EvidenceReference,
    FindingProvenance,
    FindingRepository,
    FindingWorkflowState,
    build_finding,
)
from tender_review.jobs.public import (
    CheckpointState,
    CheckpointValue,
    JobCheckpoint,
    JobLifecycle,
    ReviewJob,
    ReviewJobRepository,
    ReviewStage,
)
from tender_review.optimization.public import OptimizationRepository
from tender_review.rule_management.public import RuleVersionRepository
from tender_review.shared.clock import Clock
from tender_review.shared.config import AppSettings
from tender_review.shared.ids import IdGenerator

from .application import AuditService, Stage8QueryService
from .models import (
    ActorKind,
    AuditActor,
    AuditEvent,
    AuditProvenance,
    AuditResource,
    AuditResult,
    EvaluationReport,
    EvaluationRun,
    EvaluationRunHashes,
    MetricStatus,
    ReportMetric,
    ReportSection,
    ReportSourceType,
    RunStatus,
    WorkbenchResourceIndex,
    stable_sha256,
)
from .repositories import (
    InMemoryAuditEventSink,
    LoggingAuditEventSink,
    StaticEvaluationRunRepository,
    StaticWorkbenchIndexRepository,
)


DEMO_TIME = datetime(2026, 7, 28, 8, 0, tzinfo=timezone.utc)
DEMO_RUN_ID = "synthetic-demo-run-v1"
DEMO_REVIEW_JOB_ID = "demo-review-job-1"
DEMO_FINDING_ID = "demo-finding-1"
DEMO_REQUIRED_CASES = 4


@dataclass(frozen=True)
class Stage8Assembly:
    queries: Stage8QueryService
    audit: AuditService


def assemble_stage8(
    *,
    settings: AppSettings,
    ids: IdGenerator,
    clock: Clock,
    review_jobs: ReviewJobRepository,
    findings: FindingRepository,
    rules: RuleVersionRepository,
    optimizations: OptimizationRepository,
) -> Stage8Assembly:
    if settings.workbench_demo_enabled:
        if settings.adapter_mode != "fake" or settings.environment.lower() in {
            "prod",
            "production",
            "staging",
        }:
            raise ValueError("Stage 8 demo data is restricted to local fake assembly")
        run, report = _evaluation_artifacts()
        _seed_review_job(review_jobs)
        _seed_finding(findings, run)
        index = WorkbenchResourceIndex(
            demo_mode=True,
            environment=settings.environment,
            source_type=ReportSourceType.SYNTHETIC,
            status="provisional",
            claims_allowed=False,
            human_annotation_cases=0,
            required_human_cases=DEMO_REQUIRED_CASES,
            review_job_ids=(DEMO_REVIEW_JOB_ID,),
            finding_ids=(DEMO_FINDING_ID,),
            rule_set_ids=(),
            optimization_job_ids=(),
            evaluation_run_ids=(DEMO_RUN_ID,),
            generated_at=DEMO_TIME,
        )
        initial_audit = (_demo_loaded_event(report.report_sha256),)
        evaluation_repository = StaticEvaluationRunRepository((run,), (report,))
        audit_sink = InMemoryAuditEventSink(initial_audit)
    else:
        index = WorkbenchResourceIndex(
            demo_mode=False,
            environment=settings.environment,
            source_type=ReportSourceType.REAL,
            status="unknown",
            claims_allowed=False,
            human_annotation_cases=0,
            required_human_cases=0,
            generated_at=clock.now(),
        )
        evaluation_repository = StaticEvaluationRunRepository()
        audit_sink = LoggingAuditEventSink()

    queries = Stage8QueryService(
        evaluations=evaluation_repository,
        index=StaticWorkbenchIndexRepository(index),
        findings=findings,
        rules=rules,
    )
    return Stage8Assembly(
        queries=queries,
        audit=AuditService(audit_sink, ids, clock),
    )


def _evaluation_artifacts() -> tuple[EvaluationRun, EvaluationReport]:
    source_report = {
        "human_annotation_cases": 0,
        "required_human_cases": DEMO_REQUIRED_CASES,
        "variants": ("bm25", "vector", "hybrid_rrf"),
        "dataset_version_id": "synthetic-demo-dataset-v1",
        "source_work_package_sha256": stable_sha256("synthetic-work-package"),
        "input_sha256": stable_sha256("synthetic-input"),
        "shared_config_sha256": stable_sha256("synthetic-config"),
        "implementation_version_sha256": stable_sha256("synthetic-code"),
    }
    manifest = {
        "variant_results": {
            "bm25": "synthetic-bm25-result",
            "vector": "synthetic-vector-result",
            "hybrid_rrf": "synthetic-hybrid-result",
        }
    }
    report_payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": DEMO_RUN_ID,
        "source_type": "synthetic",
        "status": "provisional",
        "claims_allowed": False,
        "human_annotation_cases": source_report["human_annotation_cases"],
        "required_human_cases": source_report["required_human_cases"],
        "sections": [
            _section(
                "conclusion",
                "结论指标",
                (
                    _metric(
                        "human-review-coverage",
                        "人工标注与独立复核",
                        f"0/{DEMO_REQUIRED_CASES}",
                        "cases",
                        "synthetic",
                        "provisional",
                        "当前没有真实人工 chunk 级真值，不能声明准确率。",
                    ),
                    _unknown_metric(
                        "production-accuracy",
                        "生产准确率",
                        "没有独立人工金标，未计算。",
                    ),
                ),
            ),
            _section(
                "evidence",
                "证据指标",
                (
                    _metric(
                        "candidate-cases",
                        "候选导航用例",
                        source_report["required_human_cases"],
                        "cases",
                        "synthetic",
                        "provisional",
                        "候选提示为合成演示数据，只用于导航诊断。",
                    ),
                    _unknown_metric(
                        "recall-at-10",
                        "Recall@10",
                        "没有人工相关性标签，未计算。",
                    ),
                    _unknown_metric(
                        "mrr",
                        "MRR",
                        "没有人工相关性标签，未计算。",
                    ),
                ),
            ),
            _section(
                "engineering",
                "工程指标",
                (
                    _metric(
                        "retrieval-variants",
                        "同输入检索方案",
                        len(source_report["variants"]),
                        "variants",
                        "provisional",
                        "provisional",
                        "BM25、Vector-only、Hybrid/RRF 使用同一输入和共享配置。",
                    ),
                    _metric(
                        "optimization-traces",
                        "可恢复优化轨迹",
                        0,
                        "traces",
                        "synthetic",
                        "provisional",
                        "合成演示不包含优化轨迹。",
                    ),
                ),
            ),
            _section(
                "cost",
                "成本指标",
                (
                    _unknown_metric(
                        "model-cost",
                        "模型成本",
                        "合成演示未采集 token 或账单数据。",
                        source_type="synthetic",
                    ),
                    _unknown_metric(
                        "production-latency",
                        "生产延迟",
                        "仅有本地运行观察，不能解释为生产延迟，因此未采集。",
                        source_type="synthetic",
                    ),
                ),
            ),
        ],
        "limitations": [
            f"人工标注与独立复核为 0/{DEMO_REQUIRED_CASES}。",
            "本报告仅使用合成数据，不是真实业务真值。",
            "本报告不能用于声明 Recall、MRR、生产准确率、生产延迟或人工批准。",
            "合成演示不包含任何规则发布结果。",
        ],
        "generated_at": DEMO_TIME,
    }
    report = EvaluationReport(
        **report_payload,
        report_sha256=stable_sha256(report_payload),
    )
    result_hash = stable_sha256(manifest["variant_results"])
    run = EvaluationRun(
        run_id=DEMO_RUN_ID,
        name="合成检索演示",
        source_type=ReportSourceType.SYNTHETIC,
        status=RunStatus.COMPLETED,
        provenance_status="provisional",
        claims_allowed=False,
        dataset_version_id=str(source_report["dataset_version_id"]),
        input_artifact_id="synthetic://stage8/input",
        results_artifact_id="synthetic://stage8/results",
        config_artifact_id="synthetic://stage8/config",
        code_version_id="synthetic-demo-code-v1",
        hashes=EvaluationRunHashes(
            dataset_sha256=str(source_report["source_work_package_sha256"]),
            input_sha256=str(source_report["input_sha256"]),
            results_sha256=result_hash,
            config_sha256=str(source_report["shared_config_sha256"]),
            code_sha256=str(source_report["implementation_version_sha256"]),
        ),
        call_id="synthetic-local-demo",
        request_id="synthetic-demo-assembly",
        started_at=DEMO_TIME,
        completed_at=DEMO_TIME,
        report_sha256=report.report_sha256,
    )
    return run, report


def _section(
    section_id: str, title: str, metrics: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    return ReportSection(
        section_id=section_id, title=title, metrics=metrics
    ).model_dump(mode="json")


def _metric(
    metric_id: str,
    label: str,
    value: int | float | str,
    unit: str,
    source_type: str,
    status: str,
    interpretation: str,
) -> dict[str, Any]:
    return ReportMetric(
        metric_id=metric_id,
        label=label,
        value=value,
        unit=unit,
        source_type=source_type,
        status=status,
        claims_allowed=False,
        collected=True,
        interpretation=interpretation,
    ).model_dump(mode="json")


def _unknown_metric(
    metric_id: str,
    label: str,
    interpretation: str,
    *,
    source_type: str = "provisional",
) -> dict[str, Any]:
    return ReportMetric(
        metric_id=metric_id,
        label=label,
        source_type=source_type,
        status=MetricStatus.UNKNOWN,
        claims_allowed=False,
        collected=False,
        interpretation=interpretation,
    ).model_dump(mode="json")


def _seed_review_job(repository: ReviewJobRepository) -> None:
    input_fingerprint = stable_sha256(
        {
            "document": "demo-evidence-document",
            "rule": "synthetic-demo-rule-v1",
            "config": "synthetic-demo-config-v1",
        }
    )
    job = ReviewJob(
        id=DEMO_REVIEW_JOB_ID,
        document_snapshot_id="demo-evidence-document",
        rule_version_id="synthetic-demo-rule-v1",
        model_config_id="synthetic-local-demo",
        input_fingerprint=input_fingerprint,
        status=JobLifecycle.WAITING_HUMAN,
        stage=ReviewStage.REPORTING,
        attempt_count=1,
        max_attempts=3,
        available_at=DEMO_TIME,
        lease_token=2,
        created_at=DEMO_TIME,
        updated_at=DEMO_TIME,
    )
    repository.create_review_job(job)
    for sequence, stage in enumerate(ReviewStage, start=1):
        repository.save_checkpoint(
            JobCheckpoint(
                job_id=job.id,
                node_name=stage.value.lower(),
                stage=stage,
                lease_token=1,
                sequence=sequence,
                state=CheckpointState(
                    values=(
                        CheckpointValue(key="status", value="completed"),
                        CheckpointValue(
                            key="provenance_status", value="provisional"
                        ),
                    )
                ),
                output_artifact_id=f"demo-{stage.value.lower()}-artifact",
                completed_at=DEMO_TIME,
            )
        )


def _seed_finding(repository: FindingRepository, run: EvaluationRun) -> None:
    excerpt = (
        "The synthetic tender requires a signed authorization letter and "
        "two project references for the proposed project manager."
    )
    document_id = "synthetic-tender-demo.pdf"
    document_sha256 = hashlib.sha256(
        b"synthetic tender demo document"
    ).hexdigest()
    evidence = EvidenceReference(
        document_id=document_id,
        chunk_id="synthetic-evidence-1",
        page_number=1,
        section_path=("Evaluation Criteria", "Project Team"),
        excerpt=excerpt,
        text_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    )
    finding = build_finding(
        finding_id=DEMO_FINDING_ID,
        review_job_id=DEMO_REVIEW_JOB_ID,
        rule_version_id="synthetic-demo-rule-v1",
        review_item_id="authorization-letter",
        workflow_state=FindingWorkflowState.DONE,
        message=(
            "Synthetic evidence indicates a possible rule-coverage gap. "
            "It is non-claimable and requires human review."
        ),
        documents=(
            DocumentIdentity(
                document_id=document_id, document_sha256=document_sha256
            ),
        ),
        provenance=FindingProvenance(
            source_kind="provisional_retrieval",
            status="provisional",
            claims_allowed=False,
            dataset_version_id=run.dataset_version_id,
            review_input_sha256=run.hashes.input_sha256,
            retrieval_results_sha256=run.hashes.results_sha256,
            retrieval_variant="bm25-candidate",
        ),
        created_at=DEMO_TIME,
        conclusion="noncompliant",
        evidence=(evidence,),
        human_approval_allowed=False,
    )
    repository.add_finding(finding)


def _demo_loaded_event(report_sha256: str) -> AuditEvent:
    return AuditEvent(
        event_id="demo-audit-1",
        actor=AuditActor(kind=ActorKind.SYSTEM, actor_id="local-demo-assembler"),
        action="workbench.demo.loaded",
        resource=AuditResource(
            resource_type="evaluation_run", resource_id=DEMO_RUN_ID
        ),
        after_sha256=report_sha256,
        provenance=AuditProvenance(
            source_type=ReportSourceType.SYNTHETIC,
            status="provisional",
            claims_allowed=False,
            artifact_sha256s=(report_sha256,),
        ),
        call_id="synthetic-demo-assembly",
        request_id="synthetic-demo-assembly",
        occurred_at=DEMO_TIME,
        result=AuditResult.SUCCEEDED,
    )
