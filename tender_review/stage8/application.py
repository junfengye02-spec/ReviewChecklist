from __future__ import annotations

from tender_review.findings.public import Finding, FindingRepository, HumanDecision
from tender_review.rule_management.public import RuleVersion, RuleVersionRepository
from tender_review.shared.clock import Clock
from tender_review.shared.ids import IdGenerator

from .models import (
    ActorKind,
    AuditActor,
    AuditEvent,
    AuditProvenance,
    AuditResource,
    AuditResult,
    EvaluationReport,
    EvaluationRun,
    ReportSourceType,
    WorkbenchResourceIndex,
)
from .ports import AuditEventSink, EvaluationRunRepository, WorkbenchIndexRepository


class Stage8QueryService:
    def __init__(
        self,
        *,
        evaluations: EvaluationRunRepository,
        index: WorkbenchIndexRepository,
        findings: FindingRepository,
        rules: RuleVersionRepository,
    ) -> None:
        self._evaluations = evaluations
        self._index = index
        self._findings = findings
        self._rules = rules

    def get_index(self) -> WorkbenchResourceIndex:
        return self._index.get_index()

    def get_evaluation_run(self, run_id: str) -> EvaluationRun:
        return self._evaluations.get_run(run_id.strip())

    def get_evaluation_report(self, run_id: str) -> EvaluationReport:
        return self._evaluations.get_report(run_id.strip())

    def get_finding(self, finding_id: str) -> Finding:
        return self._findings.get_finding(finding_id.strip())

    def list_findings(self, review_job_id: str) -> tuple[Finding, ...]:
        return self._findings.list_findings(review_job_id.strip())

    def list_finding_decisions(self, finding_id: str) -> tuple[HumanDecision, ...]:
        return self._findings.list_decisions(finding_id.strip())

    def get_rule_version(self, version_id: str) -> RuleVersion:
        return self._rules.get_version(version_id.strip())


class AuditService:
    def __init__(self, sink: AuditEventSink, ids: IdGenerator, clock: Clock) -> None:
        self._sink = sink
        self._ids = ids
        self._clock = clock

    def record(
        self,
        *,
        actor_kind: ActorKind,
        actor_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        source_type: ReportSourceType,
        provenance_status: str,
        claims_allowed: bool,
        call_id: str,
        request_id: str,
        result: AuditResult,
        before_sha256: str | None = None,
        after_sha256: str | None = None,
        artifact_sha256s: tuple[str, ...] = (),
        job_id: str | None = None,
        thread_id: str | None = None,
        checkpoint_id: str | None = None,
        rule_version: str | None = None,
        dataset_version: str | None = None,
        model_config: str | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=self._ids.new(),
            actor=AuditActor(kind=actor_kind, actor_id=actor_id.strip()),
            action=action,
            resource=AuditResource(
                resource_type=resource_type, resource_id=resource_id
            ),
            before_sha256=before_sha256,
            after_sha256=after_sha256,
            provenance=AuditProvenance(
                source_type=source_type,
                status=provenance_status,
                claims_allowed=claims_allowed,
                artifact_sha256s=artifact_sha256s,
            ),
            call_id=call_id,
            request_id=request_id,
            job_id=job_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            rule_version=rule_version,
            dataset_version=dataset_version,
            model_configuration=model_config,
            occurred_at=self._clock.now(),
            result=result,
        )
        return self._sink.append(event)

    def list_events(self, limit: int = 100) -> tuple[AuditEvent, ...]:
        return self._sink.list_events(max(1, min(limit, 500)))
