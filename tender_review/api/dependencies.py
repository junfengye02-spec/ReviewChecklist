from __future__ import annotations

from typing import Protocol

from tender_review.jobs.public import ReviewJobService
from tender_review.documents.public import DocumentService
from tender_review.shared.config import AppSettings
from tender_review.shared.health import ReadinessCheck
from tender_review.shared.ids import IdGenerator
from tender_review.findings.public import FindingDecisionService
from tender_review.rule_management.public import RuleVersionService
from tender_review.optimization.public import OptimizationService
from tender_review.evaluation.public import EvaluationRunService
from tender_review.stage8.public import AuditService, Stage8QueryService


class ApiContainer(Protocol):
    """The narrow dependency surface used by the HTTP adapter."""

    settings: AppSettings
    ids: IdGenerator
    review_jobs: ReviewJobService
    documents: DocumentService
    finding_decisions: FindingDecisionService
    rule_versions: RuleVersionService
    optimizations: OptimizationService
    evaluations: EvaluationRunService
    stage8: Stage8QueryService
    audit: AuditService
    readiness_checks: tuple[ReadinessCheck, ...]
