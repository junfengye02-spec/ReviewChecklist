from __future__ import annotations

from tender_review.shared.errors import PermanentError

from .models import (
    JointRegressionResult,
    OptimizationCandidate,
    OptimizationJob,
    RootCauseDecision,
)


class UnavailableCandidateGenerator:
    def generate(
        self,
        job: OptimizationJob,
        attempt_number: int,
        root_cause: RootCauseDecision,
        limit: int,
    ) -> tuple[OptimizationCandidate, ...]:
        del job, attempt_number, root_cause, limit
        raise PermanentError(
            "production candidate generation is not configured",
            code="optimization_candidate_generator_unavailable",
        )


class UnavailableRegressionEvaluator:
    def evaluate(
        self,
        job: OptimizationJob,
        candidate: OptimizationCandidate,
    ) -> JointRegressionResult:
        del job, candidate
        raise PermanentError(
            "production joint regression is not configured",
            code="optimization_regression_evaluator_unavailable",
        )
