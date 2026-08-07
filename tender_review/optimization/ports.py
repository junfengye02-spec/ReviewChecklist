from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    CreateOptimizationJob,
    JointRegressionResult,
    OptimizationAttempt,
    OptimizationCandidate,
    OptimizationJob,
    RootCauseDecision,
    OptimizationReadiness,
)


@runtime_checkable
class OptimizationRepository(Protocol):
    def create_job(self, job: OptimizationJob) -> OptimizationJob: ...

    def get_job(self, optimization_job_id: str) -> OptimizationJob: ...

    def save_job(self, job: OptimizationJob) -> OptimizationJob: ...

    def get_attempt(
        self, optimization_job_id: str, attempt_number: int
    ) -> OptimizationAttempt | None: ...

    def save_attempt(self, attempt: OptimizationAttempt) -> OptimizationAttempt: ...

    def list_attempts(
        self, optimization_job_id: str
    ) -> tuple[OptimizationAttempt, ...]: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    def generate(
        self,
        job: OptimizationJob,
        attempt_number: int,
        root_cause: RootCauseDecision,
        limit: int,
    ) -> tuple[OptimizationCandidate, ...]: ...


@runtime_checkable
class RegressionEvaluator(Protocol):
    def evaluate(
        self,
        job: OptimizationJob,
        candidate: OptimizationCandidate,
    ) -> JointRegressionResult: ...


@runtime_checkable
class CandidateRuleStager(Protocol):
    def stage_candidate(
        self,
        job: OptimizationJob,
        attempt: OptimizationAttempt,
        candidate: OptimizationCandidate,
    ) -> str: ...


@runtime_checkable
class OptimizationReadinessVerifier(Protocol):
    def assess(self, command: CreateOptimizationJob) -> OptimizationReadiness: ...


@runtime_checkable
class RootCauseClassifier(Protocol):
    def analyze(
        self, job: OptimizationJob, attempt_number: int
    ) -> RootCauseDecision: ...
