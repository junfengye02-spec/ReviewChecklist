from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from threading import RLock

from tender_review.shared.errors import ConflictError, NotFoundError

from .models import (
    ROOT_CAUSE_CANDIDATE_TYPES,
    CandidateChange,
    CandidateProvenance,
    CandidateType,
    JointRegressionResult,
    OptimizationAttempt,
    OptimizationCandidate,
    OptimizationJob,
    RegressionCaseOutcome,
    RegressionStatus,
    RootCause,
    RootCauseDecision,
    SampleRole,
    canonical_json,
    stable_sha256,
)


class InMemoryOptimizationRepository:
    def __init__(self) -> None:
        self._jobs: dict[str, OptimizationJob] = {}
        self._attempts: dict[tuple[str, int], OptimizationAttempt] = {}
        self._lock = RLock()

    def create_job(self, job: OptimizationJob) -> OptimizationJob:
        with self._lock:
            if job.optimization_job_id in self._jobs:
                raise ConflictError(
                    "optimization job already exists", code="optimization_job_conflict"
                )
            self._jobs[job.optimization_job_id] = job
            return job

    def get_job(self, optimization_job_id: str) -> OptimizationJob:
        with self._lock:
            try:
                return self._jobs[optimization_job_id]
            except KeyError as exc:
                raise NotFoundError(
                    "optimization job does not exist",
                    code="optimization_job_not_found",
                ) from exc

    def save_job(self, job: OptimizationJob) -> OptimizationJob:
        with self._lock:
            current = self.get_job(job.optimization_job_id)
            if current.base_rule_version_id != job.base_rule_version_id:
                raise ConflictError(
                    "optimization input identity is immutable",
                    code="optimization_job_input_changed",
                )
            self._jobs[job.optimization_job_id] = job
            return job

    def get_attempt(
        self, optimization_job_id: str, attempt_number: int
    ) -> OptimizationAttempt | None:
        with self._lock:
            self.get_job(optimization_job_id)
            return self._attempts.get((optimization_job_id, attempt_number))

    def save_attempt(self, attempt: OptimizationAttempt) -> OptimizationAttempt:
        with self._lock:
            self.get_job(attempt.optimization_job_id)
            key = (attempt.optimization_job_id, attempt.attempt_number)
            current = self._attempts.get(key)
            if current is not None:
                _require_monotonic_attempt(current, attempt)
            self._attempts[key] = attempt
            return attempt

    def list_attempts(
        self, optimization_job_id: str
    ) -> tuple[OptimizationAttempt, ...]:
        with self._lock:
            self.get_job(optimization_job_id)
            return tuple(
                value
                for (job_id, _), value in sorted(self._attempts.items())
                if job_id == optimization_job_id
            )


def _require_monotonic_attempt(
    current: OptimizationAttempt, candidate: OptimizationAttempt
) -> None:
    if current.attempt_id != candidate.attempt_id:
        raise ConflictError(
            "attempt identity is immutable", code="optimization_attempt_changed"
        )
    if current.root_cause is not None and candidate.root_cause != current.root_cause:
        raise ConflictError(
            "root cause checkpoint is immutable", code="optimization_root_cause_changed"
        )
    if current.candidates and candidate.candidates != current.candidates:
        raise ConflictError(
            "candidate checkpoint is immutable", code="optimization_candidates_changed"
        )
    if candidate.evaluations[: len(current.evaluations)] != current.evaluations:
        raise ConflictError(
            "completed evaluations are immutable",
            code="optimization_evaluation_changed",
        )
    if current.candidate_rule_version_id is not None and (
        candidate.candidate_rule_version_id != current.candidate_rule_version_id
    ):
        raise ConflictError(
            "staged rule candidate is immutable",
            code="optimization_rule_candidate_changed",
        )


class FakeCandidateGenerator:
    def __init__(
        self,
        batches: Iterable[tuple[OptimizationCandidate, ...] | BaseException] = (),
        *,
        base_content_json: str = "{}",
        base_execution_config_json: str = "{}",
    ) -> None:
        self._batches = deque(batches)
        self.base_content_json = base_content_json
        self.base_execution_config_json = base_execution_config_json
        self.calls: list[tuple[str, int, RootCause, int]] = []

    def generate(
        self,
        job: OptimizationJob,
        attempt_number: int,
        root_cause: RootCauseDecision,
        limit: int,
    ) -> tuple[OptimizationCandidate, ...]:
        self.calls.append(
            (job.optimization_job_id, attempt_number, root_cause.root_cause, limit)
        )
        if self._batches:
            value = self._batches.popleft()
            if isinstance(value, BaseException):
                raise value
            return value[:limit]
        return tuple(
            self._candidate(job, attempt_number, root_cause, index)
            for index in range(1, limit + 1)
        )

    def _candidate(
        self,
        job: OptimizationJob,
        attempt_number: int,
        decision: RootCauseDecision,
        index: int,
    ) -> OptimizationCandidate:
        candidate_type = ROOT_CAUSE_CANDIDATE_TYPES[decision.root_cause]
        if candidate_type is None:
            raise ValueError("LABEL_UNCERTAIN cannot generate an automatic candidate")
        content = json.loads(self.base_content_json)
        execution = json.loads(self.base_execution_config_json)
        scope, path, before, after = _bounded_fake_change(
            candidate_type, content, execution, attempt_number, index
        )
        candidate_id = "candidate-" + stable_sha256(
            {
                "job": job.optimization_job_id,
                "attempt": attempt_number,
                "index": index,
                "type": candidate_type,
            }
        )[:20]
        protection_ids = tuple(
            item.sample_id for item in job.samples if item.role is SampleRole.PROTECTION
        )
        return OptimizationCandidate(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            root_cause=decision.root_cause,
            content_json=canonical_json(content),
            execution_config_json=canonical_json(execution),
            change=CandidateChange(
                scope=scope,
                path=path,
                before_json=canonical_json(before) if before is not None else None,
                after_json=canonical_json(after),
            ),
            rationale=f"Bounded {candidate_type.value} candidate for {decision.root_cause.value}.",
            target_sample_ids=decision.target_sample_ids,
            affected_protection_sample_ids=protection_ids,
            provenance=CandidateProvenance(
                optimization_job_id=job.optimization_job_id,
                attempt_number=attempt_number,
                base_rule_version_id=job.base_rule_version_id,
                dataset_version_id=job.dataset_version_id,
                hashes=job.hashes,
                source_type=job.provenance.source_type,
                status=job.provenance.status,
                claims_allowed=job.provenance.claims_allowed,
                source_artifact_sha256s=tuple(
                    item.sha256 for item in job.provenance.source_artifacts
                ),
            ),
        )


def _bounded_fake_change(
    candidate_type: CandidateType,
    content: dict[str, object],
    execution: dict[str, object],
    attempt: int,
    index: int,
) -> tuple[str, str, object | None, object]:
    value = f"bounded-{attempt}-{index}"
    if candidate_type is CandidateType.RULE_CONTENT:
        before = content.get("rule_text")
        content["rule_text"] = value
        return "content", "$.rule_text", before, value
    if candidate_type is CandidateType.RETRIEVAL_CONFIG:
        retrieval = execution.setdefault("retrieval", {})
        if not isinstance(retrieval, dict):
            raise ValueError("fake base retrieval config must be an object")
        query = retrieval.setdefault("query", {})
        if not isinstance(query, dict):
            raise ValueError("fake base query config must be an object")
        before = query.get("expansion")
        query["expansion"] = value
        return "execution_config", "$.retrieval.query.expansion", before, value
    if candidate_type is CandidateType.EXTRACTION_PROMPT_SCHEMA:
        extraction = execution.setdefault("extraction", {})
        if not isinstance(extraction, dict):
            raise ValueError("fake base extraction config must be an object")
        before = extraction.get("prompt")
        extraction["prompt"] = value
        return "execution_config", "$.extraction.prompt", before, value
    if candidate_type is CandidateType.TOOL_CONFIG:
        tools = execution.setdefault("tools", {})
        if not isinstance(tools, dict):
            raise ValueError("fake base tools config must be an object")
        before = tools.get("version")
        tools["version"] = value
        return "execution_config", "$.tools.version", before, value
    model = execution.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("fake base model config must be an object")
    before = model.get("seed")
    after = attempt * 100 + index
    model["seed"] = after
    return "execution_config", "$.model.seed", before, after


class FakeRegressionEvaluator:
    def __init__(
        self,
        plans: Iterable[tuple[bool, bool, bool] | BaseException] = (),
    ) -> None:
        self._plans = deque(plans)
        self.calls: list[str] = []

    def evaluate(
        self,
        job: OptimizationJob,
        candidate: OptimizationCandidate,
    ) -> JointRegressionResult:
        self.calls.append(candidate.candidate_id)
        value = self._plans.popleft() if self._plans else (False, True, True)
        if isinstance(value, BaseException):
            raise value
        target_passed, protection_passed, stable = value
        outcomes: list[RegressionCaseOutcome] = []
        for sample in job.samples:
            sample_passed = (
                target_passed
                if sample.role is SampleRole.TARGET
                else protection_passed
            )
            stable_hash = stable_sha256(
                {"candidate": candidate.candidate_id, "sample": sample.sample_id}
            )
            for run_number in range(1, job.required_stability_runs + 1):
                result_hash = (
                    stable_hash
                    if stable
                    else stable_sha256(
                        {
                            "candidate": candidate.candidate_id,
                            "sample": sample.sample_id,
                            "run": run_number,
                        }
                    )
                )
                outcomes.append(
                    RegressionCaseOutcome(
                        sample_id=sample.sample_id,
                        role=sample.role,
                        run_number=run_number,
                        passed=sample_passed,
                        result_sha256=result_hash,
                    )
                )
        provisional = job.provenance.status == "provisional"
        all_gates = target_passed and protection_passed and stable
        status = (
            RegressionStatus.PROVISIONAL
            if provisional and all_gates
            else RegressionStatus.PASSED
            if all_gates
            else RegressionStatus.FAILED
        )
        payload = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "required_stability_runs": job.required_stability_runs,
            "outcomes": [item.model_dump(mode="json") for item in outcomes],
            "target_gate_passed": target_passed,
            "protection_gate_passed": protection_passed,
            "stability_gate_passed": stable,
            "status": status,
            "provisional": provisional,
            "claims_allowed": job.provenance.claims_allowed if all_gates else False,
        }
        return JointRegressionResult(**payload, report_sha256=stable_sha256(payload))


class FakeCandidateRuleStager:
    def __init__(self) -> None:
        self._versions: dict[tuple[str, int, str], str] = {}
        self.calls: list[tuple[str, int, str]] = []

    def stage_candidate(
        self,
        job: OptimizationJob,
        attempt: OptimizationAttempt,
        candidate: OptimizationCandidate,
    ) -> str:
        key = (
            job.optimization_job_id,
            attempt.attempt_number,
            candidate.candidate_id,
        )
        self.calls.append(key)
        return self._versions.setdefault(
            key,
            "rule-candidate-" + stable_sha256(key)[:20],
        )
