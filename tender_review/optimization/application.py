from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import ValidationError

from tender_review.evaluation.public import DatasetVersionRepository
from tender_review.review.public import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from tender_review.rule_management.public import RuleVersionRepository
from tender_review.shared.clock import Clock
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import (
    ConflictError,
    PermanentError,
    RetryableError,
    ServiceError,
)
from tender_review.shared.ids import IdGenerator

from .domain import deterministic_root_cause, transition_job
from .models import (
    CreateOptimizationJob,
    ExecutionHashes,
    OptimizationAttempt,
    OptimizationJob,
    OptimizationStatus,
    OptimizationReadinessStatus,
    RootCauseDecision,
    RootCauseLlmOutput,
    SampleRole,
    TERMINAL_OPTIMIZATION_STATUSES,
    stable_sha256,
)
from .ports import (
    CandidateGenerator,
    CandidateRuleStager,
    OptimizationRepository,
    RegressionEvaluator,
    OptimizationReadinessVerifier,
)
from .readiness import UnavailableOptimizationReadinessVerifier


class RootCauseAnalyzer:
    def __init__(
        self,
        llm: LlmProvider | None = None,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._llm = llm
        self._timeout_seconds = timeout_seconds

    def analyze(self, job: OptimizationJob, attempt_number: int) -> RootCauseDecision:
        deterministic = deterministic_root_cause(job)
        if deterministic is not None:
            return deterministic
        if self._llm is None:
            raise PermanentError(
                "semantic root-cause classification requires an LLM provider",
                code="optimization_root_cause_llm_unavailable",
            )
        call_id = f"{job.optimization_job_id}:root-cause:{attempt_number}"[:128]
        request = LlmRequest(
            messages=(
                LlmMessage(
                    role="system",
                    content=(
                        "Classify only the unresolved semantic failure. Return strict JSON "
                        "with root_cause and rationale; do not propose a candidate."
                    ),
                ),
                LlmMessage(
                    role="user",
                    content=json.dumps(
                        [
                            {
                                "sample_id": item.sample_id,
                                "failure_summary": item.signals.failure_summary,
                                "known_signals": item.signals.model_dump(
                                    mode="json", exclude_none=True
                                ),
                            }
                            for item in job.samples
                            if item.role is SampleRole.TARGET and item.signals is not None
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ),
            response_schema_name="RootCauseLlmOutput.v1",
            temperature=0,
            call=CallContext(
                call_id=call_id,
                timeout_seconds=self._timeout_seconds,
                max_attempts=1,
            ),
        )
        response = _complete_with_timeout(self._llm, request)
        try:
            validated_response = LlmResponse.model_validate(response)
            output = RootCauseLlmOutput.model_validate_json(validated_response.content)
        except (ValidationError, ValueError, TypeError) as exc:
            raise PermanentError(
                "root-cause LLM returned invalid strict-schema output",
                code="optimization_root_cause_invalid_output",
                details={"call_id": call_id},
            ) from exc
        return RootCauseDecision(
            root_cause=output.root_cause,
            classifier="llm",
            rationale=output.rationale,
            target_sample_ids=tuple(
                item.sample_id for item in job.samples if item.role is SampleRole.TARGET
            ),
            call_id=call_id,
        )


def _complete_with_timeout(llm: LlmProvider, request: LlmRequest) -> LlmResponse:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optimization-llm")
    future = executor.submit(llm.complete, request)
    try:
        value = future.result(timeout=request.call.timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise RetryableError(
            "root-cause LLM call timed out",
            code="optimization_root_cause_timeout",
            details={"call_id": request.call.call_id},
        ) from exc
    except BaseException:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return value


class OptimizationService:
    def __init__(
        self,
        *,
        repository: OptimizationRepository,
        rule_versions: RuleVersionRepository,
        datasets: DatasetVersionRepository,
        ids: IdGenerator,
        clock: Clock,
        root_causes: RootCauseAnalyzer,
        candidates: CandidateGenerator,
        evaluator: RegressionEvaluator,
        stager: CandidateRuleStager,
        readiness: OptimizationReadinessVerifier | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self._repository = repository
        self._rule_versions = rule_versions
        self._datasets = datasets
        self._ids = ids
        self._clock = clock
        self._root_causes = root_causes
        self._candidates = candidates
        self._evaluator = evaluator
        self._stager = stager
        self._readiness = readiness or UnavailableOptimizationReadinessVerifier(clock)
        from .langgraph_workflow import LangGraphOptimizationWorkflow

        self._workflow = LangGraphOptimizationWorkflow(
            repository=repository,
            rule_versions=rule_versions,
            ids=ids,
            clock=clock,
            root_causes=root_causes,
            candidates=candidates,
            evaluator=evaluator,
            stager=stager,
            checkpointer=checkpointer,
        )

    def create(self, command: CreateOptimizationJob) -> OptimizationJob:
        base = self._rule_versions.get_version(command.base_rule_version_id)
        readiness = self._readiness.assess(command)
        dataset = None
        try:
            dataset = self._datasets.get_version(command.dataset_version_id)
        except ServiceError:
            pass
        if dataset is not None:
            dataset_sample_ids = {item.sample_id for item in dataset.samples}
            requested_sample_ids = {item.sample_id for item in command.samples}
            if not requested_sample_ids.issubset(dataset_sample_ids):
                raise PermanentError(
                    "optimization samples must belong to the selected dataset version",
                    code="optimization_sample_dataset_mismatch",
                )
        dataset_sha256 = (
            readiness.dataset_manifest_sha256
            or (dataset.manifest_sha256 if dataset is not None else None)
            or stable_sha256(
                {
                    "dataset_version_id": command.dataset_version_id,
                    "status": readiness.status,
                    "sample_ids": tuple(sorted(item.sample_id for item in command.samples)),
                }
            )
        )
        effective_provenance = command.provenance.model_copy(
            update={"claims_allowed": readiness.claims_allowed}
        )
        stable_input = command.model_dump(mode="json", exclude={"model_sha256", "prompt_sha256", "retriever_sha256", "tool_sha256"})
        hashes = ExecutionHashes(
            input_sha256=stable_sha256(stable_input),
            rule_sha256=base.content_sha256,
            dataset_sha256=dataset_sha256,
            model_sha256=command.model_sha256,
            prompt_sha256=command.prompt_sha256,
            retriever_sha256=command.retriever_sha256,
            tool_sha256=command.tool_sha256,
        )
        now = self._clock.now()
        initial_status = {
            OptimizationReadinessStatus.READY: OptimizationStatus.PENDING,
            OptimizationReadinessStatus.NOT_READY: OptimizationStatus.NOT_READY,
            OptimizationReadinessStatus.BLOCKED: OptimizationStatus.BLOCKED,
        }[readiness.status]
        return self._repository.create_job(
            OptimizationJob(
                optimization_job_id=self._ids.new(),
                base_rule_version_id=base.rule_version_id,
                dataset_version_id=dataset.dataset_version_id,
                status=initial_status,
                max_rounds=command.max_rounds,
                candidates_per_round=command.candidates_per_round,
                required_stability_runs=command.required_stability_runs,
                current_round=0,
                samples=command.samples,
                hashes=hashes,
                provenance=effective_provenance,
                readiness=readiness,
                created_at=now,
                updated_at=now,
                completed_at=(
                    now
                    if initial_status
                    in {OptimizationStatus.NOT_READY, OptimizationStatus.BLOCKED}
                    else None
                ),
            )
        )

    def get(self, optimization_job_id: str) -> OptimizationJob:
        return self._repository.get_job(optimization_job_id)

    def list_attempts(
        self, optimization_job_id: str
    ) -> tuple[OptimizationAttempt, ...]:
        return self._repository.list_attempts(optimization_job_id)

    def cancel(self, optimization_job_id: str) -> OptimizationJob:
        job = self.get(optimization_job_id)
        if job.status is OptimizationStatus.CANCELLED:
            return job
        if job.status in TERMINAL_OPTIMIZATION_STATUSES:
            raise ConflictError(
                "terminal optimization job cannot be cancelled",
                code="optimization_cancel_invalid",
            )
        return self._repository.save_job(
            transition_job(
                job,
                OptimizationStatus.CANCELLED,
                now=self._clock.now(),
            )
        )

    @property
    def compiled_graph(self):
        return self._workflow.compiled_graph

    def run(
        self,
        optimization_job_id: str,
        *,
        interrupt_after: Sequence[str] = (),
    ) -> OptimizationJob:
        return self._workflow.run(
            optimization_job_id,
            interrupt_after=interrupt_after,
        )
