from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tender_review.rule_management.public import (
    RuleVersionRepository,
    RuleVersionStatus,
)
from tender_review.shared.clock import Clock
from tender_review.shared.errors import ConflictError, PermanentError
from tender_review.shared.ids import IdGenerator
from tender_review.shared.observability import (
    CorrelationContext,
    log_event,
    record_metric,
)

from .domain import transition_job
from .models import (
    AttemptFailure,
    AttemptStatus,
    OptimizationAttempt,
    OptimizationCandidate,
    OptimizationJob,
    OptimizationReadinessStatus,
    OptimizationStatus,
    OptimizationTraceEvent,
    OptimizationTraceOutcome,
    RootCause,
    TERMINAL_OPTIMIZATION_STATUSES,
    stable_sha256,
)
from .ports import (
    CandidateGenerator,
    CandidateRuleStager,
    OptimizationRepository,
    RegressionEvaluator,
    RootCauseClassifier,
)
from .validation import (
    failure_from_exception,
    new_attempt,
    replace_attempt,
    validate_candidate_boundary,
    validate_result_boundary,
)


LOAD_FAILURE_SAMPLES = "load_failure_samples"
CLASSIFY_ROOT_CAUSE = "classify_root_cause"
GENERATE_MINIMAL_CANDIDATE = "generate_minimal_candidate"
RUN_TARGET_GATE = "run_target_gate"
RUN_PROTECTION_GATE = "run_protection_gate"
RUN_STABILITY_GATE = "run_stability_gate"
STAGE_DRAFT_RULE = "stage_draft_rule"
WAIT_FOR_HUMAN_APPROVAL = "wait_for_human_approval"

BUSINESS_NODES = (
    LOAD_FAILURE_SAMPLES,
    CLASSIFY_ROOT_CAUSE,
    GENERATE_MINIMAL_CANDIDATE,
    RUN_TARGET_GATE,
    RUN_PROTECTION_GATE,
    RUN_STABILITY_GATE,
    STAGE_DRAFT_RULE,
    WAIT_FOR_HUMAN_APPROVAL,
)

_CONTINUE = "continue"
_NEXT_CANDIDATE = "next_candidate"
_NEXT_ROUND = "next_round"
_STAGE = "stage"
_END = "end"


class OptimizationGraphState(TypedDict, total=False):
    optimization_job_id: str
    request_fingerprint: str
    attempt_number: int
    active_candidate_id: str | None
    action: str


class OptimizationGraphNodes:
    def __init__(
        self,
        *,
        repository: OptimizationRepository,
        rule_versions: RuleVersionRepository,
        ids: IdGenerator,
        clock: Clock,
        root_causes: RootCauseClassifier,
        candidates: CandidateGenerator,
        evaluator: RegressionEvaluator,
        stager: CandidateRuleStager,
    ) -> None:
        self._repository = repository
        self._rule_versions = rule_versions
        self._ids = ids
        self._clock = clock
        self._root_causes = root_causes
        self._candidates = candidates
        self._evaluator = evaluator
        self._stager = stager
        self._logger = logging.getLogger("tender_review.optimization.graph")

    def observed(
        self,
        node_name: str,
        operation: Callable[[OptimizationGraphState], OptimizationGraphState],
    ) -> Callable[[OptimizationGraphState], OptimizationGraphState]:
        def invoke(state: OptimizationGraphState) -> OptimizationGraphState:
            job = self._job(state)
            context = self._context(job)
            started = time.perf_counter()
            outcome = "failed"
            log_event(
                self._logger,
                logging.INFO,
                event="optimization.node_started",
                message="Optimization graph node started",
                context=context,
                node_name=node_name,
                attempt_number=state.get("attempt_number", 0),
            )
            try:
                result = operation(state)
            except Exception as exc:
                log_event(
                    self._logger,
                    logging.WARNING,
                    event="optimization.node_failed",
                    message="Optimization graph node failed",
                    context=context,
                    node_name=node_name,
                    error_type=type(exc).__name__,
                    error_code=getattr(exc, "code", None),
                )
                raise
            else:
                outcome = "completed"
                log_event(
                    self._logger,
                    logging.INFO,
                    event="optimization.node_completed",
                    message="Optimization graph node completed",
                    context=context,
                    node_name=node_name,
                    action=result.get("action"),
                )
                return result
            finally:
                record_metric(
                    self._logger,
                    name="optimization_node_duration",
                    value=max(0.0, (time.perf_counter() - started) * 1000.0),
                    unit="ms",
                    source="process_monotonic",
                    context=context,
                    node_name=node_name,
                    outcome=outcome,
                )

        invoke.__name__ = operation.__name__
        return invoke

    def load_failure_samples(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        if job.status in TERMINAL_OPTIMIZATION_STATUSES:
            return {"action": _END}
        if job.readiness.status is not OptimizationReadinessStatus.READY:
            raise PermanentError(
                "optimization readiness changed before graph execution",
                code="optimization_readiness_not_ready",
            )
        if job.status is OptimizationStatus.PENDING:
            job = self._repository.save_job(
                transition_job(job, OptimizationStatus.RUNNING, now=self._clock.now())
            )
        if job.current_round >= job.max_rounds:
            self._repository.save_job(
                transition_job(
                    job,
                    OptimizationStatus.OPTIMIZATION_FAILED,
                    now=self._clock.now(),
                )
            )
            return {"action": _END}
        attempt_number = job.current_round + 1
        attempt = self._repository.get_attempt(
            job.optimization_job_id, attempt_number
        )
        if attempt is None:
            attempt = self._repository.save_attempt(
                new_attempt(
                    attempt_id=self._ids.new(),
                    job_id=job.optimization_job_id,
                    attempt_number=attempt_number,
                    now=self._clock.now(),
                )
            )
        self._trace(
            job.optimization_job_id,
            LOAD_FAILURE_SAMPLES,
            OptimizationTraceOutcome.COMPLETED,
            attempt_number=attempt_number,
            result_sha256=job.readiness.a4_report_sha256,
        )
        return {
            "attempt_number": attempt_number,
            "active_candidate_id": None,
            "action": _CONTINUE,
        }

    def classify_root_cause(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        if attempt.root_cause is None:
            try:
                decision = self._root_causes.analyze(job, attempt.attempt_number)
                attempt = self._save_attempt(attempt, root_cause=decision)
            except Exception as exc:
                return self._fail_round(state, "root_cause", CLASSIFY_ROOT_CAUSE, exc)
        else:
            decision = attempt.root_cause
        self._trace(
            job.optimization_job_id,
            CLASSIFY_ROOT_CAUSE,
            OptimizationTraceOutcome.COMPLETED,
            attempt_number=attempt.attempt_number,
            root_cause=decision.root_cause,
            call_id=decision.call_id,
        )
        return {"action": _CONTINUE}

    def generate_minimal_candidate(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        decision = attempt.root_cause
        if decision is None:
            return self._fail_round(
                state,
                "candidate_generation",
                GENERATE_MINIMAL_CANDIDATE,
                PermanentError(
                    "root cause checkpoint is missing",
                    code="optimization_root_cause_missing",
                ),
            )
        if decision.root_cause is RootCause.LABEL_UNCERTAIN:
            if attempt.status is not AttemptStatus.WAITING_HUMAN:
                attempt = self._save_attempt(
                    attempt,
                    status=AttemptStatus.WAITING_HUMAN,
                    completed_at=self._clock.now(),
                )
            self._trace(
                job.optimization_job_id,
                GENERATE_MINIMAL_CANDIDATE,
                OptimizationTraceOutcome.HUMAN_REQUIRED,
                attempt_number=attempt.attempt_number,
                root_cause=decision.root_cause,
            )
            self._save_progress(
                job,
                attempt,
                terminal_status=OptimizationStatus.WAITING_HUMAN,
            )
            return {"action": _END, "active_candidate_id": None}
        try:
            if not attempt.candidates:
                generated = self._candidates.generate(
                    job,
                    attempt.attempt_number,
                    decision,
                    job.candidates_per_round,
                )
                if not generated:
                    raise PermanentError(
                        "candidate generator returned no bounded candidates",
                        code="optimization_candidates_empty",
                    )
                if len(generated) > job.candidates_per_round:
                    raise PermanentError(
                        "candidate generator exceeded the per-round limit",
                        code="optimization_candidate_limit_exceeded",
                    )
                base = self._rule_versions.get_version(job.base_rule_version_id)
                for candidate in generated:
                    validate_candidate_boundary(
                        job,
                        attempt.attempt_number,
                        base.content_json,
                        base.execution_config_json,
                        candidate,
                    )
                attempt = self._save_attempt(
                    attempt,
                    status=AttemptStatus.EVALUATING,
                    candidates=generated,
                )
            evaluated_ids = {item.candidate_id for item in attempt.evaluations}
            candidate = next(
                (
                    item
                    for item in attempt.candidates
                    if item.candidate_id not in evaluated_ids
                ),
                None,
            )
            if candidate is None:
                return self._complete_rejected_round(job, attempt)
            self._trace(
                job.optimization_job_id,
                GENERATE_MINIMAL_CANDIDATE,
                OptimizationTraceOutcome.COMPLETED,
                attempt_number=attempt.attempt_number,
                candidate_id=candidate.candidate_id,
                root_cause=decision.root_cause,
            )
            return {
                "active_candidate_id": candidate.candidate_id,
                "action": _CONTINUE,
            }
        except Exception as exc:
            return self._fail_round(
                state, "candidate_generation", GENERATE_MINIMAL_CANDIDATE, exc
            )

    def run_target_gate(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        candidate = self._candidate(state, attempt)
        result = next(
            (
                item
                for item in attempt.evaluations
                if item.candidate_id == candidate.candidate_id
            ),
            None,
        )
        try:
            if result is None:
                result = self._evaluator.evaluate(job, candidate)
                validate_result_boundary(job, result)
                attempt = self._save_attempt(
                    attempt,
                    evaluations=(*attempt.evaluations, result),
                )
            self._trace(
                job.optimization_job_id,
                RUN_TARGET_GATE,
                OptimizationTraceOutcome.COMPLETED,
                attempt_number=attempt.attempt_number,
                candidate_id=candidate.candidate_id,
                gate_passed=result.target_gate_passed,
                result_sha256=result.report_sha256,
            )
            return {"action": _CONTINUE}
        except Exception as exc:
            return self._fail_round(state, "evaluation", RUN_TARGET_GATE, exc)

    def run_protection_gate(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        candidate = self._candidate(state, attempt)
        result = self._evaluation(attempt, candidate)
        self._trace(
            job.optimization_job_id,
            RUN_PROTECTION_GATE,
            OptimizationTraceOutcome.COMPLETED,
            attempt_number=attempt.attempt_number,
            candidate_id=candidate.candidate_id,
            gate_passed=result.protection_gate_passed,
            result_sha256=result.report_sha256,
        )
        return {"action": _CONTINUE}

    def run_stability_gate(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        candidate = self._candidate(state, attempt)
        result = self._evaluation(attempt, candidate)
        self._trace(
            job.optimization_job_id,
            RUN_STABILITY_GATE,
            OptimizationTraceOutcome.COMPLETED,
            attempt_number=attempt.attempt_number,
            candidate_id=candidate.candidate_id,
            gate_passed=result.stability_gate_passed,
            result_sha256=result.report_sha256,
        )
        if result.accepted_for_manual_review:
            return {"action": _STAGE}
        evaluated_ids = {item.candidate_id for item in attempt.evaluations}
        if any(
            item.candidate_id not in evaluated_ids for item in attempt.candidates
        ):
            return {"action": _NEXT_CANDIDATE, "active_candidate_id": None}
        return self._complete_rejected_round(job, attempt)

    def stage_draft_rule(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        candidate = self._candidate(state, attempt)
        try:
            if attempt.selected_candidate_id is None:
                attempt = self._save_attempt(
                    attempt,
                    status=AttemptStatus.COMPLETED,
                    selected_candidate_id=candidate.candidate_id,
                    completed_at=self._clock.now(),
                )
            version_id = attempt.candidate_rule_version_id
            if version_id is None:
                version_id = self._stager.stage_candidate(job, attempt, candidate)
                attempt = self._save_attempt(
                    attempt, candidate_rule_version_id=version_id
                )
            version = self._rule_versions.get_version(version_id)
            if (
                version.status is not RuleVersionStatus.DRAFT
                or version.parent_version_id != job.base_rule_version_id
            ):
                raise PermanentError(
                    "optimization graph may stage only a child DRAFT rule version",
                    code="optimization_staged_rule_boundary",
                )
            self._trace(
                job.optimization_job_id,
                STAGE_DRAFT_RULE,
                OptimizationTraceOutcome.COMPLETED,
                attempt_number=attempt.attempt_number,
                candidate_id=candidate.candidate_id,
                result_sha256=version.content_sha256,
            )
            return {"action": _CONTINUE}
        except Exception as exc:
            return self._fail_round(state, "staging", STAGE_DRAFT_RULE, exc)

    def wait_for_human_approval(
        self, state: OptimizationGraphState
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        if attempt.candidate_rule_version_id is None:
            return self._fail_round(
                state,
                "staging",
                WAIT_FOR_HUMAN_APPROVAL,
                PermanentError(
                    "human approval boundary requires a staged DRAFT",
                    code="optimization_draft_missing",
                ),
            )
        self._trace(
            job.optimization_job_id,
            WAIT_FOR_HUMAN_APPROVAL,
            OptimizationTraceOutcome.COMPLETED,
            attempt_number=attempt.attempt_number,
            candidate_id=attempt.selected_candidate_id,
            result_sha256=attempt.checkpoint_sha256,
        )
        self._save_progress(
            job,
            attempt,
            terminal_status=OptimizationStatus.WAITING_APPROVAL,
            candidate_rule_version_id=attempt.candidate_rule_version_id,
        )
        return {"action": _END}

    def _complete_rejected_round(
        self, job: OptimizationJob, attempt: OptimizationAttempt
    ) -> OptimizationGraphState:
        if attempt.status is not AttemptStatus.COMPLETED:
            attempt = self._save_attempt(
                attempt,
                status=AttemptStatus.COMPLETED,
                completed_at=self._clock.now(),
            )
        terminal = (
            OptimizationStatus.OPTIMIZATION_FAILED
            if attempt.attempt_number >= job.max_rounds
            else None
        )
        self._save_progress(job, attempt, terminal_status=terminal)
        return {"action": _END if terminal else _NEXT_ROUND}

    def _fail_round(
        self,
        state: OptimizationGraphState,
        phase: str,
        node: str,
        exc: Exception,
    ) -> OptimizationGraphState:
        job = self._job(state)
        attempt = self._attempt(state)
        if attempt.status is AttemptStatus.FAILED and attempt.failure is not None:
            failure = attempt.failure
        else:
            failure = failure_from_exception(phase, exc)
            attempt = self._save_attempt(
                attempt,
                status=AttemptStatus.FAILED,
                failure=failure,
                completed_at=self._clock.now(),
            )
        self._trace(
            job.optimization_job_id,
            node,
            OptimizationTraceOutcome.FAILED,
            attempt_number=attempt.attempt_number,
            candidate_id=state.get("active_candidate_id"),
            failure_code=failure.code,
            call_id=failure.call_id,
        )
        terminal = (
            OptimizationStatus.OPTIMIZATION_FAILED
            if attempt.attempt_number >= job.max_rounds
            else None
        )
        self._save_progress(
            job,
            attempt,
            terminal_status=terminal,
            failure=failure,
        )
        return {"action": _END if terminal else _NEXT_ROUND}

    def _job(self, state: OptimizationGraphState) -> OptimizationJob:
        return self._repository.get_job(state["optimization_job_id"])

    def _attempt(self, state: OptimizationGraphState) -> OptimizationAttempt:
        attempt = self._repository.get_attempt(
            state["optimization_job_id"], state["attempt_number"]
        )
        if attempt is None:
            raise ConflictError(
                "optimization attempt checkpoint is missing",
                code="optimization_attempt_missing",
            )
        return attempt

    @staticmethod
    def _candidate(
        state: OptimizationGraphState, attempt: OptimizationAttempt
    ) -> OptimizationCandidate:
        candidate_id = state.get("active_candidate_id")
        candidate = next(
            (item for item in attempt.candidates if item.candidate_id == candidate_id),
            None,
        )
        if candidate is None:
            raise ConflictError(
                "active optimization candidate is missing",
                code="optimization_candidate_missing",
            )
        return candidate

    @staticmethod
    def _evaluation(
        attempt: OptimizationAttempt, candidate: OptimizationCandidate
    ):
        result = next(
            (
                item
                for item in attempt.evaluations
                if item.candidate_id == candidate.candidate_id
            ),
            None,
        )
        if result is None:
            raise ConflictError(
                "candidate evaluation checkpoint is missing",
                code="optimization_evaluation_missing",
            )
        return result

    def _save_attempt(
        self, attempt: OptimizationAttempt, **updates: Any
    ) -> OptimizationAttempt:
        return self._repository.save_attempt(
            replace_attempt(attempt, self._clock.now(), **updates)
        )

    def _save_progress(
        self,
        job: OptimizationJob,
        attempt: OptimizationAttempt,
        *,
        terminal_status: OptimizationStatus | None,
        candidate_rule_version_id: str | None = None,
        failure: AttemptFailure | None = None,
    ) -> OptimizationJob:
        current = self._repository.get_job(job.optimization_job_id)
        if current.status is OptimizationStatus.CANCELLED:
            return current
        updates: dict[str, Any] = {
            "current_round": attempt.attempt_number,
            "last_checkpoint_sha256": attempt.checkpoint_sha256,
        }
        if candidate_rule_version_id is not None:
            updates["candidate_rule_version_id"] = candidate_rule_version_id
        if failure is not None and (
            not current.failure_trajectory
            or current.failure_trajectory[-1] != failure
            or current.last_checkpoint_sha256 != attempt.checkpoint_sha256
        ):
            updates["failure_trajectory"] = (*current.failure_trajectory, failure)
        if terminal_status is not None:
            saved = transition_job(
                current, terminal_status, now=self._clock.now(), **updates
            )
        else:
            payload = current.model_dump(mode="json")
            payload.update(updates)
            payload["updated_at"] = self._clock.now()
            saved = OptimizationJob.model_validate(payload)
        return self._repository.save_job(saved)

    def _trace(
        self,
        optimization_job_id: str,
        node: str,
        outcome: OptimizationTraceOutcome,
        *,
        attempt_number: int,
        candidate_id: str | None = None,
        root_cause: RootCause | None = None,
        call_id: str | None = None,
        gate_passed: bool | None = None,
        result_sha256: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        identity = {
            "optimization_job_id": optimization_job_id,
            "node": node,
            "outcome": outcome,
            "attempt_number": attempt_number,
            "candidate_id": candidate_id,
            "result_sha256": result_sha256,
            "failure_code": failure_code,
        }
        event_id = stable_sha256(identity)
        job = self._repository.get_job(optimization_job_id)
        if any(item.event_id == event_id for item in job.graph_trace):
            return
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "node": node,
            "outcome": outcome,
            "attempt_number": attempt_number,
            "candidate_id": candidate_id,
            "root_cause": root_cause,
            "call_id": call_id,
            "gate_passed": gate_passed,
            "result_sha256": result_sha256,
            "failure_code": failure_code,
            "recorded_at": self._clock.now(),
        }
        event = OptimizationTraceEvent(
            **payload, event_sha256=stable_sha256(payload)
        )
        self._repository.save_job(
            job.model_copy(
                update={
                    "graph_trace": (*job.graph_trace, event),
                    "updated_at": self._clock.now(),
                }
            )
        )
        log_event(
            self._logger,
            logging.INFO,
            event="optimization.trace_recorded",
            message="Optimization graph trace recorded",
            context=self._context(job, call_id=call_id),
            node_name=node,
            outcome=outcome.value,
            attempt_number=attempt_number,
            candidate_id=candidate_id,
            failure_code=failure_code,
            trace_event_id=event.event_id,
        )

    @staticmethod
    def _context(
        job: OptimizationJob, *, call_id: str | None = None
    ) -> CorrelationContext:
        return CorrelationContext(
            job_id=job.optimization_job_id,
            thread_id=f"optimization:{job.optimization_job_id}",
            checkpoint_id=job.last_checkpoint_sha256,
            call_id=call_id or f"optimization:{job.optimization_job_id}",
            rule_version=job.base_rule_version_id,
            dataset_version=job.dataset_version_id,
            model_config=f"sha256:{job.hashes.model_sha256}",
        )


def _route(state: OptimizationGraphState) -> str:
    return state.get("action", _END)


def build_optimization_graph(
    nodes: OptimizationGraphNodes,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph:
    builder = StateGraph(OptimizationGraphState)
    builder.add_node(
        LOAD_FAILURE_SAMPLES,
        nodes.observed(LOAD_FAILURE_SAMPLES, nodes.load_failure_samples),
    )
    builder.add_node(
        CLASSIFY_ROOT_CAUSE,
        nodes.observed(CLASSIFY_ROOT_CAUSE, nodes.classify_root_cause),
    )
    builder.add_node(
        GENERATE_MINIMAL_CANDIDATE,
        nodes.observed(GENERATE_MINIMAL_CANDIDATE, nodes.generate_minimal_candidate),
    )
    builder.add_node(
        RUN_TARGET_GATE,
        nodes.observed(RUN_TARGET_GATE, nodes.run_target_gate),
    )
    builder.add_node(
        RUN_PROTECTION_GATE,
        nodes.observed(RUN_PROTECTION_GATE, nodes.run_protection_gate),
    )
    builder.add_node(
        RUN_STABILITY_GATE,
        nodes.observed(RUN_STABILITY_GATE, nodes.run_stability_gate),
    )
    builder.add_node(
        STAGE_DRAFT_RULE,
        nodes.observed(STAGE_DRAFT_RULE, nodes.stage_draft_rule),
    )
    builder.add_node(
        WAIT_FOR_HUMAN_APPROVAL,
        nodes.observed(WAIT_FOR_HUMAN_APPROVAL, nodes.wait_for_human_approval),
    )

    builder.add_edge(START, LOAD_FAILURE_SAMPLES)
    builder.add_conditional_edges(
        LOAD_FAILURE_SAMPLES,
        _route,
        {_CONTINUE: CLASSIFY_ROOT_CAUSE, _END: END},
    )
    builder.add_conditional_edges(
        CLASSIFY_ROOT_CAUSE,
        _route,
        {
            _CONTINUE: GENERATE_MINIMAL_CANDIDATE,
            _NEXT_ROUND: LOAD_FAILURE_SAMPLES,
            _END: END,
        },
    )
    builder.add_conditional_edges(
        GENERATE_MINIMAL_CANDIDATE,
        _route,
        {
            _CONTINUE: RUN_TARGET_GATE,
            _NEXT_ROUND: LOAD_FAILURE_SAMPLES,
            _END: END,
        },
    )
    builder.add_conditional_edges(
        RUN_TARGET_GATE,
        _route,
        {
            _CONTINUE: RUN_PROTECTION_GATE,
            _NEXT_ROUND: LOAD_FAILURE_SAMPLES,
            _END: END,
        },
    )
    builder.add_edge(RUN_PROTECTION_GATE, RUN_STABILITY_GATE)
    builder.add_conditional_edges(
        RUN_STABILITY_GATE,
        _route,
        {
            _STAGE: STAGE_DRAFT_RULE,
            _NEXT_CANDIDATE: GENERATE_MINIMAL_CANDIDATE,
            _NEXT_ROUND: LOAD_FAILURE_SAMPLES,
            _END: END,
        },
    )
    builder.add_conditional_edges(
        STAGE_DRAFT_RULE,
        _route,
        {
            _CONTINUE: WAIT_FOR_HUMAN_APPROVAL,
            _NEXT_ROUND: LOAD_FAILURE_SAMPLES,
            _END: END,
        },
    )
    builder.add_edge(WAIT_FOR_HUMAN_APPROVAL, END)
    return builder.compile(checkpointer=checkpointer, name="bounded-rule-optimization")


class LangGraphOptimizationWorkflow:
    def __init__(
        self,
        *,
        repository: OptimizationRepository,
        rule_versions: RuleVersionRepository,
        ids: IdGenerator,
        clock: Clock,
        root_causes: RootCauseClassifier,
        candidates: CandidateGenerator,
        evaluator: RegressionEvaluator,
        stager: CandidateRuleStager,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self._repository = repository
        self._checkpointer = checkpointer or InMemorySaver()
        self._nodes = OptimizationGraphNodes(
            repository=repository,
            rule_versions=rule_versions,
            ids=ids,
            clock=clock,
            root_causes=root_causes,
            candidates=candidates,
            evaluator=evaluator,
            stager=stager,
        )
        self._graph = build_optimization_graph(self._nodes, self._checkpointer)

    @property
    def compiled_graph(self) -> CompiledStateGraph:
        return self._graph

    def run(
        self,
        optimization_job_id: str,
        *,
        interrupt_after: Sequence[str] = (),
    ) -> OptimizationJob:
        job = self._repository.get_job(optimization_job_id)
        if job.status in TERMINAL_OPTIMIZATION_STATUSES:
            return job
        config = _thread_config(optimization_job_id)
        fingerprint = _job_fingerprint(job)
        snapshot = self._graph.get_state(config)
        if snapshot.values:
            if snapshot.values.get("request_fingerprint") != fingerprint:
                raise ConflictError(
                    "optimization checkpoint belongs to different immutable inputs",
                    code="optimization_checkpoint_input_mismatch",
                )
            self._graph.invoke(
                None,
                config,
                interrupt_after=tuple(interrupt_after),
            )
        else:
            self._graph.invoke(
                {
                    "optimization_job_id": optimization_job_id,
                    "request_fingerprint": fingerprint,
                    "attempt_number": 0,
                    "active_candidate_id": None,
                    "action": _CONTINUE,
                },
                config,
                interrupt_after=tuple(interrupt_after),
            )
        return self._repository.get_job(optimization_job_id)


def _thread_config(optimization_job_id: str) -> dict[str, dict[str, str]]:
    return {
        "configurable": {
            "thread_id": f"optimization:{optimization_job_id}",
        }
    }


def _job_fingerprint(job: OptimizationJob) -> str:
    return stable_sha256(
        {
            "optimization_job_id": job.optimization_job_id,
            "base_rule_version_id": job.base_rule_version_id,
            "dataset_version_id": job.dataset_version_id,
            "max_rounds": job.max_rounds,
            "candidates_per_round": job.candidates_per_round,
            "required_stability_runs": job.required_stability_runs,
            "samples": job.samples,
            "hashes": job.hashes,
            "provenance": job.provenance,
            "readiness": job.readiness,
        }
    )
