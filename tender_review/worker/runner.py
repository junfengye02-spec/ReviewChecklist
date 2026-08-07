from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from tender_review.jobs.models import (
    CheckpointState,
    JobCheckpoint,
    JobFailure,
    JobHandlerOutcome,
    JobHandlerStatus,
    JobLease,
    JobMessage,
    JobResult,
    ReviewStage,
)
from tender_review.jobs.ports import JobRepository, LeaseManager
from tender_review.shared.clock import Clock
from tender_review.shared.errors import ConflictError, ErrorCategory, ServiceError
from tender_review.shared.observability import (
    CorrelationContext,
    log_event,
    record_metric,
)

from .contracts import WorkHandler


class _LeaseState:
    def __init__(self, lease: JobLease) -> None:
        self._lease = lease
        self._lost = False
        self._lock = threading.Lock()

    def current(self) -> JobLease:
        with self._lock:
            return self._lease

    def replace(self, lease: JobLease) -> None:
        with self._lock:
            self._lease = lease

    def mark_lost(self) -> None:
        with self._lock:
            self._lost = True

    def lost(self) -> bool:
        with self._lock:
            return self._lost


class WorkerExecutionContext:
    """Lease-aware operations available to a resumable two-argument handler."""

    def __init__(
        self,
        *,
        job_id: str,
        repository: JobRepository,
        leases: LeaseManager,
        lease_state: _LeaseState,
        clock: Clock,
        lease_seconds: int,
    ) -> None:
        self.job_id = job_id
        self._repository = repository
        self._leases = leases
        self._lease_state = lease_state
        self._clock = clock
        self._lease_seconds = lease_seconds

    @property
    def lease(self) -> JobLease:
        return self._lease_state.current()

    def latest_checkpoint(self) -> object | None:
        lister = getattr(self._repository, "list_checkpoints", None)
        if callable(lister):
            checkpoints = lister(self.job_id)
            return checkpoints[-1] if checkpoints else None
        # Older stage-1 adapters exposed only this read helper.  Retain the
        # fallback for CLI compatibility while durable adapters use DTOs.
        loader = getattr(self._repository, "load_latest_checkpoint", None)
        return loader(self.job_id) if callable(loader) else None

    def save_checkpoint(
        self,
        *,
        node_name: str,
        stage: str,
        state_json: Mapping[str, Any],
        output_artifact_id: str | None = None,
    ) -> object:
        saver = getattr(self._repository, "save_checkpoint", None)
        if not callable(saver):
            raise RuntimeError("The configured repository does not support checkpoints")
        self.heartbeat()
        checkpoint = JobCheckpoint(
            job_id=self.job_id,
            node_name=node_name,
            stage=ReviewStage(stage),
            lease_token=self.lease.token,
            state=CheckpointState.model_validate(state_json),
            output_artifact_id=output_artifact_id,
            completed_at=self._clock.now(),
        )
        return saver(checkpoint)

    def heartbeat(self) -> JobLease:
        renewed = self._leases.renew(
            self.lease,
            now=self._clock.now(),
            lease_seconds=self._lease_seconds,
        )
        self._lease_state.replace(renewed)
        return renewed


class Worker:
    def __init__(
        self,
        *,
        worker_id: str,
        repository: JobRepository,
        leases: LeaseManager,
        handlers: Mapping[str, WorkHandler],
        clock: Clock,
        lease_seconds: int = 30,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must not be negative")
        self.worker_id = worker_id
        self.repository = repository
        self.leases = leases
        self.handlers = handlers
        self.clock = clock
        self.lease_seconds = lease_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.logger = logging.getLogger("tender_review.worker")

    def run_once(self) -> bool:
        claim_started = time.perf_counter()
        claimed = self._claim_work()
        claim_latency_ms = (time.perf_counter() - claim_started) * 1000.0
        claimed_job = claimed[0] if claimed is not None else None
        claim_context = (
            self._job_context(claimed_job)
            if claimed_job is not None
            else CorrelationContext()
        )
        record_metric(
            self.logger,
            name="job_claim_latency",
            value=claim_latency_ms,
            unit="ms",
            source="worker_process_monotonic",
            context=claim_context,
            worker_id=self.worker_id,
            claimed=claimed is not None,
        )
        record_metric(
            self.logger,
            name="worker_empty_poll",
            value=0 if claimed is not None else 1,
            unit="boolean",
            source="worker_claim_result",
            context=claim_context,
            worker_id=self.worker_id,
        )
        if claimed is None:
            return False
        job, lease = claimed
        lease_state = _LeaseState(lease)
        context = WorkerExecutionContext(
            job_id=job.job_id,
            repository=self.repository,
            leases=self.leases,
            lease_state=lease_state,
            clock=self.clock,
            lease_seconds=self.lease_seconds,
        )
        correlation = self._job_context(job)
        log_event(
            self.logger,
            logging.INFO,
            event="worker.job_started",
            message="Worker job started",
            context=correlation,
            worker_id=self.worker_id,
            job_type=job.job_type,
            lease_token=lease.token,
            attempt=job.attempt,
            recovery_count=max(0, job.attempt - 1),
        )
        if job.enqueued_at is None:
            record_metric(
                self.logger,
                name="job_queue_wait",
                value=None,
                unit="ms",
                source="durable_job_created_at",
                status="not_collected",
                context=correlation,
            )
        else:
            queue_wait_ms = max(
                0.0,
                (self.clock.now() - job.enqueued_at).total_seconds() * 1000.0,
            )
            record_metric(
                self.logger,
                name="job_queue_wait",
                value=queue_wait_ms,
                unit="ms",
                source="durable_job_created_at",
                context=correlation,
            )
        stop_heartbeat, heartbeat_thread = self._start_heartbeat(job, lease_state)
        result: JobResult | JobHandlerOutcome | None = None
        service_error: ServiceError | None = None
        unhandled_error: Exception | None = None
        try:
            result = self._dispatch(job, context)
        except ServiceError as exc:
            service_error = exc
        except Exception as exc:
            unhandled_error = exc
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join()

        current_lease = lease_state.current()
        try:
            if not lease_state.lost():
                try:
                    current_lease = context.heartbeat()
                except Exception:
                    lease_state.mark_lost()
            if lease_state.lost():
                log_event(
                    self.logger,
                    logging.WARNING,
                    event="worker.lease_lost",
                    message="Worker lease was lost before a final write",
                    context=correlation,
                    worker_id=self.worker_id,
                    lease_token=current_lease.token,
                )
            elif result is not None:
                self._write_outcome(job, current_lease, result)
            elif service_error is not None:
                self._mark_service_failure(job, current_lease, service_error)
            elif unhandled_error is not None:
                self._mark_unhandled_failure(job, current_lease, unhandled_error)
        finally:
            self.leases.release(current_lease)
        return True

    def _claim_work(self) -> tuple[JobMessage, JobLease] | None:
        atomic_claim = getattr(self.repository, "claim_next", None)
        if callable(atomic_claim):
            claimed = atomic_claim(
                self.worker_id,
                now=self.clock.now(),
                lease_seconds=self.lease_seconds,
            )
            if claimed is None:
                return None
            return claimed.message, claimed.lease

        job = self.repository.next_queued()
        if job is None:
            return None
        lease = self.leases.acquire(
            job.job_id,
            self.worker_id,
            now=self.clock.now(),
            lease_seconds=self.lease_seconds,
        )
        return None if lease is None else (job, lease)

    def _mark_completed(
        self, job: JobMessage, lease: JobLease, result: JobResult
    ) -> None:
        try:
            self.repository.mark_completed(job.job_id, lease.token, result)
            log_event(
                self.logger,
                logging.INFO,
                event="worker.job_completed",
                message="Worker job completed",
                context=self._job_context(job),
                worker_id=self.worker_id,
                lease_token=lease.token,
            )
        except ConflictError:
            self._log_fenced_final_write(job, lease)

    def _write_outcome(
        self,
        job: JobMessage,
        lease: JobLease,
        outcome: JobResult | JobHandlerOutcome,
    ) -> None:
        if isinstance(outcome, JobResult):
            self._mark_completed(job, lease, outcome)
            return
        if outcome.status is JobHandlerStatus.COMPLETED:
            self._mark_completed(job, lease, outcome.result)
            return
        if outcome.status is JobHandlerStatus.WAITING_HUMAN:
            try:
                self.repository.mark_waiting_human(
                    job.job_id, lease.token, outcome.result
                )
                log_event(
                    self.logger,
                    logging.INFO,
                    event="worker.job_waiting_human",
                    message="Worker handed the job to human review",
                    context=self._job_context(job),
                    worker_id=self.worker_id,
                    lease_token=lease.token,
                )
            except ConflictError:
                self._log_fenced_final_write(job, lease)
            return
        assert outcome.failure is not None
        self._mark_failure(job, lease, outcome.failure)

    def _mark_failure(
        self, job: JobMessage, lease: JobLease, failure: JobFailure
    ) -> None:
        try:
            self.repository.mark_failed(job.job_id, lease.token, failure)
            log_event(
                self.logger,
                logging.WARNING,
                event="worker.job_failed",
                message="Worker recorded a typed job failure",
                context=self._job_context(job),
                worker_id=self.worker_id,
                lease_token=lease.token,
                error_code=failure.code,
                error_category=failure.category.value,
                retryable=failure.retryable,
                failure_stage=failure.stage.value if failure.stage else None,
            )
            if failure.retryable:
                record_metric(
                    self.logger,
                    name="job_retry_scheduled",
                    value=1,
                    unit="count",
                    source="job_repository_outcome",
                    context=self._job_context(job),
                    attempt=job.attempt,
                    error_code=failure.code,
                )
        except ConflictError:
            self._log_fenced_final_write(job, lease)

    def _mark_service_failure(
        self, job: JobMessage, lease: JobLease, exc: ServiceError
    ) -> None:
        try:
            self.repository.mark_failed(
                job.job_id,
                lease.token,
                JobFailure(
                    code=exc.code,
                    message=exc.message,
                    category=exc.category,
                    retryable=exc.retryable,
                    stage=ReviewStage.PARSING,
                ),
            )
            log_event(
                self.logger,
                logging.WARNING,
                event="worker.job_failed",
                message="Worker job rejected",
                context=self._job_context(job),
                worker_id=self.worker_id,
                error_code=exc.code,
                retryable=exc.retryable,
            )
        except ConflictError:
            self._log_fenced_final_write(job, lease)

    def _mark_unhandled_failure(
        self, job: JobMessage, lease: JobLease, exc: Exception
    ) -> None:
        try:
            self.repository.mark_failed(
                job.job_id,
                lease.token,
                JobFailure(
                    code="unhandled_worker_error",
                    message="Unhandled worker error",
                    category=ErrorCategory.INTERNAL,
                    retryable=True,
                    stage=ReviewStage.PARSING,
                ),
            )
            log_event(
                self.logger,
                logging.ERROR,
                event="worker.unhandled_error",
                message="Unhandled worker error",
                context=self._job_context(job),
                worker_id=self.worker_id,
                error_type=type(exc).__name__,
            )
        except ConflictError:
            self._log_fenced_final_write(job, lease)

    def _log_fenced_final_write(self, job: JobMessage, lease: JobLease) -> None:
        log_event(
            self.logger,
            logging.WARNING,
            event="worker.final_write_fenced",
            message="Worker final write was rejected by the lease fence",
            context=self._job_context(job),
            worker_id=self.worker_id,
            lease_token=lease.token,
        )

    def _start_heartbeat(
        self, job: JobMessage, lease_state: _LeaseState
    ) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()
        interval = max(0.1, min(10.0, self.lease_seconds / 3))

        def heartbeat_loop() -> None:
            while not stop.wait(interval):
                try:
                    renewed = self.leases.renew(
                        lease_state.current(),
                        now=self.clock.now(),
                        lease_seconds=self.lease_seconds,
                    )
                    lease_state.replace(renewed)
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        event="worker.heartbeat_renewed",
                        message="Worker lease heartbeat renewed",
                        context=self._job_context(job),
                        worker_id=self.worker_id,
                        lease_token=renewed.token,
                    )
                except Exception as exc:
                    lease_state.mark_lost()
                    log_event(
                        self.logger,
                        logging.WARNING,
                        event="worker.heartbeat_failed",
                        message="Worker heartbeat failed",
                        context=self._job_context(job),
                        worker_id=self.worker_id,
                        error_type=type(exc).__name__,
                    )
                    return

        thread = threading.Thread(
            target=heartbeat_loop,
            name=f"job-heartbeat-{job.job_id}",
            daemon=True,
        )
        thread.start()
        return stop, thread

    def run_forever(
        self,
        stop_event: threading.Event | None = None,
        *,
        max_iterations: int | None = None,
    ) -> int:
        stop = stop_event or threading.Event()
        iterations = 0
        while not stop.is_set():
            if max_iterations is not None and iterations >= max_iterations:
                break
            processed = self.run_once()
            iterations += 1
            if not processed and self.poll_interval_seconds:
                stop.wait(self.poll_interval_seconds)
        return iterations

    def _dispatch(
        self, job: JobMessage, context: WorkerExecutionContext
    ) -> JobResult | JobHandlerOutcome:
        handler = self.handlers.get(job.job_type)
        if handler is None:
            from tender_review.shared.errors import PermanentError

            raise PermanentError(
                f"No worker handler registered for {job.job_type!r}",
                code="unknown_job_type",
                details={"job_type": job.job_type},
            )
        if self._accepts_execution_context(handler):
            result = handler(job, context)  # type: ignore[call-arg]
        else:
            result = handler(job)
        if not isinstance(result, (JobResult, JobHandlerOutcome)):
            raise TypeError("Worker handlers must return JobResult or JobHandlerOutcome")
        return result

    @staticmethod
    def _accepts_execution_context(handler: WorkHandler) -> bool:
        try:
            parameters = tuple(inspect.signature(handler).parameters.values())
        except (TypeError, ValueError):
            return False
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        )
        return len(positional) >= 2 or any(
            parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters
        )

    def _job_context(self, job: JobMessage) -> CorrelationContext:
        getter = getattr(self.repository, "get_review_execution_spec", None)
        if callable(getter):
            try:
                spec = getter(job.job_id)
            except Exception:
                spec = None
            if spec is not None:
                return CorrelationContext(
                    job_id=job.job_id,
                    thread_id=job.job_id,
                    call_id=f"review:{job.job_id}",
                    rule_version=getattr(spec, "rule_version_id", None),
                    dataset_version=getattr(spec, "dataset_version_id", None),
                    model_config=getattr(spec, "model_config_id", None),
                )
        call_prefix = "document-parse" if job.job_type == "document_parse" else job.job_type
        return CorrelationContext(
            job_id=job.job_id,
            thread_id=job.job_id,
            call_id=f"{call_prefix}:{job.job_id}",
        )
