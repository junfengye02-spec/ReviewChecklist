from __future__ import annotations

import logging
from threading import RLock

from tender_review.shared.errors import NotFoundError

from .models import AuditEvent, EvaluationReport, EvaluationRun, WorkbenchResourceIndex


class StaticEvaluationRunRepository:
    """Immutable run/report catalog. Empty catalogs are valid production state."""

    def __init__(
        self,
        runs: tuple[EvaluationRun, ...] = (),
        reports: tuple[EvaluationReport, ...] = (),
    ) -> None:
        self._runs = {item.run_id: item for item in runs}
        self._reports = {item.run_id: item for item in reports}

    def get_run(self, run_id: str) -> EvaluationRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise NotFoundError(
                "evaluation run does not exist", code="evaluation_run_not_found"
            ) from exc

    def get_report(self, run_id: str) -> EvaluationReport:
        try:
            return self._reports[run_id]
        except KeyError as exc:
            raise NotFoundError(
                "evaluation report does not exist", code="evaluation_report_not_found"
            ) from exc


class StaticWorkbenchIndexRepository:
    def __init__(self, index: WorkbenchResourceIndex) -> None:
        self._index = index

    def get_index(self) -> WorkbenchResourceIndex:
        return self._index


class InMemoryAuditEventSink:
    def __init__(self, initial: tuple[AuditEvent, ...] = ()) -> None:
        self._events = list(initial)
        self._lock = RLock()

    def append(self, event: AuditEvent) -> AuditEvent:
        with self._lock:
            self._events.append(event)
            return event

    def list_events(self, limit: int) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(reversed(self._events[-limit:]))


class LoggingAuditEventSink:
    """Production sink: structured logs only, with no sensitive source text."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("tender_review.audit")

    def append(self, event: AuditEvent) -> AuditEvent:
        payload = event.model_dump(mode="json")
        payload["model_config"] = payload.pop("model_configuration", None)
        self._logger.info(
            "Audit event",
            extra={"event": "audit.event", **payload},
        )
        return event

    def list_events(self, limit: int) -> tuple[AuditEvent, ...]:
        del limit
        return ()
