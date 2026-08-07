from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import AuditEvent, EvaluationReport, EvaluationRun, WorkbenchResourceIndex


@runtime_checkable
class EvaluationRunRepository(Protocol):
    def get_run(self, run_id: str) -> EvaluationRun: ...

    def get_report(self, run_id: str) -> EvaluationReport: ...


@runtime_checkable
class WorkbenchIndexRepository(Protocol):
    def get_index(self) -> WorkbenchResourceIndex: ...


@runtime_checkable
class AuditEventSink(Protocol):
    def append(self, event: AuditEvent) -> AuditEvent: ...

    def list_events(self, limit: int) -> tuple[AuditEvent, ...]: ...
