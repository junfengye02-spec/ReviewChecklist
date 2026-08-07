from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Finding, HumanDecision


@runtime_checkable
class FindingRepository(Protocol):
    def add_finding(self, finding: Finding) -> Finding: ...

    def get_finding(self, finding_id: str) -> Finding: ...

    def list_findings(self, review_job_id: str) -> tuple[Finding, ...]: ...

    def list_decisions(self, finding_id: str) -> tuple[HumanDecision, ...]: ...

    def append_decision(
        self, *, finding: Finding, decision: HumanDecision
    ) -> Finding: ...
