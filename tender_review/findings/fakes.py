from __future__ import annotations

from threading import RLock

from tender_review.shared.errors import ConflictError, NotFoundError

from .models import Finding, HumanDecision


class InMemoryFindingRepository:
    def __init__(self) -> None:
        self._findings: dict[str, Finding] = {}
        self._decisions: dict[str, list[HumanDecision]] = {}
        self._lock = RLock()

    def add_finding(self, finding: Finding) -> Finding:
        with self._lock:
            existing = self._findings.get(finding.finding_id)
            if existing is not None:
                if existing == finding:
                    return existing
                raise ConflictError("finding already exists", code="finding_conflict")
            self._findings[finding.finding_id] = finding
            self._decisions[finding.finding_id] = []
            return finding

    def get_finding(self, finding_id: str) -> Finding:
        with self._lock:
            try:
                return self._findings[finding_id]
            except KeyError as exc:
                raise NotFoundError("finding does not exist", code="finding_not_found") from exc

    def list_findings(self, review_job_id: str) -> tuple[Finding, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        item
                        for item in self._findings.values()
                        if item.review_job_id == review_job_id
                    ),
                    key=lambda item: (item.created_at, item.finding_id),
                )
            )

    def list_decisions(self, finding_id: str) -> tuple[HumanDecision, ...]:
        with self._lock:
            self.get_finding(finding_id)
            return tuple(self._decisions[finding_id])

    def append_decision(
        self, *, finding: Finding, decision: HumanDecision
    ) -> Finding:
        with self._lock:
            current = self.get_finding(finding.finding_id)
            if current.finding_content_sha256 != finding.finding_content_sha256:
                raise ConflictError("finding content is immutable", code="finding_content_changed")
            if any(
                item.decision_id == decision.decision_id
                or item.decision_sha256 == decision.decision_sha256
                for item in self._decisions[finding.finding_id]
            ):
                raise ConflictError("decision already exists", code="decision_duplicate")
            self._decisions[finding.finding_id].append(decision)
            self._findings[finding.finding_id] = finding
            return finding
