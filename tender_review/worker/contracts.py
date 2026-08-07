from __future__ import annotations

from typing import Protocol, runtime_checkable

from tender_review.jobs.models import JobHandlerOutcome, JobMessage, JobResult


@runtime_checkable
class WorkHandler(Protocol):
    def __call__(self, job: JobMessage) -> JobResult | JobHandlerOutcome: ...
