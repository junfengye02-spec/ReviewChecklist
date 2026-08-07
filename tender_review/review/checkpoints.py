from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import ReviewGraphNode, ReviewLifecycle, ReviewProcessingStage


def review_thread_id(review_job_id: str) -> str:
    """Map one review job to the stable LangGraph thread used for recovery."""

    if not review_job_id.strip():
        raise ValueError("review_job_id must not be blank")
    if len(review_job_id) > 128:
        raise ValueError("review_job_id must not exceed 128 characters")
    return review_job_id


@dataclass(frozen=True, slots=True)
class ReviewCheckpointPointer:
    review_job_id: str
    thread_id: str
    checkpoint_id: str
    node: ReviewGraphNode
    stage: ReviewProcessingStage | None
    lifecycle: ReviewLifecycle


@runtime_checkable
class ReviewCheckpointReader(Protocol):
    def latest_checkpoint(
        self, review_job_id: str
    ) -> ReviewCheckpointPointer | None: ...
