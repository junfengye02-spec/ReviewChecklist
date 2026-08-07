"""Job application boundary and offline test doubles."""

from .fakes import FakeLeaseManager, InMemoryJobRepository
from .models import (
    JobFailure,
    JobHandlerOutcome,
    JobHandlerStatus,
    JobLease,
    JobMessage,
    JobResult,
)
from .ports import JobRepository, LeaseManager, ReviewJobRepository
from .public import (
    CheckpointState,
    CheckpointValue,
    CreateReviewJobCommand,
    ExecutionArtifactReference,
    IdempotencyRecord,
    IdempotentReviewJob,
    JobCheckpoint,
    JobLifecycle,
    JobStatus,
    ReviewJob,
    ReviewExecutionSpec,
    ReviewExecutionSpecDraft,
    ReviewExecutionSpecParser,
    ReviewJobService,
    ReviewStage,
)

__all__ = [
    "CheckpointState",
    "CheckpointValue",
    "CreateReviewJobCommand",
    "ExecutionArtifactReference",
    "FakeLeaseManager",
    "IdempotencyRecord",
    "IdempotentReviewJob",
    "InMemoryJobRepository",
    "JobCheckpoint",
    "JobFailure",
    "JobHandlerOutcome",
    "JobHandlerStatus",
    "JobLease",
    "JobLifecycle",
    "JobStatus",
    "JobMessage",
    "JobRepository",
    "JobResult",
    "LeaseManager",
    "ReviewJob",
    "ReviewExecutionSpec",
    "ReviewExecutionSpecDraft",
    "ReviewExecutionSpecParser",
    "ReviewJobRepository",
    "ReviewJobService",
    "ReviewStage",
]
