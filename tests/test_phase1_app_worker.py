import unittest
from datetime import datetime, timezone
from types import MappingProxyType

from tender_review.bootstrap import build_container
from tender_review.jobs.models import JobMessage, JobResult
from tender_review.shared.clock import FixedClock
from tender_review.shared.config import AppSettings
from tender_review.shared.ids import SequentialIdGenerator
from tender_review.worker.cli import main
from tender_review.worker.runner import Worker


class WorkerEntryTests(unittest.TestCase):
    def setUp(self):
        self.repository_container = build_container(
            AppSettings(
                environment="test", log_json=False, worker_poll_interval_seconds=0
            )
        ).with_overrides(
            clock=FixedClock(datetime(2026, 7, 27, tzinfo=timezone.utc)),
            ids=SequentialIdGenerator(["worker-test"]),
        )

    def test_worker_dispatches_one_job_with_explicit_dependencies(self):
        seen = []

        def handler(job):
            seen.append(job)
            return JobResult(output_reference="memory://result", summary="done")

        container = self.repository_container.with_overrides(
            worker_handlers=MappingProxyType({"review": handler})
        )
        job = JobMessage(
            job_id="job-1",
            job_type="review",
            input_reference="memory://document",
        )
        container.job_repository.enqueue(job)
        worker = Worker(
            worker_id="worker-test",
            repository=container.job_repository,
            leases=container.lease_manager,
            handlers=container.worker_handlers,
            clock=container.clock,
            poll_interval_seconds=0,
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(seen, [job])
        self.assertEqual(
            container.job_repository.completed["job-1"].output_reference,
            "memory://result",
        )
        self.assertFalse(worker.run_once())

    def test_unknown_job_type_is_mapped_to_permanent_failure(self):
        job = JobMessage(
            job_id="job-unknown",
            job_type="unknown",
            input_reference="memory://document",
        )
        self.repository_container.job_repository.enqueue(job)
        worker = Worker(
            worker_id="worker-test",
            repository=self.repository_container.job_repository,
            leases=self.repository_container.lease_manager,
            handlers=self.repository_container.worker_handlers,
            clock=self.repository_container.clock,
            poll_interval_seconds=0,
        )

        self.assertTrue(worker.run_once())
        failure = self.repository_container.job_repository.failed["job-unknown"]
        self.assertEqual(failure.code, "unknown_job_type")
        self.assertFalse(failure.retryable)

    def test_worker_cli_once_exits_cleanly_with_empty_fake_queue(self):
        self.assertEqual(main(["--once"], container=self.repository_container), 0)


if __name__ == "__main__":
    unittest.main()
