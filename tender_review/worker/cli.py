from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Sequence

from tender_review.bootstrap import ApplicationContainer, build_container
from tender_review.shared.config import AppSettings
from tender_review.shared.logging import configure_logging

from .runner import Worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tender review background worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit, including when the queue is empty",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Stop after this many polling iterations",
    )
    parser.add_argument("--worker-id")
    parser.add_argument("--poll-interval-seconds", type=float)
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help="Check configured dependencies once and exit",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    container: ApplicationContainer | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    resolved_container = container or build_container(AppSettings.from_env())
    settings = resolved_container.settings
    configure_logging(settings.log_level, json_output=settings.log_json)
    if args.check_readiness:
        try:
            return _check_readiness(resolved_container)
        finally:
            resolved_container.close()
    worker_id = (
        args.worker_id or settings.worker_id or f"worker-{resolved_container.ids.new()}"
    )
    poll_interval = (
        settings.worker_poll_interval_seconds
        if args.poll_interval_seconds is None
        else args.poll_interval_seconds
    )
    worker = Worker(
        worker_id=worker_id,
        repository=resolved_container.job_repository,
        leases=resolved_container.lease_manager,
        handlers=resolved_container.worker_handlers,
        clock=resolved_container.clock,
        lease_seconds=settings.worker_lease_seconds,
        poll_interval_seconds=poll_interval,
    )
    logger = logging.getLogger("tender_review.worker")
    logger.info(
        "Worker started",
        extra={
            "event": "worker.started",
            "worker_id": worker_id,
            "process_id": os.getpid(),
            "process_started_ns": time.time_ns(),
        },
    )
    try:
        if args.once:
            worker.run_once()
        else:
            worker.run_forever(max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        logger.info(
            "Worker stopped",
            extra={"event": "worker.stopped", "worker_id": worker_id},
        )
    finally:
        resolved_container.close()
    return 0


def _check_readiness(container: ApplicationContainer) -> int:
    logger = logging.getLogger("tender_review.worker")
    ready = True
    for check in container.readiness_checks:
        try:
            result = check.check()
            component_ready = bool(result.ready)
        except Exception as exc:
            component_ready = False
            logger.warning(
                "Worker readiness check failed",
                extra={
                    "event": "worker.readiness_failed",
                    "check": check.name,
                    "error_type": type(exc).__name__,
                },
            )
        ready = ready and component_ready
    return 0 if ready else 1
