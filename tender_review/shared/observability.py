from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Bounded identifiers shared by logs, audit records, and metric events."""

    job_id: str | None = None
    thread_id: str | None = None
    checkpoint_id: str | None = None
    call_id: str | None = None
    rule_version: str | None = None
    dataset_version: str | None = None
    model_config: str | None = None

    def with_checkpoint(self, checkpoint_id: str | None) -> "CorrelationContext":
        return replace(self, checkpoint_id=checkpoint_id)

    def fields(self) -> dict[str, str | None]:
        return {
            "job_id": self.job_id,
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "call_id": self.call_id,
            "rule_version": self.rule_version,
            "dataset_version": self.dataset_version,
            "model_config": self.model_config,
        }


def log_event(
    logger: logging.Logger,
    level: int,
    *,
    event: str,
    message: str,
    context: CorrelationContext | None = None,
    **fields: Any,
) -> None:
    extra = (context or CorrelationContext()).fields()
    extra.update({"event": event, **fields})
    logger.log(level, message, extra=extra)


def record_metric(
    logger: logging.Logger,
    *,
    name: str,
    value: int | float | None,
    unit: str,
    source: str,
    context: CorrelationContext | None = None,
    status: Literal["collected", "not_collected"] = "collected",
    **fields: Any,
) -> None:
    log_event(
        logger,
        logging.INFO,
        event="metric.observation",
        message="Operational metric observed",
        context=context,
        metric_name=name,
        metric_value=value,
        metric_unit=unit,
        metric_source=source,
        metric_status=status,
        **fields,
    )
