from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Framework-neutral health result returned by infrastructure adapters."""

    service: str
    healthy: bool
    detail: str
    schema_version: str = "1"

    @property
    def name(self) -> str:
        return self.service

    @property
    def ready(self) -> bool:
        return self.healthy
