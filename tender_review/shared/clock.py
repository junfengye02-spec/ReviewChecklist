from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock:
    def __init__(self, value: datetime) -> None:
        if value.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self.value = value

    def now(self) -> datetime:
        return self.value
