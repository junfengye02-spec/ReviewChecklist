from __future__ import annotations

from threading import Lock
from typing import Protocol, runtime_checkable

from tender_review.shared.errors import RetryableError


class InjectedFault(RetryableError):
    default_code = "injected_fault"

    def __init__(self, point: str) -> None:
        super().__init__(
            "A configured local reliability drill interrupted execution",
            code=self.default_code,
            details={"fault_point": point},
        )
        self.point = point


@runtime_checkable
class FaultInjector(Protocol):
    def is_armed(self, point: str) -> bool: ...

    def trip(self, point: str) -> None: ...


class DisabledFaultInjector:
    def is_armed(self, point: str) -> bool:
        del point
        return False

    def trip(self, point: str) -> None:
        del point


class OneShotFaultInjector:
    """In-process drill helper; a named fault is consumed before it is raised."""

    def __init__(self, *points: str) -> None:
        normalized = {point.strip() for point in points if point.strip()}
        if len(normalized) != len(points):
            raise ValueError("fault points must be unique and non-blank")
        self._armed = normalized
        self._lock = Lock()

    def is_armed(self, point: str) -> bool:
        with self._lock:
            return point in self._armed

    def trip(self, point: str) -> None:
        with self._lock:
            if point not in self._armed:
                return
            self._armed.remove(point)
        raise InjectedFault(point)
