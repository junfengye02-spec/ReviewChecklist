from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol, runtime_checkable
from uuid import uuid4


@runtime_checkable
class IdGenerator(Protocol):
    def new(self) -> str: ...


class UuidGenerator:
    def new(self) -> str:
        return str(uuid4())


class SequentialIdGenerator:
    def __init__(
        self, values: Iterable[str] | None = None, *, prefix: str = "id"
    ) -> None:
        self._values = deque(values or ())
        self._prefix = prefix
        self._counter = 0

    def new(self) -> str:
        if self._values:
            return self._values.popleft()
        self._counter += 1
        return f"{self._prefix}-{self._counter}"
