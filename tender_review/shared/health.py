from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CheckResult:
    name: str
    ready: bool
    detail: str = ""


@runtime_checkable
class ReadinessResult(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def ready(self) -> bool: ...

    @property
    def detail(self) -> str: ...


@runtime_checkable
class ReadinessCheck(Protocol):
    @property
    def name(self) -> str: ...

    def check(self) -> ReadinessResult: ...


class StaticReadinessCheck:
    def __init__(self, name: str, *, ready: bool = True, detail: str = "") -> None:
        self._name = name
        self._ready = ready
        self._detail = detail

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> CheckResult:
        return CheckResult(name=self.name, ready=self._ready, detail=self._detail)
