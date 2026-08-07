from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    LlmRequest,
    LlmResponse,
    ReviewGraphState,
    ReviewRequest,
    ToolRequest,
    ToolResult,
)


@runtime_checkable
class ReviewWorkflow(Protocol):
    def run(self, request: ReviewRequest) -> ReviewGraphState: ...


@runtime_checkable
class LlmProvider(Protocol):
    def complete(self, request: LlmRequest) -> LlmResponse: ...


@runtime_checkable
class ReviewTool(Protocol):
    @property
    def name(self) -> str: ...

    def execute(self, request: ToolRequest) -> ToolResult: ...
