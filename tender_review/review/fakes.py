from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from tender_review.shared.contracts import ensure_call_active

from .models import LlmRequest, LlmResponse, ToolRequest, ToolResult


class FakeLlmProvider:
    def __init__(self, responses: Iterable[str | BaseException] = ()) -> None:
        self._responses = deque(responses)
        self.calls: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        ensure_call_active(request.call)
        self.calls.append(request)
        content = self._responses.popleft() if self._responses else "{}"
        if isinstance(content, BaseException):
            raise content
        return LlmResponse(model="fake-llm", content=content)


class FakeReviewTool:
    def __init__(self, name: str, output_json: str = "{}", version: str = "1") -> None:
        self._name = name
        self.output_json = output_json
        self.version = version
        self.calls: list[ToolRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def execute(self, request: ToolRequest) -> ToolResult:
        ensure_call_active(request.call)
        if request.tool_name != self.name:
            raise ValueError(f"Tool {self.name!r} cannot handle {request.tool_name!r}")
        self.calls.append(request)
        return ToolResult(
            tool_name=self.name,
            tool_version=self.version,
            output_json=self.output_json,
        )
