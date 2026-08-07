from __future__ import annotations

from tender_review.shared.contracts import ensure_call_active

from .models import OcrRequest, OcrResult, sha256_text


class FakeOcrProvider:
    name = "fake-ocr"
    version = "1"

    def __init__(
        self,
        responses: dict[int, str | OcrResult | Exception] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[OcrRequest] = []

    def recognize(self, request: OcrRequest) -> OcrResult:
        ensure_call_active(request.call)
        self.calls.append(request)
        response = self.responses.get(request.page_number, "")
        if isinstance(response, Exception):
            raise response
        if isinstance(response, OcrResult):
            return response
        return OcrResult(
            provider=self.name,
            provider_version=self.version,
            text=response,
            text_sha256=sha256_text(response),
            confidence_lower=0.8 if response else None,
            confidence_upper=0.9 if response else None,
        )


class UnavailableOcrProvider:
    """Explicit offline OCR adapter used by audits without an OCR deployment."""

    name = "unavailable-ocr"
    version = "1"

    def __init__(self, reason: str = "no OCR provider is configured") -> None:
        self.reason = reason
        self.calls: list[OcrRequest] = []

    def recognize(self, request: OcrRequest) -> OcrResult:
        ensure_call_active(request.call)
        self.calls.append(request)
        raise RuntimeError(self.reason)
