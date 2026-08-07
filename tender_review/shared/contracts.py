from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)


class CallContext(ContractModel):
    call_id: str = Field(min_length=1, max_length=128)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=1, ge=1)
    cancelled: bool = False


def ensure_call_active(call: CallContext) -> None:
    if call.cancelled:
        from .errors import CancelledError

        raise CancelledError(
            f"Call {call.call_id!r} was cancelled",
            code="call_cancelled",
            details={"call_id": call.call_id},
        )
