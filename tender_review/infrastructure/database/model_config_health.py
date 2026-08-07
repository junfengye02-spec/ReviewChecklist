from __future__ import annotations

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.infrastructure.health import HealthStatus

from .models import ModelConfig


class ModelConfigHealthAdapter:
    """Readiness check for the non-secret model identity pinned by execution specs."""

    name = "review_model_config"

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        model_config_id: str,
        model_config_hash: str,
    ) -> None:
        self._sessions = sessions
        self._model_config_id = model_config_id
        self._model_config_hash = model_config_hash

    def check(self) -> HealthStatus:
        try:
            with self._sessions() as session:
                row = session.get(ModelConfig, self._model_config_id)
                if row is None:
                    return HealthStatus(
                        service=self.name,
                        healthy=False,
                        detail="configured model identity is not registered",
                    )
                if row.config_hash != self._model_config_hash:
                    return HealthStatus(
                        service=self.name,
                        healthy=False,
                        detail="registered model identity hash does not match runtime",
                    )
        except SQLAlchemyError:
            return HealthStatus(
                service=self.name,
                healthy=False,
                detail="model identity could not be verified",
            )
        return HealthStatus(
            service=self.name,
            healthy=True,
            detail="model identity and canonical hash match",
        )
