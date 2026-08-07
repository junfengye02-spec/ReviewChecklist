from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.infrastructure.health import HealthStatus


class DatabaseConfigurationError(ValueError):
    pass


def create_database_engine(
    database_url: str,
    *,
    echo: bool = False,
    pool_pre_ping: bool = True,
    pool_recycle_seconds: int = 1800,
    connect_timeout_seconds: int = 5,
) -> Engine:
    """Create an engine without reading process-global configuration."""

    if not database_url.strip():
        raise DatabaseConfigurationError("database_url must not be empty")
    if connect_timeout_seconds <= 0:
        raise DatabaseConfigurationError("connect_timeout_seconds must be positive")

    url = make_url(database_url)
    connect_args: dict[str, object] = {}
    engine_options: dict[str, object] = {
        "echo": echo,
        "pool_pre_ping": pool_pre_ping,
    }
    if url.get_backend_name() == "mysql":
        connect_args["connect_timeout"] = connect_timeout_seconds
        engine_options["pool_recycle"] = pool_recycle_seconds
    elif url.get_backend_name() == "sqlite":
        connect_args["timeout"] = float(connect_timeout_seconds)

    return create_engine(url, connect_args=connect_args, **engine_options)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine, class_=Session, expire_on_commit=False, autoflush=False
    )


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Own one transaction and always release the session."""

    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


@dataclass(frozen=True, slots=True)
class DatabaseHealthAdapter:
    engine: Engine

    @property
    def name(self) -> str:
        return "database"

    def check(self) -> HealthStatus:
        try:
            with self.engine.connect() as connection:
                value = connection.scalar(text("SELECT 1"))
        except SQLAlchemyError as exc:
            return HealthStatus(
                service=self.name,
                healthy=False,
                detail=f"database unavailable: {type(exc).__name__}",
            )
        return HealthStatus(
            service=self.name,
            healthy=value == 1,
            detail="ok" if value == 1 else "unexpected health query result",
        )
