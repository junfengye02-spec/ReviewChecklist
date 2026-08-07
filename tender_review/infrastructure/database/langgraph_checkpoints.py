from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import func

from .base import NAMING_CONVENTION


_BLOB = LargeBinary().with_variant(mysql.LONGBLOB(), "mysql")
LANGGRAPH_CHECKPOINT_METADATA = MetaData(naming_convention=NAMING_CONVENTION)

langgraph_checkpoints = Table(
    "langgraph_checkpoints",
    LANGGRAPH_CHECKPOINT_METADATA,
    Column("checkpoint_key_sha256", String(64), primary_key=True),
    Column("thread_id", String(128), nullable=False),
    Column("checkpoint_ns", String(255), nullable=False, server_default=""),
    Column("checkpoint_id", String(64), nullable=False),
    Column("parent_checkpoint_id", String(64), nullable=True),
    Column("checkpoint_type", String(255), nullable=False),
    Column("checkpoint_data", _BLOB, nullable=False),
    Column("metadata_type", String(255), nullable=False),
    Column("metadata_data", _BLOB, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Index(
        "ix_langgraph_checkpoints_thread_namespace_id",
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
    ),
)

langgraph_checkpoint_writes = Table(
    "langgraph_checkpoint_writes",
    LANGGRAPH_CHECKPOINT_METADATA,
    Column("write_key_sha256", String(64), primary_key=True),
    Column("thread_id", String(128), nullable=False),
    Column("checkpoint_ns", String(255), nullable=False, server_default=""),
    Column("checkpoint_id", String(64), nullable=False),
    Column("task_id", String(64), nullable=False),
    Column("task_path", String(1024), nullable=False, server_default=""),
    Column("write_index", Integer, nullable=False),
    Column("channel", String(255), nullable=False),
    Column("value_type", String(255), nullable=False),
    Column("value_data", _BLOB, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Index(
        "ix_langgraph_checkpoint_writes_checkpoint",
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
    ),
)


class SqlAlchemyCheckpointSaver(BaseCheckpointSaver[Any]):
    """Synchronous LangGraph saver for the repository's SQLAlchemy stack."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        super().__init__()
        self._sessions = sessions

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns = _scope(config)
        statement = select(langgraph_checkpoints).where(
            langgraph_checkpoints.c.thread_id == thread_id,
            langgraph_checkpoints.c.checkpoint_ns == checkpoint_ns,
        )
        if checkpoint_id := get_checkpoint_id(config):
            statement = statement.where(
                langgraph_checkpoints.c.checkpoint_id == checkpoint_id
            )
        else:
            statement = statement.order_by(
                langgraph_checkpoints.c.checkpoint_id.desc()
            ).limit(1)
        with self._sessions() as session:
            row = session.execute(statement).mappings().first()
            if row is None:
                return None
            return self._to_tuple(session, row)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        statement = select(langgraph_checkpoints)
        if config is not None:
            configurable = config["configurable"]
            statement = statement.where(
                langgraph_checkpoints.c.thread_id
                == configurable["thread_id"]
            )
            if (checkpoint_ns := configurable.get("checkpoint_ns")) is not None:
                statement = statement.where(
                    langgraph_checkpoints.c.checkpoint_ns == checkpoint_ns
                )
            if checkpoint_id := get_checkpoint_id(config):
                statement = statement.where(
                    langgraph_checkpoints.c.checkpoint_id == checkpoint_id
                )
        if before_id := get_checkpoint_id(before) if before else None:
            statement = statement.where(
                langgraph_checkpoints.c.checkpoint_id < before_id
            )
        statement = statement.order_by(
            langgraph_checkpoints.c.checkpoint_id.desc()
        )
        yielded = 0
        with self._sessions() as session:
            rows = tuple(session.execute(statement).mappings())
            for row in rows:
                metadata = self.serde.loads_typed(
                    (row["metadata_type"], row["metadata_data"])
                )
                if filter and not all(
                    metadata.get(key) == value for key, value in filter.items()
                ):
                    continue
                if limit is not None and yielded >= limit:
                    break
                yielded += 1
                yield self._to_tuple(session, row, metadata=metadata)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions
        thread_id, checkpoint_ns = _scope(config)
        checkpoint_id = checkpoint["id"]
        checkpoint_type, checkpoint_data = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_data = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )
        values = {
            "checkpoint_key_sha256": _key_hash(
                thread_id, checkpoint_ns, checkpoint_id
            ),
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": config["configurable"].get(
                "checkpoint_id"
            ),
            "checkpoint_type": checkpoint_type,
            "checkpoint_data": checkpoint_data,
            "metadata_type": metadata_type,
            "metadata_data": metadata_data,
        }
        with self._sessions.begin() as session:
            _upsert(
                session,
                langgraph_checkpoints,
                values,
                key_column="checkpoint_key_sha256",
                update_columns=(
                    "parent_checkpoint_id",
                    "checkpoint_type",
                    "checkpoint_data",
                    "metadata_type",
                    "metadata_data",
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns = _scope(config)
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self._sessions.begin() as session:
            for position, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, position)
                value_type, value_data = self.serde.dumps_typed(value)
                values = {
                    "write_key_sha256": _key_hash(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        str(write_index),
                    ),
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "write_index": write_index,
                    "channel": channel,
                    "value_type": value_type,
                    "value_data": value_data,
                }
                if write_index < 0:
                    _upsert(
                        session,
                        langgraph_checkpoint_writes,
                        values,
                        key_column="write_key_sha256",
                        update_columns=(
                            "task_path",
                            "channel",
                            "value_type",
                            "value_data",
                        ),
                    )
                else:
                    _insert_once(
                        session,
                        langgraph_checkpoint_writes,
                        values,
                        key_column="write_key_sha256",
                    )

    def delete_thread(self, thread_id: str) -> None:
        with self._sessions.begin() as session:
            session.execute(
                delete(langgraph_checkpoint_writes).where(
                    langgraph_checkpoint_writes.c.thread_id == thread_id
                )
            )
            session.execute(
                delete(langgraph_checkpoints).where(
                    langgraph_checkpoints.c.thread_id == thread_id
                )
            )

    def _to_tuple(
        self,
        session: Session,
        row: Any,
        *,
        metadata: CheckpointMetadata | None = None,
    ) -> CheckpointTuple:
        thread_id = row["thread_id"]
        checkpoint_ns = row["checkpoint_ns"]
        checkpoint_id = row["checkpoint_id"]
        write_rows = session.execute(
            select(langgraph_checkpoint_writes)
            .where(
                langgraph_checkpoint_writes.c.thread_id == thread_id,
                langgraph_checkpoint_writes.c.checkpoint_ns
                == checkpoint_ns,
                langgraph_checkpoint_writes.c.checkpoint_id
                == checkpoint_id,
            )
            .order_by(
                langgraph_checkpoint_writes.c.task_id,
                langgraph_checkpoint_writes.c.write_index,
            )
        ).mappings()
        pending_writes = [
            (
                write["task_id"],
                write["channel"],
                self.serde.loads_typed(
                    (write["value_type"], write["value_data"])
                ),
            )
            for write in write_rows
        ]
        parent_id = row["parent_checkpoint_id"]
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=self.serde.loads_typed(
                (row["checkpoint_type"], row["checkpoint_data"])
            ),
            metadata=(
                metadata
                if metadata is not None
                else self.serde.loads_typed(
                    (row["metadata_type"], row["metadata_data"])
                )
            ),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=pending_writes,
        )


def _scope(config: RunnableConfig) -> tuple[str, str]:
    configurable = config["configurable"]
    return configurable["thread_id"], configurable.get("checkpoint_ns", "")


def _key_hash(*parts: str) -> str:
    encoded = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _upsert(
    session: Session,
    table: Table,
    values: dict[str, Any],
    *,
    key_column: str,
    update_columns: Sequence[str],
) -> None:
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        statement = sqlite.insert(table).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[key_column],
            set_={name: statement.excluded[name] for name in update_columns},
        )
    elif dialect == "mysql":
        statement = mysql.insert(table).values(**values)
        statement = statement.on_duplicate_key_update(
            **{name: statement.inserted[name] for name in update_columns}
        )
    else:
        existing = session.scalar(
            select(table.c[key_column]).where(
                table.c[key_column] == values[key_column]
            )
        )
        if existing is None:
            statement = insert(table).values(**values)
        else:
            statement = (
                update(table)
                .where(table.c[key_column] == values[key_column])
                .values(**{name: values[name] for name in update_columns})
            )
    session.execute(statement)


def _insert_once(
    session: Session,
    table: Table,
    values: dict[str, Any],
    *,
    key_column: str,
) -> None:
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        statement = sqlite.insert(table).values(**values).on_conflict_do_nothing(
            index_elements=[key_column]
        )
    elif dialect == "mysql":
        statement = mysql.insert(table).values(**values).prefix_with("IGNORE")
    else:
        existing = session.scalar(
            select(table.c[key_column]).where(
                table.c[key_column] == values[key_column]
            )
        )
        if existing is not None:
            return
        statement = insert(table).values(**values)
    session.execute(statement)
