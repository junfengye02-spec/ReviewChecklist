from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select

from tender_review.infrastructure.database import create_session_factory
from tender_review.infrastructure.database.langgraph_checkpoints import (
    LANGGRAPH_CHECKPOINT_METADATA,
    SqlAlchemyCheckpointSaver,
    langgraph_checkpoint_writes,
    langgraph_checkpoints,
)
from tender_review.review.langgraph_workflow import (
    EXTRACT_STRUCTURED_FIELDS,
    LangGraphReviewWorkflow,
)
from tender_review.review.models import ReviewGraphNode
from tender_review.review.workflow import SingleReviewWorkflow
from tender_review.shared.ids import SequentialIdGenerator

from test_phase5_review import review_request, text_extraction


class NoRepeatLlm:
    def complete(self, request):
        raise AssertionError("completed extraction must not be called again")


class SqlAlchemyCheckpointSaverTests(unittest.TestCase):
    def test_fresh_workflow_instance_resumes_from_sql_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "checkpoints.db"
            engine = create_engine(f"sqlite:///{database_path.as_posix()}")
            LANGGRAPH_CHECKPOINT_METADATA.create_all(engine)
            sessions = create_session_factory(engine)
            request, llm = review_request(text_extraction())
            saver = SqlAlchemyCheckpointSaver(sessions)
            first = LangGraphReviewWorkflow(
                SingleReviewWorkflow(
                    llm,
                    id_generator=SequentialIdGenerator(("unused-id",)),
                ),
                checkpointer=saver,
            )

            paused = first.run(
                request,
                interrupt_after=(EXTRACT_STRUCTURED_FIELDS,),
            )
            self.assertEqual(paused.node, ReviewGraphNode.COMPARISON)
            self.assertEqual(len(llm.calls), 1)

            restored_saver = SqlAlchemyCheckpointSaver(sessions)
            restored = LangGraphReviewWorkflow(
                SingleReviewWorkflow(
                    NoRepeatLlm(),
                    id_generator=SequentialIdGenerator(("finding-1",)),
                ),
                checkpointer=restored_saver,
            )
            completed = restored.run(request)

            self.assertEqual(completed.node, ReviewGraphNode.DONE)
            self.assertEqual(completed.finding.finding_id, "finding-1")
            checkpoints = tuple(
                restored_saver.list(
                    {
                        "configurable": {
                            "thread_id": request.review_job_id,
                        }
                    }
                )
            )
            self.assertGreaterEqual(len(checkpoints), 2)
            self.assertTrue(any(item.pending_writes for item in checkpoints))
            with engine.connect() as connection:
                checkpoint_count = len(
                    connection.execute(select(langgraph_checkpoints)).all()
                )
                write_count = len(
                    connection.execute(
                        select(langgraph_checkpoint_writes)
                    ).all()
                )
            self.assertGreater(checkpoint_count, 0)
            self.assertGreater(write_count, 0)

            restored_saver.delete_thread(request.review_job_id)
            self.assertIsNone(
                restored_saver.get_tuple(
                    {
                        "configurable": {
                            "thread_id": request.review_job_id,
                        }
                    }
                )
            )
            engine.dispose()


class LangGraphCheckpointMigrationTests(unittest.TestCase):
    def test_migration_upgrades_and_downgrades_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "migration.db"
            database_url = f"sqlite:///{database_path.as_posix()}"
            config = Config(str(Path("alembic.ini").resolve()))
            config.set_main_option("sqlalchemy.url", database_url)

            command.upgrade(config, "head")
            engine = create_engine(database_url)
            tables = set(inspect(engine).get_table_names())
            self.assertIn("langgraph_checkpoints", tables)
            self.assertIn("langgraph_checkpoint_writes", tables)
            columns = {
                item["name"]
                for item in inspect(engine).get_columns(
                    "langgraph_checkpoints"
                )
            }
            self.assertTrue(
                {
                    "thread_id",
                    "checkpoint_ns",
                    "checkpoint_id",
                    "checkpoint_data",
                    "metadata_data",
                }.issubset(columns)
            )
            engine.dispose()

            command.downgrade(config, "e5c8a7b9d204")
            engine = create_engine(database_url)
            tables = set(inspect(engine).get_table_names())
            self.assertNotIn("langgraph_checkpoints", tables)
            self.assertNotIn("langgraph_checkpoint_writes", tables)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
