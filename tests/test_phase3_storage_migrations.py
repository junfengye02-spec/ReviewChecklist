from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from tender_review.config import PROJECT_DIR


class Phase3MigrationTests(unittest.TestCase):
    def test_upgrade_downgrade_round_trip_includes_latest_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "migration.db"
            config = Config(str(PROJECT_DIR / "alembic.ini"), stdout=io.StringIO())
            config.set_main_option("script_location", str(PROJECT_DIR / "alembic"))
            config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
            command.upgrade(config, "head")
            engine = create_engine(f"sqlite:///{database_path}")
            try:
                with engine.connect() as connection:
                    self.assertEqual(
                        connection.scalar(text("SELECT version_num FROM alembic_version")),
                        "d5b0f6a8c214",
                    )
            finally:
                engine.dispose()
            command.downgrade(config, "7e2c4f9a1b35")
            engine = create_engine(f"sqlite:///{database_path}")
            try:
                with engine.connect() as connection:
                    self.assertEqual(
                        connection.scalar(text("SELECT version_num FROM alembic_version")),
                        "7e2c4f9a1b35",
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
