from __future__ import annotations

import sqlite3
from pathlib import Path

from local_semantic_engine.storage.database import initialize_database


def test_database_initialization_applies_migrations(tmp_path: Path) -> None:
    database_path = tmp_path / "state" / "engine.sqlite3"

    initialize_database(database_path)
    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        migrations = connection.execute("SELECT version FROM schema_migrations").fetchall()

    assert {"schema_migrations", "corpus_builds", "movie_records", "request_traces"} <= tables
    assert migrations == [(1,)]
