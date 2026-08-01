"""SQLite initialization and low-level connection helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS corpus_builds (
            build_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            manifest_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS movie_records (
            id TEXT PRIMARY KEY,
            build_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            FOREIGN KEY(build_id) REFERENCES corpus_builds(build_id)
        );
        CREATE TABLE IF NOT EXISTS request_traces (
            trace_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            pipeline_name TEXT NOT NULL,
            trace_json TEXT NOT NULL
        );
        """,
    ),
)


@contextmanager
def connect(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection configured for local concurrent reads."""

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: Path) -> None:
    """Create the database and apply the project's numbered migrations."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        applied = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
        }
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            connection.executescript(sql)
            connection.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
