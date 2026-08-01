from __future__ import annotations

from pathlib import Path

from local_semantic_engine.cli.main import _read_local_dotenv


def test_dotenv_parser_reads_tmdb_values_without_exposing_them(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# local only\nTMDB_API_KEY='test-key'\n", encoding="utf-8")

    values = _read_local_dotenv(path)

    assert values["TMDB_API_KEY"] == "test-key"
