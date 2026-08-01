"""Command-line entry points for local setup and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from local_semantic_engine.config import load_settings
from local_semantic_engine.storage.database import initialize_database

app = typer.Typer(no_args_is_help=True, help="Local Semantic Engine commands.")
ConfigPathOption = Annotated[
    Path | None,
    typer.Option("--config", exists=True, readable=True, help="Path to an optional TOML file."),
]


def _settings_summary(config_path: Path | None) -> dict[str, object]:
    settings = load_settings(config_path)
    return {
        "ollama_base_url": settings.ollama.base_url,
        "generation_model_configured": bool(settings.ollama.generation_model),
        "embedding_model": settings.ollama.embedding_model,
        "database_path": str(settings.storage.database_path),
        "raw_data_dir": str(settings.storage.raw_data_dir),
        "processed_data_dir": str(settings.storage.processed_data_dir),
        "index_data_dir": str(settings.storage.index_data_dir),
        "api_host": settings.api.host,
        "api_port": settings.api.port,
        "profiles": sorted(settings.profiles),
    }


@app.command()
def doctor(
    config: ConfigPathOption = None,
) -> None:
    """Print safe local configuration diagnostics without downloading anything."""

    typer.echo(json.dumps(_settings_summary(config), indent=2, sort_keys=True))


@app.command("init-db")
def init_db(
    config: ConfigPathOption = None,
) -> None:
    """Create or upgrade the local SQLite database."""

    settings = load_settings(config)
    initialize_database(settings.storage.database_path)
    typer.echo(f"Initialized {settings.storage.database_path}")
