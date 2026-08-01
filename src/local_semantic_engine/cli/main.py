"""Command-line entry points for local setup and diagnostics."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from local_semantic_engine.config import load_settings
from local_semantic_engine.ingestion.movies.builder import ImdbDatasetDownloader, MovieCorpusBuilder
from local_semantic_engine.ingestion.movies.tmdb import TmdbClient
from local_semantic_engine.storage.database import initialize_database

app = typer.Typer(no_args_is_help=True, help="Local Semantic Engine commands.")
corpus_app = typer.Typer(no_args_is_help=True, help="Explicit corpus setup commands.")
movies_app = typer.Typer(no_args_is_help=True, help="Movie corpus commands.")
app.add_typer(corpus_app, name="corpus")
corpus_app.add_typer(movies_app, name="movies")
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


@movies_app.command("build")
def build_movie_corpus(
    config: ConfigPathOption = None,
    imdb_only: bool = typer.Option(False, "--imdb-only"),
    limit: int = typer.Option(1000, min=1, max=5000),
) -> None:
    """Download/select IMDb movies and optionally enrich the snapshot from TMDB."""

    settings = load_settings(config)
    token = os.environ.get("TMDB_READ_ACCESS_TOKEN", "")
    if not imdb_only and not token:
        raise typer.BadParameter(
            "Set TMDB_READ_ACCESS_TOKEN or use --imdb-only for a metadata-only development corpus."
        )

    async def build() -> object:
        downloader = ImdbDatasetDownloader()
        try:
            basics_path, ratings_path = await downloader.download_required_files(
                settings.storage.raw_data_dir
            )
        finally:
            await downloader.aclose()
        if imdb_only:
            return await MovieCorpusBuilder().build(
                basics_path=basics_path,
                ratings_path=ratings_path,
                output_directory=settings.storage.processed_data_dir,
                limit=limit,
            )
        tmdb = TmdbClient(token)
        try:
            return await MovieCorpusBuilder(enricher=tmdb).build(
                basics_path=basics_path,
                ratings_path=ratings_path,
                output_directory=settings.storage.processed_data_dir,
                limit=limit,
            )
        finally:
            await tmdb.aclose()

    result = asyncio.run(build())
    typer.echo(
        f"Built {len(result.records)} movies; "
        f"{result.enriched_record_count} enriched; "
        f"{result.partial_record_count} partial."
    )
