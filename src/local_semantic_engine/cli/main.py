"""Command-line entry points for local setup and diagnostics."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from local_semantic_engine.config import load_settings
from local_semantic_engine.domains.movies.models import MovieRecommendationRequest
from local_semantic_engine.domains.movies.recommender import MovieRecommender
from local_semantic_engine.domains.movies.representation import (
    REPRESENTATION_VERSION,
    with_representation_hash,
)
from local_semantic_engine.embeddings.ollama import OllamaEmbeddingProvider
from local_semantic_engine.ingestion.movies.builder import ImdbDatasetDownloader, MovieCorpusBuilder
from local_semantic_engine.ingestion.movies.indexer import build_movie_index, load_movie_corpus
from local_semantic_engine.ingestion.movies.tmdb import TmdbClient
from local_semantic_engine.llm.ollama import OllamaClient
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex
from local_semantic_engine.storage.database import initialize_database

app = typer.Typer(no_args_is_help=True, help="Local Semantic Engine commands.")
corpus_app = typer.Typer(no_args_is_help=True, help="Explicit corpus setup commands.")
movies_app = typer.Typer(no_args_is_help=True, help="Movie corpus commands.")
index_app = typer.Typer(no_args_is_help=True, help="Explicit local index build commands.")
search_app = typer.Typer(no_args_is_help=True, help="Local semantic search commands.")
recommend_app = typer.Typer(no_args_is_help=True, help="Local movie recommendation commands.")
app.add_typer(corpus_app, name="corpus")
corpus_app.add_typer(movies_app, name="movies")
app.add_typer(index_app, name="index")
app.add_typer(search_app, name="search")
app.add_typer(recommend_app, name="recommend")
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
    token, api_key = _tmdb_credentials()
    if not imdb_only and not (token or api_key):
        raise typer.BadParameter(
            "Set TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY, or use --imdb-only for development."
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
        tmdb = TmdbClient(token, api_key=api_key)
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


@index_app.command("movies")
def build_movie_vector_index(
    config: ConfigPathOption = None,
    batch_size: int = typer.Option(16, min=1, max=128),
) -> None:
    """Build the local movie vector index from the frozen JSONL corpus."""

    settings = load_settings(config)

    async def build() -> object:
        provider = OllamaEmbeddingProvider(settings.ollama)
        try:
            return await build_movie_index(
                corpus_path=settings.storage.processed_data_dir / "movies.jsonl",
                output_directory=settings.storage.index_data_dir / "movies",
                embedding_provider=provider,
                embedding_model=settings.ollama.embedding_model,
                batch_size=batch_size,
            )
        finally:
            await provider.aclose()

    result = asyncio.run(build())
    typer.echo(f"Indexed {result.record_count} movies ({result.dimensions} dimensions).")


@search_app.command("movies")
def search_movies(
    query: str = typer.Argument(..., min=1, help="Natural-language movie request."),
    count: int = typer.Option(5, min=1, max=20),
    config: ConfigPathOption = None,
) -> None:
    """Inspect raw semantic matches from the local movie index.

    This is a retrieval diagnostic, not yet the final constrained recommender.
    """

    settings = load_settings(config)
    movies = load_movie_corpus(settings.storage.processed_data_dir / "movies.jsonl")
    hashed_movies = [with_representation_hash(movie) for movie in movies]
    index = NumpyVectorIndex.load(
        settings.storage.index_data_dir / "movies",
        embedding_model=settings.ollama.embedding_model,
        representation_version=REPRESENTATION_VERSION,
        record_hashes={movie.id: movie.content_hash for movie in hashed_movies},
    )

    async def search() -> list[float]:
        provider = OllamaEmbeddingProvider(settings.ollama)
        try:
            return await provider.embed_query(query)
        finally:
            await provider.aclose()

    vector = asyncio.run(search())
    movies_by_id = {movie.id: movie for movie in movies}
    matches = []
    for result in index.search(vector, top_k=count):
        movie = movies_by_id[result.item_id]
        matches.append(
            {
                "id": movie.id,
                "title": movie.title,
                "year": movie.year,
                "runtime_minutes": movie.runtime_minutes,
                "genres": movie.genres,
                "semantic_score": round(result.score, 4),
            }
        )
    typer.echo(json.dumps({"query": query, "matches": matches}, indent=2))


@recommend_app.command("movies")
def recommend_movies(
    query: str = typer.Argument(..., min=1, help="Natural-language movie request."),
    count: int = typer.Option(5, min=1, max=10),
    profile: str = typer.Option("balanced", case_sensitive=False),
    config: ConfigPathOption = None,
) -> None:
    """Recommend movies with deterministic hard-constraint enforcement."""

    settings = load_settings(config)
    movies = load_movie_corpus(settings.storage.processed_data_dir / "movies.jsonl")
    hashed_movies = [with_representation_hash(movie) for movie in movies]
    index = NumpyVectorIndex.load(
        settings.storage.index_data_dir / "movies",
        embedding_model=settings.ollama.embedding_model,
        representation_version=REPRESENTATION_VERSION,
        record_hashes={movie.id: movie.content_hash for movie in hashed_movies},
    )
    request = MovieRecommendationRequest(query=query, count=count, profile=profile.lower())

    async def recommend() -> object:
        generator = OllamaClient(settings.ollama)
        embedder = OllamaEmbeddingProvider(settings.ollama)
        try:
            recommender = MovieRecommender(
                settings=settings,
                generator=generator,
                embedder=embedder,
                index=index,
                movies_by_id={movie.id: movie for movie in movies},
            )
            return await recommender.recommend(request)
        finally:
            await generator.aclose()
            await embedder.aclose()

    response = asyncio.run(recommend())
    typer.echo(response.model_dump_json(indent=2))


def _tmdb_credentials() -> tuple[str | None, str | None]:
    dotenv_values = _read_local_dotenv(Path(".env"))
    token = os.environ.get("TMDB_READ_ACCESS_TOKEN") or dotenv_values.get("TMDB_READ_ACCESS_TOKEN")
    api_key = os.environ.get("TMDB_API_KEY") or dotenv_values.get("TMDB_API_KEY")
    return token, api_key


def _read_local_dotenv(path: Path) -> dict[str, str]:
    """Read simple local secrets without printing, logging, or committing them."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"'")
    return values
