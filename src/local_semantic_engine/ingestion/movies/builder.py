"""Explicit, reproducible construction of the local movie corpus."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

from local_semantic_engine.domains.movies.models import MovieRecord
from local_semantic_engine.ingestion.movies.imdb import ImdbMovieCandidate, select_most_voted_movies
from local_semantic_engine.ingestion.movies.normalizers import normalize_movie

IMDB_DATASET_BASE_URL = "https://datasets.imdbws.com"
IMDB_DATASET_FILES = ("title.basics.tsv.gz", "title.ratings.tsv.gz")


class MovieEnricher(Protocol):
    async def enrich_movie(
        self, imdb_id: str, *, include_reviews: bool = False
    ) -> dict[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    records: list[MovieRecord]
    partial_record_count: int
    enriched_record_count: int
    output_path: Path
    manifest_path: Path


class ImdbDatasetDownloader:
    """Download only the official IMDb TSV files required by Version 1."""

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def download_required_files(self, destination: Path) -> tuple[Path, Path]:
        destination.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for filename in IMDB_DATASET_FILES:
            path = destination / filename
            if not path.exists():
                await self._download(f"{IMDB_DATASET_BASE_URL}/{filename}", path)
            paths.append(path)
        return paths[0], paths[1]

    async def _download(self, url: str, destination: Path) -> None:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)


class MovieCorpusBuilder:
    """Select IMDb movies, optionally enrich them, and write a local snapshot."""

    def __init__(
        self,
        *,
        enricher: MovieEnricher | None = None,
        enrichment_concurrency: int = 4,
    ) -> None:
        if enrichment_concurrency < 1:
            raise ValueError("Enrichment concurrency must be positive.")
        self._enricher = enricher
        self._enrichment_concurrency = enrichment_concurrency

    async def build(
        self,
        *,
        basics_path: Path,
        ratings_path: Path,
        output_directory: Path,
        limit: int = 1000,
    ) -> CorpusBuildResult:
        candidates = select_most_voted_movies(basics_path, ratings_path, limit=limit)
        if len(candidates) != limit:
            raise ValueError(
                f"IMDb selection returned {len(candidates)} records, expected {limit}."
            )
        cache_directory = output_directory / "tmdb_cache"
        semaphore = asyncio.Semaphore(self._enrichment_concurrency)

        async def build_record(
            candidate_id: str,
            candidate: ImdbMovieCandidate,
        ) -> tuple[MovieRecord, bool]:
            async with semaphore:
                enrichment = await self._load_or_enrich(candidate_id, cache_directory)
            return normalize_movie(candidate, enrichment), enrichment is not None

        built_records = await asyncio.gather(
            *(build_record(candidate.imdb_id, candidate) for candidate in candidates)
        )
        records = [record for record, _ in built_records]
        enriched_record_count = sum(enriched for _, enriched in built_records)

        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / "movies.jsonl"
        manifest_path = output_directory / "corpus_manifest.json"
        _atomic_text_write(
            output_path,
            "".join(record.model_dump_json() + "\n" for record in records),
        )
        partial_record_count = len(records) - enriched_record_count
        manifest = {
            "schema_version": "1",
            "created_at": datetime.now(UTC).isoformat(),
            "selection": {
                "name": "IMDb Most-Voted 1000",
                "limit": limit,
                "sort": ["numVotes descending", "averageRating descending", "tconst ascending"],
            },
            "source_files": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in (basics_path, ratings_path)
            },
            "record_count": len(records),
            "enriched_record_count": enriched_record_count,
            "partial_record_count": partial_record_count,
            "tmdb_enrichment_enabled": self._enricher is not None,
        }
        _atomic_text_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        return CorpusBuildResult(
            records=records,
            partial_record_count=partial_record_count,
            enriched_record_count=enriched_record_count,
            output_path=output_path,
            manifest_path=manifest_path,
        )

    async def _load_or_enrich(
        self,
        imdb_id: str,
        cache_directory: Path,
    ) -> dict[str, object] | None:
        if self._enricher is None:
            return None
        cache_path = cache_directory / f"{imdb_id}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        enrichment = await self._enricher.enrich_movie(imdb_id)
        if enrichment is not None:
            cache_directory.mkdir(parents=True, exist_ok=True)
            _atomic_text_write(cache_path, json.dumps(enrichment, indent=2, sort_keys=True))
        return enrichment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
