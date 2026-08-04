"""Build a reproducible local vector index from the frozen movie corpus."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from local_semantic_engine.core.errors import CorpusNotReadyError
from local_semantic_engine.domains.movies.models import MovieRecord
from local_semantic_engine.domains.movies.representation import (
    REPRESENTATION_VERSION,
    render_movie_search_text,
    with_representation_hash,
)
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex, new_manifest


class EmbeddingProvider(Protocol):
    """The narrow embedding capability required by an index build."""

    async def embed_texts(self, texts: Sequence[str]): ...


@dataclass(frozen=True, slots=True)
class MovieIndexBuildResult:
    record_count: int
    dimensions: int
    output_directory: Path


async def build_movie_index(
    *,
    corpus_path: Path,
    output_directory: Path,
    embedding_provider: EmbeddingProvider,
    embedding_model: str,
    batch_size: int = 16,
) -> MovieIndexBuildResult:
    """Embed the complete corpus and replace its index only after success."""

    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive.")
    movies = load_movie_corpus(corpus_path)
    rendered = [with_representation_hash(movie) for movie in movies]
    item_ids = [movie.id for movie in rendered]
    if len(set(item_ids)) != len(item_ids):
        raise CorpusNotReadyError("The movie corpus contains duplicate IMDb IDs.")

    vectors: list[list[float]] = []
    for start in range(0, len(rendered), batch_size):
        texts = [render_movie_search_text(movie) for movie in rendered[start : start + batch_size]]
        batch = await embedding_provider.embed_texts(texts)
        if len(batch.embeddings) != len(texts):
            raise CorpusNotReadyError("The embedding provider returned an incomplete batch.")
        vectors.extend(batch.embeddings)
        await asyncio.sleep(0)

    index = NumpyVectorIndex(item_ids, np.asarray(vectors, dtype=np.float32))
    index.save(
        output_directory,
        new_manifest(
            embedding_model=embedding_model,
            dimensions=index.dimensions,
            representation_version=REPRESENTATION_VERSION,
            record_hashes={movie.id: movie.content_hash for movie in rendered},
        ),
    )
    return MovieIndexBuildResult(
        record_count=len(rendered), dimensions=index.dimensions, output_directory=output_directory
    )


def load_movie_corpus(corpus_path: Path) -> list[MovieRecord]:
    """Load the frozen corpus with a controlled readiness error."""

    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise CorpusNotReadyError(
            "The movie corpus is unavailable. Run `lse corpus movies build` first."
        ) from exc
    movies = [MovieRecord.model_validate_json(line) for line in lines if line.strip()]
    if not movies:
        raise CorpusNotReadyError("The movie corpus is empty.")
    return movies
