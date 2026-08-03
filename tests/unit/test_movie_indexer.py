from __future__ import annotations

from pathlib import Path

import pytest

from local_semantic_engine.core.models import EmbeddingBatch
from local_semantic_engine.domains.movies.models import MovieRecord
from local_semantic_engine.domains.movies.representation import with_representation_hash
from local_semantic_engine.ingestion.movies.indexer import build_movie_index
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex


class FakeEmbeddingProvider:
    async def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        vectors = [[float(index + 1), 1.0] for index, _ in enumerate(texts)]
        return EmbeddingBatch(embeddings=vectors, model="test-embedding", dimensions=2)


@pytest.mark.asyncio
async def test_build_movie_index_persists_compatible_index(tmp_path: Path) -> None:
    corpus = tmp_path / "movies.jsonl"
    records = [
        MovieRecord(id="tt0000001", title="One", year=2001, genres=["Drama"]),
        MovieRecord(id="tt0000002", title="Two", year=2002, genres=["Comedy"]),
    ]
    corpus.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )

    result = await build_movie_index(
        corpus_path=corpus,
        output_directory=tmp_path / "index",
        embedding_provider=FakeEmbeddingProvider(),
        embedding_model="test-embedding",
        batch_size=1,
    )

    assert result.record_count == 2
    index = NumpyVectorIndex.load(
        tmp_path / "index",
        embedding_model="test-embedding",
        representation_version="1",
        record_hashes={
            record.id: with_representation_hash(record).content_hash for record in records
        },
    )
    assert index.dimensions == 2
