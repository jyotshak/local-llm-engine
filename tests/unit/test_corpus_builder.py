from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from local_semantic_engine.ingestion.movies.builder import MovieCorpusBuilder


class FakeEnricher:
    async def enrich_movie(
        self, imdb_id: str, *, include_reviews: bool = False
    ) -> dict[str, object] | None:
        return {"id": 1, "title": f"Enriched {imdb_id}", "overview": "An overview."}


def _write_gzip(path: Path, content: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(content)


@pytest.mark.asyncio
async def test_builder_writes_snapshot_and_provenance_manifest(tmp_path: Path) -> None:
    basics = tmp_path / "title.basics.tsv.gz"
    ratings = tmp_path / "title.ratings.tsv.gz"
    _write_gzip(
        basics,
        "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres\n"
        "tt001\tmovie\tOne\tOne\t0\t2000\t\\N\t100\tDrama\n"
        "tt002\tmovie\tTwo\tTwo\t0\t2001\t\\N\t90\tComedy\n",
    )
    _write_gzip(
        ratings,
        "tconst\taverageRating\tnumVotes\ntt001\t7.0\t1000\ntt002\t8.0\t2000\n",
    )

    result = await MovieCorpusBuilder(enricher=FakeEnricher()).build(
        basics_path=basics,
        ratings_path=ratings,
        output_directory=tmp_path / "processed",
        limit=2,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.enriched_record_count == 2
    assert result.partial_record_count == 0
    assert manifest["record_count"] == 2
    assert records[0]["title"] == "Enriched tt002"
