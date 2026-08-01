from __future__ import annotations

import gzip
from pathlib import Path

from local_semantic_engine.ingestion.movies.imdb import ImdbMovieCandidate, select_most_voted_movies
from local_semantic_engine.ingestion.movies.normalizers import normalize_movie


def _write_gzip(path: Path, content: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(content)


def test_select_most_voted_eligible_movies(tmp_path: Path) -> None:
    basics = tmp_path / "title.basics.tsv.gz"
    ratings = tmp_path / "title.ratings.tsv.gz"
    basics_content = (
        "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres\n"
        "tt001\tmovie\tPopular\tPopular\t0\t2000\t\\N\t100\tDrama\n"
        "tt002\tmovie\tHigher Rated\tHigher Rated\t0\t2001\t\\N\t90\tComedy\n"
        "tt003\tshort\tNot a Movie\tNot a Movie\t0\t2002\t\\N\t10\tDrama\n"
        "tt004\tmovie\tAdult\tAdult\t1\t2003\t\\N\t100\tDrama\n"
    )
    ratings_content = (
        "tconst\taverageRating\tnumVotes\n"
        "tt001\t7.0\t1000\n"
        "tt002\t8.0\t1000\n"
        "tt003\t9.0\t5000\n"
        "tt004\t9.0\t5000\n"
    )
    _write_gzip(basics, basics_content)
    _write_gzip(ratings, ratings_content)

    selected = select_most_voted_movies(basics, ratings, limit=2)

    assert [item.imdb_id for item in selected] == ["tt002", "tt001"]


def test_normalization_preserves_source_provenance() -> None:
    record = normalize_movie(
        ImdbMovieCandidate(
            imdb_id="tt001",
            title="Source Title",
            original_title=None,
            year=2000,
            runtime_minutes=100,
            genres=["Drama"],
            average_rating=7.5,
            vote_count=1000,
        ),
        {"id": 42, "title": "Enriched Title", "overview": "A compact overview."},
    )

    assert record.title == "Enriched Title"
    assert record.field_provenance["title"].source.value == "tmdb"
    assert record.content_hash
