from __future__ import annotations

from local_semantic_engine.domains.movies.models import MovieRecord


def test_movie_record_collections_are_not_shared() -> None:
    first = MovieRecord(id="tt001", title="First")
    second = MovieRecord(id="tt002", title="Second")

    first.genres.append("Drama")

    assert second.genres == []
