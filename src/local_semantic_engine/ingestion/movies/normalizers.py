"""Convert source data into validated local movie records."""

from __future__ import annotations

from collections.abc import Mapping

from local_semantic_engine.domains.movies.models import (
    FieldProvenance,
    FieldSource,
    MovieRecord,
    SourceReference,
)
from local_semantic_engine.domains.movies.representation import with_representation_hash
from local_semantic_engine.ingestion.movies.imdb import ImdbMovieCandidate


def normalize_movie(
    candidate: ImdbMovieCandidate,
    tmdb: Mapping[str, object] | None = None,
) -> MovieRecord:
    """Build a record without guessing when setup-time enrichment is unavailable."""

    tmdb = tmdb or {}
    tmdb_id = tmdb.get("id")
    source_refs = [SourceReference(source=FieldSource.IMDB, identifier=candidate.imdb_id)]
    if tmdb_id is not None:
        source_refs.append(SourceReference(source=FieldSource.TMDB, identifier=str(tmdb_id)))

    raw_genres = tmdb.get("genres", [])
    tmdb_genres = (
        [
            str(item.get("name"))
            for item in raw_genres
            if isinstance(item, Mapping) and item.get("name")
        ]
        if isinstance(raw_genres, list)
        else []
    )
    raw_countries = tmdb.get("production_countries", [])
    countries = (
        [
            str(item.get("iso_3166_1"))
            for item in raw_countries
            if isinstance(item, Mapping) and item.get("iso_3166_1")
        ]
        if isinstance(raw_countries, list)
        else []
    )
    enriched_title = _optional_string(tmdb.get("title"))
    enriched_original_title = _optional_string(tmdb.get("original_title"))
    directors = _string_list(tmdb.get("directors"))
    principal_cast = _string_list(tmdb.get("principal_cast"))
    keywords = _string_list(tmdb.get("keywords"))

    record = MovieRecord(
        id=candidate.imdb_id,
        title=enriched_title or candidate.title,
        original_title=enriched_original_title or candidate.original_title,
        year=candidate.year,
        genres=tmdb_genres or candidate.genres,
        runtime_minutes=_optional_int(tmdb.get("runtime")) or candidate.runtime_minutes,
        imdb_rating=candidate.average_rating,
        imdb_vote_count=candidate.vote_count,
        overview=_optional_string(tmdb.get("overview")) or "",
        original_language=_optional_string(tmdb.get("original_language")),
        production_countries=countries,
        directors=directors,
        principal_cast=principal_cast,
        keywords=keywords,
        source_refs=source_refs,
        field_provenance={
            "title": FieldProvenance(
                source=FieldSource.TMDB if enriched_title else FieldSource.IMDB,
                source_identifier=str(tmdb_id) if enriched_title else candidate.imdb_id,
            ),
            "imdb_rating": FieldProvenance(
                source=FieldSource.IMDB, source_identifier=candidate.imdb_id
            ),
            "overview": FieldProvenance(
                source=FieldSource.TMDB,
                source_identifier=str(tmdb_id) if tmdb_id is not None else None,
            ),
        },
    )
    return with_representation_hash(record)


def _optional_string(value: object) -> str | None:
    return str(value) if value not in {None, ""} else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
