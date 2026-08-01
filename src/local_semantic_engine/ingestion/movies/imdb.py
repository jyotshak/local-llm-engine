"""Deterministic selection from IMDb's official non-commercial TSV files."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImdbMovieCandidate:
    imdb_id: str
    title: str
    original_title: str | None
    year: int
    runtime_minutes: int | None
    genres: list[str]
    average_rating: float
    vote_count: int


def _nullable(value: str | None) -> str | None:
    if value is None or value == "\\N" or not value.strip():
        return None
    return value.strip()


def _integer(value: str | None) -> int | None:
    normalized = _nullable(value)
    if normalized is None:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    normalized = _nullable(value)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _read_tsv(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def select_most_voted_movies(
    basics_path: Path,
    ratings_path: Path,
    *,
    limit: int = 1000,
) -> list[ImdbMovieCandidate]:
    """Return eligible top ``limit`` movies sorted by IMDb vote count."""

    if limit < 1:
        raise ValueError("limit must be positive.")

    eligible_basics: dict[str, tuple[str, str | None, int, int | None, list[str]]] = {}
    for row in _read_tsv(basics_path):
        if row.get("titleType") != "movie" or row.get("isAdult") != "0":
            continue
        title = _nullable(row.get("primaryTitle"))
        year = _integer(row.get("startYear"))
        if title is None or year is None:
            continue
        genres = [
            genre.strip()
            for genre in (_nullable(row.get("genres")) or "").split(",")
            if genre.strip()
        ]
        imdb_id = row.get("tconst")
        if not imdb_id:
            continue
        eligible_basics[imdb_id] = (
            title,
            _nullable(row.get("originalTitle")),
            year,
            _integer(row.get("runtimeMinutes")),
            genres,
        )

    ranked: list[ImdbMovieCandidate] = []
    for row in _read_tsv(ratings_path):
        imdb_id = row.get("tconst")
        basic = eligible_basics.get(imdb_id or "")
        if basic is None:
            continue
        rating = _float(row.get("averageRating"))
        votes = _integer(row.get("numVotes"))
        if rating is None or votes is None or votes <= 0:
            continue
        title, original_title, year, runtime_minutes, genres = basic
        ranked.append(
            ImdbMovieCandidate(
                imdb_id=imdb_id or "",
                title=title,
                original_title=original_title,
                year=year,
                runtime_minutes=runtime_minutes,
                genres=genres,
                average_rating=rating,
                vote_count=votes,
            )
        )

    ranked.sort(key=lambda item: (-item.vote_count, -item.average_rating, item.imdb_id))
    return ranked[:limit]
