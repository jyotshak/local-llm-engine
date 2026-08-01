"""Compact, versioned movie text used for semantic retrieval."""

from __future__ import annotations

import hashlib
import re

from local_semantic_engine.domains.movies.models import MovieRecord

REPRESENTATION_VERSION = "1"
_WHITESPACE = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")


def _clean(value: str, *, limit: int) -> str:
    without_html = _HTML.sub(" ", value)
    collapsed = _WHITESPACE.sub(" ", without_html).strip()
    return collapsed[:limit].rstrip()


def render_movie_search_text(movie: MovieRecord) -> str:
    """Render only validated, compact fields for a stable embedding input."""

    lines = [f"Title: {_clean(movie.title, limit=180)}"]
    if movie.year is not None:
        lines[0] += f" ({movie.year})"
    if movie.genres:
        lines.append(f"Genres: {', '.join(movie.genres[:6])}")
    if movie.overview:
        lines.append(f"Overview: {_clean(movie.overview, limit=700)}")
    if movie.directors:
        lines.append(f"Director: {', '.join(movie.directors[:3])}")
    if movie.principal_cast:
        lines.append(f"Cast: {', '.join(movie.principal_cast[:6])}")
    if movie.keywords:
        lines.append(f"Keywords: {', '.join(movie.keywords[:16])}")
    if movie.runtime_minutes is not None:
        lines.append(f"Runtime: {movie.runtime_minutes} minutes")
    if movie.imdb_rating is not None and movie.imdb_vote_count is not None:
        lines.append(f"IMDb rating: {movie.imdb_rating:.1f} from {movie.imdb_vote_count} votes")
    attributes = movie.semantic_attributes
    if attributes.themes:
        lines.append(f"Derived themes: {', '.join(attributes.themes[:8])} [inferred]")
    if attributes.tone:
        lines.append(f"Derived tone: {', '.join(attributes.tone[:8])} [inferred]")
    if attributes.pacing:
        lines.append(f"Derived pacing: {', '.join(attributes.pacing[:4])} [inferred]")
    present_warnings = [
        name for name, state in attributes.content_warnings.items() if state.value == "present"
    ]
    if present_warnings:
        lines.append(f"Content signals: {', '.join(sorted(present_warnings))}")
    return "\n".join(lines)


def with_representation_hash(movie: MovieRecord) -> MovieRecord:
    """Return a copy whose hash reflects the current renderer version and text."""

    content = f"{REPRESENTATION_VERSION}\n{render_movie_search_text(movie)}"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return movie.model_copy(update={"content_hash": content_hash})
