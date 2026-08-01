"""Deterministic movie eligibility filters."""

from __future__ import annotations

from dataclasses import dataclass, field

from local_semantic_engine.domains.movies.models import (
    EvidenceState,
    MissingDataPolicy,
    MovieHardConstraints,
    MovieRecord,
)


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    eligible: list[MovieRecord]
    excluded_reasons: dict[str, list[str]]
    missing_evidence: list[str] = field(default_factory=list)


def apply_hard_constraints(
    movies: list[MovieRecord],
    constraints: MovieHardConstraints,
    *,
    missing_data_policy: MissingDataPolicy,
) -> EligibilityResult:
    """Filter records without relying on the LLM to enforce factual requirements."""

    eligible: list[MovieRecord] = []
    excluded_reasons: dict[str, list[str]] = {}
    missing_evidence: set[str] = set()
    for movie in movies:
        reasons, missing = _violations(movie, constraints, missing_data_policy)
        missing_evidence.update(missing)
        if reasons:
            excluded_reasons[movie.id] = reasons
        else:
            eligible.append(movie)
    return EligibilityResult(
        eligible=eligible,
        excluded_reasons=excluded_reasons,
        missing_evidence=sorted(missing_evidence),
    )


def _violations(
    movie: MovieRecord,
    constraints: MovieHardConstraints,
    policy: MissingDataPolicy,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    missing: list[str] = []

    def require_value(value: object | None, label: str) -> bool:
        if value is not None:
            return True
        missing.append(label)
        if policy == MissingDataPolicy.STRICT:
            reasons.append(f"missing {label}")
        return False

    if movie.id in constraints.excluded_catalogue_ids:
        reasons.append("explicitly excluded catalogue ID")
    if constraints.maximum_runtime_minutes is not None and require_value(
        movie.runtime_minutes, "runtime"
    ):
        if (
            movie.runtime_minutes is not None
            and movie.runtime_minutes > constraints.maximum_runtime_minutes
        ):
            reasons.append("runtime exceeds maximum")
    if constraints.minimum_runtime_minutes is not None and require_value(
        movie.runtime_minutes, "runtime"
    ):
        if (
            movie.runtime_minutes is not None
            and movie.runtime_minutes < constraints.minimum_runtime_minutes
        ):
            reasons.append("runtime is below minimum")
    if constraints.minimum_year is not None and require_value(movie.year, "year"):
        if movie.year is not None and movie.year < constraints.minimum_year:
            reasons.append("year is below minimum")
    if constraints.maximum_year is not None and require_value(movie.year, "year"):
        if movie.year is not None and movie.year > constraints.maximum_year:
            reasons.append("year exceeds maximum")
    if constraints.minimum_imdb_rating is not None and require_value(
        movie.imdb_rating, "IMDb rating"
    ):
        if movie.imdb_rating is not None and movie.imdb_rating < constraints.minimum_imdb_rating:
            reasons.append("IMDb rating is below minimum")

    genres = {genre.casefold() for genre in movie.genres}
    included_genres = {genre.casefold() for genre in constraints.included_genres}
    excluded_genres = {genre.casefold() for genre in constraints.excluded_genres}
    if included_genres and not genres.intersection(included_genres):
        reasons.append("does not include a required genre")
    if genres.intersection(excluded_genres):
        reasons.append("includes an excluded genre")

    language = movie.original_language.casefold() if movie.original_language else None
    allowed_languages = {language.casefold() for language in constraints.allowed_languages}
    excluded_languages = {language.casefold() for language in constraints.excluded_languages}
    if allowed_languages:
        if language is None:
            require_value(language, "original language")
        elif language not in allowed_languages:
            reasons.append("language is not allowed")
    if language is not None and language in excluded_languages:
        reasons.append("language is excluded")

    for attribute in constraints.excluded_content_attributes:
        state = movie.semantic_attributes.content_warnings.get(attribute, EvidenceState.UNKNOWN)
        if state == EvidenceState.PRESENT:
            reasons.append(f"contains excluded content attribute: {attribute}")
        elif state == EvidenceState.UNKNOWN:
            missing.append(f"content attribute: {attribute}")
            if policy == MissingDataPolicy.STRICT:
                reasons.append(f"unknown excluded content attribute: {attribute}")
    return reasons, missing
