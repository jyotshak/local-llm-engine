"""Validate model-selected movie IDs against local candidate facts."""

from __future__ import annotations

from local_semantic_engine.core.errors import ModelOutputInvalidError
from local_semantic_engine.domains.movies.filters import apply_hard_constraints
from local_semantic_engine.domains.movies.models import (
    MissingDataPolicy,
    MovieHardConstraints,
    MovieRecord,
    RerankerResponse,
)


def validate_reranker_response(
    response: RerankerResponse,
    *,
    candidate_movies: list[MovieRecord],
    requested_count: int,
    constraints: MovieHardConstraints,
    missing_data_policy: MissingDataPolicy,
) -> RerankerResponse:
    """Reject invented, duplicate, or ineligible recommendations."""

    errors: list[str] = []
    candidate_by_id = {movie.id: movie for movie in candidate_movies}
    item_ids = [item.item_id for item in response.recommendations]
    if len(item_ids) != requested_count:
        errors.append(f"Expected {requested_count} recommendations, received {len(item_ids)}.")
    if len(set(item_ids)) != len(item_ids):
        errors.append("Recommendations contain duplicate IDs.")
    unknown_ids = sorted(set(item_ids).difference(candidate_by_id))
    if unknown_ids:
        errors.append(
            f"Recommendations contain IDs outside the supplied candidates: {unknown_ids}."
        )
    known_items = [candidate_by_id[item_id] for item_id in item_ids if item_id in candidate_by_id]
    violations = apply_hard_constraints(
        known_items,
        constraints,
        missing_data_policy=missing_data_policy,
    ).excluded_reasons
    if violations:
        errors.append(f"Recommendations violate hard constraints: {violations}.")
    if errors:
        raise ModelOutputInvalidError(errors)
    return response
