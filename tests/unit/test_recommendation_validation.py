from __future__ import annotations

import pytest

from local_semantic_engine.core.errors import ModelOutputInvalidError
from local_semantic_engine.domains.movies.models import (
    MissingDataPolicy,
    MovieHardConstraints,
    MovieRecord,
    RerankerRecommendationItem,
    RerankerResponse,
)
from local_semantic_engine.validation.recommendations import validate_reranker_response


def test_validator_rejects_invented_and_duplicate_ids() -> None:
    response = RerankerResponse(
        recommendations=[
            RerankerRecommendationItem(item_id="a", score=90, reason="Fits."),
            RerankerRecommendationItem(item_id="a", score=80, reason="Also fits."),
            RerankerRecommendationItem(item_id="invented", score=70, reason="Invented."),
        ]
    )

    with pytest.raises(ModelOutputInvalidError) as exc_info:
        validate_reranker_response(
            response,
            candidate_movies=[MovieRecord(id="a", title="A")],
            requested_count=3,
            constraints=MovieHardConstraints(),
            missing_data_policy=MissingDataPolicy.STRICT,
        )

    assert "duplicate IDs" in exc_info.value.details[0]
    assert "outside the supplied candidates" in exc_info.value.details[1]


def test_validator_accepts_exact_eligible_candidate_ids() -> None:
    response = RerankerResponse(
        recommendations=[
            RerankerRecommendationItem(item_id="a", score=90, reason="Fits."),
            RerankerRecommendationItem(item_id="b", score=80, reason="Fits too."),
        ]
    )

    validated = validate_reranker_response(
        response,
        candidate_movies=[MovieRecord(id="a", title="A"), MovieRecord(id="b", title="B")],
        requested_count=2,
        constraints=MovieHardConstraints(),
        missing_data_policy=MissingDataPolicy.STRICT,
    )

    assert validated == response
