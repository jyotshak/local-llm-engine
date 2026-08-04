from __future__ import annotations

import numpy as np
import pytest

from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.core.models import ChatMessage
from local_semantic_engine.domains.movies.models import (
    MovieHardConstraints,
    MoviePreferences,
    MovieRecommendationRequest,
    MovieRecord,
    RerankerRecommendationItem,
    RerankerResponse,
)
from local_semantic_engine.domains.movies.recommender import (
    MovieRecommender,
    _apply_deterministic_query_constraints,
)
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex


class FakeGenerator:
    def __init__(self) -> None:
        self.messages: list[list[ChatMessage]] = []

    async def generate_structured(self, messages, schema, settings):
        self.messages.append(list(messages))
        if schema is MoviePreferences:
            return MoviePreferences(
                hard_constraints=MovieHardConstraints(maximum_runtime_minutes=120)
            )
        return RerankerResponse(
            recommendations=[
                RerankerRecommendationItem(item_id="short", score=91, reason="Fits the request.")
            ]
        )


class FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_recommender_filters_hard_constraints_before_reranking() -> None:
    long_movie = MovieRecord(id="long", title="Too Long", runtime_minutes=172)
    short_movie = MovieRecord(id="short", title="Short Enough", runtime_minutes=105)
    generator = FakeGenerator()
    recommender = MovieRecommender(
        settings=AppSettings(),
        generator=generator,
        embedder=FakeEmbedder(),
        index=NumpyVectorIndex(["long", "short"], np.array([[1.0, 0.0], [0.9, 0.1]])),
        movies_by_id={"long": long_movie, "short": short_movie},
    )

    response = await recommender.recommend(
        MovieRecommendationRequest(query="thoughtful sci-fi under two hours", count=1)
    )

    assert [item.item_id for item in response.recommendations] == ["short"]
    rerank_prompt = generator.messages[1][1].content
    assert "Too Long" not in rerank_prompt
    assert "Short Enough" in rerank_prompt


def test_runtime_phrase_guardrail_overrides_missing_model_constraint() -> None:
    preferences = _apply_deterministic_query_constraints(
        "thoughtful science fiction under two hours", MoviePreferences()
    )

    assert preferences.hard_constraints.maximum_runtime_minutes == 120
