from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from local_semantic_engine.api import create_app
from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.domains.movies.models import (
    MovieRecommendationResponse,
    RecommendationProfile,
    UncertaintyReport,
)


class FakeRecommender:
    async def recommend(self, request, *, on_progress=None) -> MovieRecommendationResponse:
        if on_progress is not None:
            await on_progress("filtering", "Applying deterministic catalogue constraints.")
        return MovieRecommendationResponse(
            recommendations=[],
            uncertainty=UncertaintyReport(uncertain=False),
            trace_id="test-trace",
            profile=RecommendationProfile.BALANCED,
        )


@dataclass
class FakeRuntime:
    recommender: FakeRecommender
    closed: bool = False

    async def close(self) -> None:
        self.closed = True

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "corpus_ready": True, "index_ready": True}


def test_api_serves_health_recommendation_and_stream() -> None:
    runtime = FakeRuntime(recommender=FakeRecommender())

    async def factory(settings: AppSettings) -> FakeRuntime:
        return runtime

    with TestClient(create_app(AppSettings(), runtime_factory=factory)) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/v1/movies/recommend", json={"query": "anything"})
        assert response.status_code == 200
        assert response.json()["trace_id"] == "test-trace"
        stream = client.post("/v1/movies/recommend/stream", json={"query": "anything"})
        assert "event: progress" in stream.text
        assert "event: result" in stream.text

    assert runtime.closed
