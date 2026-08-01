from __future__ import annotations

import httpx
import pytest

from local_semantic_engine.ingestion.movies.tmdb import TmdbClient


@pytest.mark.asyncio
async def test_enrichment_combines_details_credits_and_keywords() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/3/find/tt001":
            return httpx.Response(200, json={"movie_results": [{"id": 7}]})
        if request.url.path == "/3/movie/7":
            return httpx.Response(200, json={"id": 7, "title": "Example", "overview": "Plot."})
        if request.url.path == "/3/movie/7/credits":
            return httpx.Response(
                200,
                json={
                    "crew": [{"job": "Director", "name": "Director Name"}],
                    "cast": [{"name": "Actor Name"}],
                },
            )
        if request.url.path == "/3/movie/7/keywords":
            return httpx.Response(200, json={"keywords": [{"name": "time travel"}]})
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.themoviedb.org/3"
    ) as http_client:
        client = TmdbClient("token", client=http_client)
        result = await client.enrich_movie("tt001")

    assert result is not None
    assert result["directors"] == ["Director Name"]
    assert result["principal_cast"] == ["Actor Name"]
    assert result["keywords"] == ["time travel"]


@pytest.mark.asyncio
async def test_api_key_is_sent_as_a_query_parameter() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "legacy-key"
        return httpx.Response(200, json={"movie_results": []})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.themoviedb.org/3"
    ) as http_client:
        client = TmdbClient(api_key="legacy-key", client=http_client)
        result = await client.enrich_movie("tt001")

    assert result is None
