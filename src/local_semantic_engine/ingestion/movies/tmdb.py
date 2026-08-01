"""Explicit setup-time TMDB enrichment client."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx


class TmdbClient:
    """Fetch TMDB metadata for an IMDb-selected movie using a user-supplied token."""

    def __init__(
        self,
        read_access_token: str | None = None,
        *,
        api_key: str | None = None,
        language: str = "en-US",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not (read_access_token or api_key):
            raise ValueError("A TMDB read access token or API key is required for enrichment.")
        self._language = language
        self._api_key = api_key
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {read_access_token}"} if read_access_token else {}
        self._client = client or httpx.AsyncClient(
            base_url="https://api.themoviedb.org/3",
            headers=headers,
            timeout=httpx.Timeout(30.0),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def enrich_movie(
        self, imdb_id: str, *, include_reviews: bool = False
    ) -> dict[str, object] | None:
        """Resolve an IMDb ID and collect compact metadata; reviews are opt-in."""

        resolved = await self._get(
            f"/find/{imdb_id}", params={"external_source": "imdb_id", "language": self._language}
        )
        movie_results = resolved.get("movie_results")
        if not isinstance(movie_results, list) or not movie_results:
            return None
        first_result = movie_results[0]
        if not isinstance(first_result, Mapping) or not isinstance(first_result.get("id"), int):
            return None
        movie_id = first_result["id"]
        responses = await asyncio.gather(
            self._get(f"/movie/{movie_id}", params={"language": self._language}),
            self._get(f"/movie/{movie_id}/credits", params={"language": self._language}),
            self._get(f"/movie/{movie_id}/keywords"),
            *(
                [self._get(f"/movie/{movie_id}/reviews", params={"language": self._language})]
                if include_reviews
                else []
            ),
        )
        details, credits, keyword_data, *review_data = responses
        directors = [
            str(person["name"])
            for person in credits.get("crew", [])
            if isinstance(person, Mapping)
            and person.get("job") == "Director"
            and person.get("name")
        ]
        cast = [
            str(person["name"])
            for person in credits.get("cast", [])[:8]
            if isinstance(person, Mapping) and person.get("name")
        ]
        keywords = [
            str(keyword["name"])
            for keyword in keyword_data.get("keywords", [])
            if isinstance(keyword, Mapping) and keyword.get("name")
        ]
        enriched = dict(details)
        enriched.update(
            {
                "directors": directors,
                "principal_cast": cast,
                "keywords": keywords,
            }
        )
        if review_data:
            enriched["reviews"] = review_data[0].get("results", [])
        return enriched

    async def _get(self, path: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
        request_params = dict(params or {})
        if self._api_key:
            request_params["api_key"] = self._api_key
        response = await self._client.get(path, params=request_params)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"TMDB returned an unexpected response for {path}.")
        return body
