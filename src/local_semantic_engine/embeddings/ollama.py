"""Ollama embedding provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from local_semantic_engine.config.models import OllamaSettings
from local_semantic_engine.core.errors import (
    ModelOutputInvalidError,
    ModelTimeoutError,
    OllamaUnavailableError,
)
from local_semantic_engine.core.models import EmbeddingBatch, ProviderHealth
from local_semantic_engine.llm.ollama import OllamaClient


class OllamaEmbeddingProvider:
    """Generate normalized local embeddings with Ollama's `/api/embed` endpoint."""

    provider_name = "ollama"

    def __init__(
        self, settings: OllamaSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout_seconds,
                read=settings.read_timeout_seconds,
                write=settings.read_timeout_seconds,
                pool=settings.connect_timeout_seconds,
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> ProviderHealth:
        """Reuse the same local health semantics as the generation provider."""

        checker = OllamaClient(self._settings, client=self._client)
        return await checker.health()

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(
                embeddings=[],
                model=self._settings.embedding_model,
                dimensions=0,
            )
        payload = {"model": self._settings.embedding_model, "input": list(texts)}
        try:
            response = await self._client.post("/api/embed", json=payload)
            if response.status_code == 404:
                from local_semantic_engine.core.errors import ModelNotInstalledError

                raise ModelNotInstalledError(self._settings.embedding_model)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError() from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise OllamaUnavailableError() from exc

        data = self._decode(response)
        raw_embeddings = data.get("embeddings")
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(texts):
            raise ModelOutputInvalidError(["Ollama returned an unexpected embedding batch."])
        embeddings: list[list[float]] = []
        dimensions: int | None = None
        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, list) or not raw_embedding:
                raise ModelOutputInvalidError(["Ollama returned an empty embedding vector."])
            vector = [float(value) for value in raw_embedding]
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ModelOutputInvalidError(
                    ["Ollama returned inconsistent embedding dimensions."]
                )
            embeddings.append(vector)
        return EmbeddingBatch(
            embeddings=embeddings,
            model=self._settings.embedding_model,
            dimensions=dimensions or 0,
            normalized=True,
        )

    async def embed_query(self, text: str) -> list[float]:
        batch = await self.embed_texts([text])
        return batch.embeddings[0]

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        data = response.json()
        if not isinstance(data, Mapping):
            raise ModelOutputInvalidError(
                ["Ollama returned an unexpected embedding response shape."]
            )
        return dict(data)
