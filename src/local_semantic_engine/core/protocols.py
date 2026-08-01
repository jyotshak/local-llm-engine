"""Dependency-injection contracts used by the application layer."""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

from local_semantic_engine.core.models import (
    ChatMessage,
    EmbeddingBatch,
    GenerationChunk,
    GenerationResult,
    GenerationSettings,
    ProviderHealth,
    ScoredId,
)

StructuredT = TypeVar("StructuredT", bound=BaseModel)


class LocalLLM(Protocol):
    """A replaceable local text-generation provider."""

    async def health(self) -> ProviderHealth: ...

    async def generate_text(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> GenerationResult: ...

    def stream_text(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> AsyncIterator[GenerationChunk]: ...

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[StructuredT],
        settings: GenerationSettings,
    ) -> StructuredT: ...


class EmbeddingProvider(Protocol):
    """A replaceable local embedding provider."""

    async def health(self) -> ProviderHealth: ...

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch: ...

    async def embed_query(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    """A similarity index over stable local item IDs."""

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int,
        eligible_ids: Collection[str] | None = None,
    ) -> list[ScoredId]: ...
