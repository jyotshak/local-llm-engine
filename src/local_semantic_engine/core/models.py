"""Provider-neutral models shared by pipelines and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class GenerationSettings(BaseModel):
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1, le=8192)
    stop_sequences: list[str] = Field(default_factory=list)
    seed: int | None = None
    context_tokens: int = Field(default=8192, ge=512)
    thinking: bool = False
    keep_alive: str = "10m"


class ProviderHealth(BaseModel):
    available: bool
    provider: str
    message: str | None = None
    version: str | None = None
    installed_models: list[str] = Field(default_factory=list)


class GenerationUsage(BaseModel):
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None


class GenerationResult(BaseModel):
    text: str
    model: str
    finish_reason: str | None = None
    usage: GenerationUsage = Field(default_factory=GenerationUsage)
    raw_response: Mapping[str, Any] | None = None


class GenerationChunk(BaseModel):
    content: str = ""
    done: bool = False
    finish_reason: str | None = None


class EmbeddingBatch(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    normalized: bool = True


class ScoredId(BaseModel):
    item_id: str
    score: float
