"""Schemas for grounded, cited document question answering."""

from __future__ import annotations

from pydantic import BaseModel, Field

from local_semantic_engine.domains.movies.models import RecommendationProfile


class DocumentChunk(BaseModel):
    id: str
    document_name: str
    page_number: int = Field(ge=1)
    chunk_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    content_hash: str = ""


class DocumentQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    profile: RecommendationProfile = RecommendationProfile.BALANCED
    max_chunks: int = Field(default=6, ge=1, le=20)


class DocumentCitation(BaseModel):
    document_name: str
    page_number: int = Field(ge=1)


class GroundedAnswer(BaseModel):
    """Minimal schema kept compatible with Ollama's constrained JSON grammar."""

    answer: str
    cited_chunk_ids: list[str]
    insufficient_evidence: bool


class DocumentAnswerResponse(BaseModel):
    answer: str
    citations: list[DocumentCitation] = Field(default_factory=list)
    uncertain: bool
    warnings: list[str] = Field(default_factory=list)
    trace_id: str
    timings_ms: dict[str, float] = Field(default_factory=dict)
