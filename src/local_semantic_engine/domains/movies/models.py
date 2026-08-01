"""Schemas for movie ingestion and recommendation requests."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class FieldSource(StrEnum):
    IMDB = "imdb"
    TMDB = "tmdb"
    INFERRED = "inferred"
    USER = "user"


class EvidenceState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class FieldProvenance(BaseModel):
    source: FieldSource
    source_identifier: str | None = None
    inferred_by: str | None = None


class SourceReference(BaseModel):
    source: FieldSource
    identifier: str
    retrieved_at: str | None = None


class SemanticAttributes(BaseModel):
    themes: list[str] = Field(default_factory=list)
    tone: list[str] = Field(default_factory=list)
    pacing: list[str] = Field(default_factory=list)
    content_warnings: dict[str, EvidenceState] = Field(default_factory=dict)


class MovieRecord(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    original_title: str | None = None
    year: int | None = Field(default=None, ge=1888, le=2200)
    genres: list[str] = Field(default_factory=list)
    runtime_minutes: int | None = Field(default=None, ge=1, le=1000)
    imdb_rating: float | None = Field(default=None, ge=0.0, le=10.0)
    imdb_vote_count: int | None = Field(default=None, ge=0)
    overview: str = ""
    original_language: str | None = None
    production_countries: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    principal_cast: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    collection_id: str | None = None
    semantic_attributes: SemanticAttributes = Field(default_factory=SemanticAttributes)
    source_refs: list[SourceReference] = Field(default_factory=list)
    field_provenance: dict[str, FieldProvenance] = Field(default_factory=dict)
    content_hash: str = ""
    schema_version: str = "1"

    @field_validator("genres", "production_countries", "directors", "principal_cast", "keywords")
    @classmethod
    def remove_blank_items(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class MovieHardConstraints(BaseModel):
    maximum_runtime_minutes: int | None = Field(default=None, ge=1, le=1000)
    minimum_runtime_minutes: int | None = Field(default=None, ge=1, le=1000)
    minimum_year: int | None = Field(default=None, ge=1888, le=2200)
    maximum_year: int | None = Field(default=None, ge=1888, le=2200)
    included_genres: list[str] = Field(default_factory=list)
    excluded_genres: list[str] = Field(default_factory=list)
    allowed_languages: list[str] = Field(default_factory=list)
    excluded_languages: list[str] = Field(default_factory=list)
    minimum_imdb_rating: float | None = Field(default=None, ge=0.0, le=10.0)
    excluded_content_attributes: list[str] = Field(default_factory=list)
    excluded_catalogue_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self) -> MovieHardConstraints:
        if (
            self.minimum_runtime_minutes is not None
            and self.maximum_runtime_minutes is not None
            and self.minimum_runtime_minutes > self.maximum_runtime_minutes
        ):
            raise ValueError("Minimum runtime cannot exceed maximum runtime.")
        if (
            self.minimum_year is not None
            and self.maximum_year is not None
            and self.minimum_year > self.maximum_year
        ):
            raise ValueError("Minimum year cannot exceed maximum year.")
        return self


class MovieSoftConstraints(BaseModel):
    preferred_genres: list[str] = Field(default_factory=list)
    preferred_year_range: tuple[int, int] | None = None


class MoviePreferences(BaseModel):
    positive_preferences: list[str] = Field(default_factory=list)
    negative_preferences: list[str] = Field(default_factory=list)
    hard_constraints: MovieHardConstraints = Field(default_factory=MovieHardConstraints)
    soft_constraints: MovieSoftConstraints = Field(default_factory=MovieSoftConstraints)
    liked_title_mentions: list[str] = Field(default_factory=list)
    disliked_title_mentions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class RecommendationProfile(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class MissingDataPolicy(StrEnum):
    STRICT = "strict"
    ALLOW_WITH_WARNING = "allow_with_warning"


class MovieRecommendationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    count: int = Field(default=5, ge=1, le=10)
    profile: RecommendationProfile = RecommendationProfile.BALANCED
    debug: bool = False
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.STRICT


class RecommendationItem(BaseModel):
    item_id: str
    title: str
    year: int | None = None
    score: int = Field(ge=0, le=100)
    reason: str
    matching_attributes: list[str] = Field(default_factory=list)
    possible_mismatches: list[str] = Field(default_factory=list)


class RerankerRecommendationItem(BaseModel):
    item_id: str
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=1000)
    matching_attributes: list[str] = Field(default_factory=list)
    possible_mismatches: list[str] = Field(default_factory=list)


class RerankerResponse(BaseModel):
    recommendations: list[RerankerRecommendationItem] = Field(default_factory=list)


class UncertaintyReport(BaseModel):
    uncertain: bool
    reasons: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class MovieRecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]
    uncertainty: UncertaintyReport
    warnings: list[str] = Field(default_factory=list)
    trace_id: str
    timings_ms: dict[str, float] = Field(default_factory=dict)
    profile: RecommendationProfile
    debug: dict[str, object] | None = None
