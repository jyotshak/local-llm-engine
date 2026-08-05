"""Local, validated movie recommendation pipeline."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.core.models import ChatMessage, GenerationSettings
from local_semantic_engine.domains.movies.filters import apply_hard_constraints
from local_semantic_engine.domains.movies.models import (
    MoviePreferences,
    MovieRecommendationRequest,
    MovieRecommendationResponse,
    MovieRecord,
    RecommendationItem,
    RerankerResponse,
    UncertaintyReport,
)
from local_semantic_engine.domains.movies.representation import render_movie_search_text
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex
from local_semantic_engine.validation.recommendations import validate_reranker_response


class StructuredGenerator(Protocol):
    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type,
        settings: GenerationSettings,
    ): ...


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...


ProgressCallback = Callable[[str, str], Awaitable[None] | None]


@dataclass(slots=True)
class MovieRecommender:
    settings: AppSettings
    generator: StructuredGenerator
    embedder: QueryEmbedder
    index: NumpyVectorIndex
    movies_by_id: dict[str, MovieRecord]

    async def recommend(
        self, request: MovieRecommendationRequest, *, on_progress: ProgressCallback | None = None
    ) -> MovieRecommendationResponse:
        """Produce recommendations that are always checked against local facts."""

        await _notify(on_progress, "parsing", "Interpreting preferences and hard constraints.")
        preferences = await self._parse_preferences(request.query, request.profile.value)
        movies = list(self.movies_by_id.values())
        await _notify(on_progress, "filtering", "Applying deterministic catalogue constraints.")
        eligibility = apply_hard_constraints(
            movies,
            preferences.hard_constraints,
            missing_data_policy=request.missing_data_policy,
        )
        if not eligibility.eligible:
            return MovieRecommendationResponse(
                recommendations=[],
                uncertainty=UncertaintyReport(
                    uncertain=True,
                    reasons=["No movies satisfy the stated hard constraints."],
                    missing_evidence=eligibility.missing_evidence,
                ),
                warnings=[],
                trace_id=str(uuid4()),
                profile=request.profile,
            )

        await _notify(on_progress, "retrieving", "Searching eligible movies semantically.")
        query_vector = await self.embedder.embed_query(request.query)
        candidate_count = min(
            self.settings.profile(request.profile.value).rerank_candidate_count,
            self.settings.retrieval.broad_candidate_count,
            len(eligibility.eligible),
        )
        semantic_matches = self.index.search(
            query_vector,
            top_k=candidate_count,
            eligible_ids={movie.id for movie in eligibility.eligible},
        )
        candidates = [self.movies_by_id[match.item_id] for match in semantic_matches]
        requested_count = min(request.count, len(candidates))
        await _notify(on_progress, "ranking", "Ranking the local candidate set.")
        reranked = await self._rerank(
            query=request.query,
            candidates=candidates,
            requested_count=requested_count,
            profile=request.profile.value,
        )
        await _notify(
            on_progress, "validating", "Verifying returned recommendations against facts."
        )
        validated = validate_reranker_response(
            reranked,
            candidate_movies=candidates,
            requested_count=requested_count,
            constraints=preferences.hard_constraints,
            missing_data_policy=request.missing_data_policy,
        )
        recommendations = [
            RecommendationItem(
                item_id=item.item_id,
                title=self.movies_by_id[item.item_id].title,
                year=self.movies_by_id[item.item_id].year,
                score=item.score,
                reason=item.reason,
                matching_attributes=item.matching_attributes,
                possible_mismatches=item.possible_mismatches,
            )
            for item in validated.recommendations
        ]
        uncertainty_reasons = list(preferences.ambiguities)
        if requested_count < request.count:
            uncertainty_reasons.append("Fewer eligible candidates were available than requested.")
        return MovieRecommendationResponse(
            recommendations=recommendations,
            uncertainty=UncertaintyReport(
                uncertain=bool(uncertainty_reasons or eligibility.missing_evidence),
                reasons=uncertainty_reasons,
                missing_evidence=eligibility.missing_evidence,
            ),
            warnings=[],
            trace_id=str(uuid4()),
            profile=request.profile,
        )
    async def _parse_preferences(self, query: str, profile: str) -> MoviePreferences:
        parsed = await self.generator.generate_structured(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Extract movie preferences into the provided JSON schema. "
                        "Treat explicit limits as hard constraints. Examples: 'under two hours' "
                        "means maximum_runtime_minutes=120; 'after 2010' means minimum_year=2010. "
                        "Do not invent constraints or facts."
                    ),
                ),
                ChatMessage(role="user", content=query),
            ],
            MoviePreferences,
            self._generation_settings(profile, max_output_tokens=400),
        )
        return _apply_deterministic_query_constraints(query, parsed)

    async def _rerank(
        self, *, query: str, candidates: list[MovieRecord], requested_count: int, profile: str
    ) -> RerankerResponse:
        profile_settings = self.settings.profile(profile)
        candidate_text = "\n\n".join(
            f"ID: {movie.id}\n"
            f"{render_movie_search_text(movie)[:profile_settings.candidate_summary_characters]}"
            for movie in candidates
        )
        return await self.generator.generate_structured(
            [
                ChatMessage(
                    role="system",
                    content=(
                        f"Select exactly {requested_count} distinct candidates from the "
                        "supplied catalogue. Return only schema-valid JSON. Use only supplied "
                        "IDs. Score fitness from 0 to 100. Give concise, evidence-grounded "
                        "reasons and list possible mismatches when appropriate."
                    ),
                ),
                ChatMessage(
                    role="user", content=f"Request:\n{query}\n\nCandidates:\n{candidate_text}"
                ),
            ],
            RerankerResponse,
            self._generation_settings(
                profile, max_output_tokens=profile_settings.max_output_tokens
            ),
        )

    def _generation_settings(self, profile: str, *, max_output_tokens: int) -> GenerationSettings:
        selected = self.settings.profile(profile)
        return GenerationSettings(
            model=self.settings.ollama.generation_model,
            temperature=selected.temperature,
            max_output_tokens=max_output_tokens,
            context_tokens=selected.context_tokens,
            thinking=selected.thinking,
            keep_alive=self.settings.ollama.keep_alive,
        )


async def _notify(callback: ProgressCallback | None, stage: str, message: str) -> None:
    if callback is None:
        return
    result = callback(stage, message)
    if inspect.isawaitable(result):
        await result


_RUNTIME_WORD_VALUES = {"one": 1, "two": 2, "three": 3, "four": 4}
_MAX_RUNTIME_PATTERN = re.compile(
    r"\b(?:under|below|within|less\s+than|at\s+most|no\s+more\s+than)\s+"
    r"(?:(?P<hours>\d+(?:\.\d+)?|one|two|three|four)\s*(?:hours?|hrs?|h)|"
    r"(?P<minutes>\d+)\s*(?:minutes?|mins?|m))\b",
    re.IGNORECASE,
)


def _apply_deterministic_query_constraints(
    query: str, preferences: MoviePreferences
) -> MoviePreferences:
    """Guarantee obvious numeric limits even when an LLM misses their schema field."""

    match = _MAX_RUNTIME_PATTERN.search(query)
    if match is None:
        return preferences
    if match.group("minutes"):
        maximum_runtime = int(match.group("minutes"))
    else:
        raw_hours = match.group("hours")
        if raw_hours is None:
            return preferences
        hours = float(_RUNTIME_WORD_VALUES.get(raw_hours.casefold(), raw_hours))
        maximum_runtime = round(hours * 60)
    existing = preferences.hard_constraints.maximum_runtime_minutes
    merged_maximum = min(existing, maximum_runtime) if existing is not None else maximum_runtime
    return preferences.model_copy(
        update={
            "hard_constraints": preferences.hard_constraints.model_copy(
                update={"maximum_runtime_minutes": merged_maximum}
            )
        }
    )
