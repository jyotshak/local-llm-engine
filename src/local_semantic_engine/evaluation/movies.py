"""Small, transparent evaluation harness for movie recommendations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from local_semantic_engine.domains.movies.models import (
    MovieRecommendationRequest,
    RecommendationProfile,
)
from local_semantic_engine.domains.movies.recommender import MovieRecommender


class MovieEvaluationCase(BaseModel):
    id: str
    query: str
    count: int = Field(default=3, ge=1, le=10)
    profile: RecommendationProfile = RecommendationProfile.BALANCED
    maximum_runtime_minutes: int | None = Field(default=None, ge=1)
    minimum_year: int | None = Field(default=None, ge=1888)
    minimum_results: int = Field(default=1, ge=0, le=10)


class MovieEvaluationResult(BaseModel):
    id: str
    passed: bool
    duration_ms: float
    recommendation_count: int
    violations: list[str] = Field(default_factory=list)


class MovieEvaluationReport(BaseModel):
    case_count: int
    passed_count: int
    pass_rate: float
    median_duration_ms: float
    results: list[MovieEvaluationResult]


def load_movie_evaluation_cases(path: Path) -> list[MovieEvaluationCase]:
    return [MovieEvaluationCase.model_validate_json(line) for line in path.read_text().splitlines()]


async def evaluate_movies(
    recommender: MovieRecommender, cases: list[MovieEvaluationCase]
) -> MovieEvaluationReport:
    results: list[MovieEvaluationResult] = []
    for case in cases:
        started = perf_counter()
        response = await recommender.recommend(
            MovieRecommendationRequest(query=case.query, count=case.count, profile=case.profile)
        )
        duration_ms = round((perf_counter() - started) * 1000, 2)
        violations: list[str] = []
        if len(response.recommendations) < case.minimum_results:
            violations.append(
                f"Expected at least {case.minimum_results} recommendations, got "
                f"{len(response.recommendations)}."
            )
        for recommendation in response.recommendations:
            movie = recommender.movies_by_id[recommendation.item_id]
            if (
                case.maximum_runtime_minutes is not None
                and (
                    movie.runtime_minutes is None
                    or movie.runtime_minutes > case.maximum_runtime_minutes
                )
            ):
                violations.append(f"{movie.title} violates the maximum runtime.")
            if case.minimum_year is not None and (
                movie.year is None or movie.year < case.minimum_year
            ):
                violations.append(f"{movie.title} violates the minimum year.")
        results.append(
            MovieEvaluationResult(
                id=case.id,
                passed=not violations,
                duration_ms=duration_ms,
                recommendation_count=len(response.recommendations),
                violations=violations,
            )
        )
    durations = sorted(result.duration_ms for result in results)
    middle = len(durations) // 2
    median = (
        durations[middle]
        if len(durations) % 2
        else (durations[middle - 1] + durations[middle]) / 2
    )
    pass_rate = sum(result.passed for result in results) / len(results) if results else 0.0
    return MovieEvaluationReport(
        case_count=len(results),
        passed_count=sum(result.passed for result in results),
        pass_rate=round(pass_rate, 3),
        median_duration_ms=round(median, 2) if durations else 0.0,
        results=results,
    )
