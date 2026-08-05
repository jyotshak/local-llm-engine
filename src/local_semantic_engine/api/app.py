"""Loopback API for the local movie recommendation pipeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.core.errors import LocalSemanticEngineError
from local_semantic_engine.domains.movies.models import (
    MovieRecommendationRequest,
    MovieRecommendationResponse,
)
from local_semantic_engine.domains.movies.recommender import MovieRecommender
from local_semantic_engine.domains.movies.representation import (
    REPRESENTATION_VERSION,
    with_representation_hash,
)
from local_semantic_engine.embeddings.ollama import OllamaEmbeddingProvider
from local_semantic_engine.ingestion.movies.indexer import load_movie_corpus
from local_semantic_engine.llm.ollama import OllamaClient
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex

RuntimeFactory = Callable[[AppSettings], Awaitable["MovieRuntime"]]


@dataclass(slots=True)
class MovieRuntime:
    recommender: MovieRecommender
    generator: OllamaClient
    embedder: OllamaEmbeddingProvider

    async def close(self) -> None:
        await self.generator.aclose()
        await self.embedder.aclose()

    async def health(self) -> dict[str, Any]:
        provider = await self.generator.health()
        return {
            "status": "ok" if provider.available else "degraded",
            "provider": provider.model_dump(),
            "corpus_ready": True,
            "index_ready": True,
        }


async def build_movie_runtime(settings: AppSettings) -> MovieRuntime:
    movies = load_movie_corpus(settings.storage.processed_data_dir / "movies.jsonl")
    hashed_movies = [with_representation_hash(movie) for movie in movies]
    index = NumpyVectorIndex.load(
        settings.storage.index_data_dir / "movies",
        embedding_model=settings.ollama.embedding_model,
        representation_version=REPRESENTATION_VERSION,
        record_hashes={movie.id: movie.content_hash for movie in hashed_movies},
    )
    generator = OllamaClient(settings.ollama)
    embedder = OllamaEmbeddingProvider(settings.ollama)
    return MovieRuntime(
        recommender=MovieRecommender(
            settings=settings,
            generator=generator,
            embedder=embedder,
            index=index,
            movies_by_id={movie.id: movie for movie in movies},
        ),
        generator=generator,
        embedder=embedder,
    )


def create_app(
    settings: AppSettings,
    *,
    runtime_factory: RuntimeFactory = build_movie_runtime,
) -> FastAPI:
    """Create an API intentionally restricted to local configuration."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = await runtime_factory(settings)
        app.state.runtime = runtime
        app.state.inference_semaphore = asyncio.Semaphore(settings.api.inference_concurrency)
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Local LLM Engine", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(LocalSemanticEngineError)
    async def local_error_handler(
        request: Request, exc: LocalSemanticEngineError
    ) -> JSONResponse:
        status_code = 503 if exc.retryable else 422
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return await request.app.state.runtime.health()

    @app.get("/v1/models/status")
    async def model_status(request: Request) -> dict[str, Any]:
        return await request.app.state.runtime.health()

    @app.post("/v1/movies/recommend", response_model=MovieRecommendationResponse)
    async def recommend(
        payload: MovieRecommendationRequest, request: Request
    ) -> MovieRecommendationResponse:
        started = perf_counter()
        async with request.app.state.inference_semaphore:
            response = await request.app.state.runtime.recommender.recommend(payload)
        response.timings_ms["total"] = round((perf_counter() - started) * 1000, 2)
        return response

    @app.post("/v1/movies/recommend/stream")
    async def stream_recommend(
        payload: MovieRecommendationRequest, request: Request
    ) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

            async def emit(stage: str, message: str) -> None:
                await queue.put((stage, message))

            async with request.app.state.inference_semaphore:
                task = asyncio.create_task(
                    request.app.state.runtime.recommender.recommend(payload, on_progress=emit)
                )
                while not task.done() or not queue.empty():
                    try:
                        stage, message = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except TimeoutError:
                        continue
                    yield _sse("progress", {"stage": stage, "message": message})
                response = await task
            yield _sse("result", response.model_dump(mode="json"))

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
