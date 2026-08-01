"""Ollama HTTP client with explicit retries, timing, and schema validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from local_semantic_engine.config.models import OllamaSettings
from local_semantic_engine.core.errors import (
    ModelNotInstalledError,
    ModelOutputInvalidError,
    ModelTimeoutError,
    OllamaUnavailableError,
)
from local_semantic_engine.core.models import (
    ChatMessage,
    GenerationChunk,
    GenerationResult,
    GenerationSettings,
    GenerationUsage,
    ProviderHealth,
)

StructuredT = TypeVar("StructuredT", bound=BaseModel)


class OllamaClient:
    """A small async wrapper around the documented local Ollama API."""

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
        """Close the internally created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> ProviderHealth:
        """Report local server availability and installed model names."""

        try:
            version_response = await self._client.get("/api/version")
            version_response.raise_for_status()
            models_response = await self._client.get("/api/tags")
            models_response.raise_for_status()
        except (httpx.ConnectError, httpx.NetworkError):
            return ProviderHealth(
                available=False,
                provider=self.provider_name,
                message="Ollama is not reachable on the configured local URL.",
            )
        except httpx.TimeoutException:
            return ProviderHealth(
                available=False,
                provider=self.provider_name,
                message="Ollama did not respond before the configured timeout.",
            )
        except httpx.HTTPError as exc:
            return ProviderHealth(available=False, provider=self.provider_name, message=str(exc))

        version_data = version_response.json()
        models_data = models_response.json()
        models = [
            str(item["name"])
            for item in models_data.get("models", [])
            if isinstance(item, Mapping) and item.get("name")
        ]
        return ProviderHealth(
            available=True,
            provider=self.provider_name,
            version=version_data.get("version"),
            installed_models=models,
        )

    async def generate_text(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> GenerationResult:
        """Generate a completed, non-streaming response."""

        response = await self._chat(messages, settings, response_format=None)
        return self._generation_result(response, settings.model)

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[StructuredT],
        settings: GenerationSettings,
    ) -> StructuredT:
        """Request JSON-schema output and independently validate it with Pydantic."""

        response = await self._chat(messages, settings, response_format=schema.model_json_schema())
        result = self._generation_result(response, settings.model)
        try:
            return schema.model_validate_json(result.text)
        except ValidationError as exc:
            raise ModelOutputInvalidError([str(exc)]) from exc

    async def _chat(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = self._chat_payload(
            messages, settings, stream=False, response_format=response_format
        )
        response = await self._request_with_retry("POST", "/api/chat", json=payload)
        return self._decode_response(response)

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> AsyncIterator[GenerationChunk]:
        """Yield text chunks from Ollama's newline-delimited JSON stream."""

        payload = self._chat_payload(messages, settings, stream=True, response_format=None)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                self._raise_for_provider_error(response, settings.model)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ModelOutputInvalidError(
                            ["Ollama returned malformed stream JSON."]
                        ) from exc
                    message = chunk.get("message") or {}
                    yield GenerationChunk(
                        content=str(message.get("content") or ""),
                        done=bool(chunk.get("done", False)),
                        finish_reason=chunk.get("done_reason"),
                    )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError() from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise OllamaUnavailableError() from exc

    def _chat_payload(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        stream: bool,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": settings.temperature,
            "num_predict": settings.max_output_tokens,
            "num_ctx": settings.context_tokens,
        }
        if settings.stop_sequences:
            options["stop"] = settings.stop_sequences
        if settings.seed is not None:
            options["seed"] = settings.seed

        payload: dict[str, Any] = {
            "model": settings.model,
            "messages": [message.model_dump() for message in messages],
            "stream": stream,
            "options": options,
            "keep_alive": settings.keep_alive,
            "think": settings.thinking,
        }
        if response_format is not None:
            payload["format"] = response_format
        return payload

    async def _request_with_retry(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        attempts = self._settings.max_retries + 1
        for attempt in range(attempts):
            try:
                response = await self._client.request(method, path, **kwargs)
                self._raise_for_provider_error(
                    response, self._model_from_payload(kwargs.get("json"))
                )
                return response
            except httpx.TimeoutException as exc:
                if attempt + 1 == attempts:
                    raise ModelTimeoutError() from exc
            except (httpx.ConnectError, httpx.NetworkError) as exc:
                if attempt + 1 == attempts:
                    raise OllamaUnavailableError() from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt + 1 == attempts:
                    raise
            await asyncio.sleep(0.15 * (2**attempt))
        raise AssertionError("Retry loop ended unexpectedly.")

    @staticmethod
    def _model_from_payload(payload: object) -> str:
        return str(payload.get("model", "")) if isinstance(payload, Mapping) else ""

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ModelOutputInvalidError(["Ollama returned non-JSON output."]) from exc
        if not isinstance(body, dict):
            raise ModelOutputInvalidError(["Ollama returned an unexpected response shape."])
        return body

    @staticmethod
    def _generation_result(response: Mapping[str, Any], requested_model: str) -> GenerationResult:
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise ModelOutputInvalidError(["Ollama response did not contain assistant text."])
        usage = GenerationUsage(
            prompt_tokens=response.get("prompt_eval_count"),
            output_tokens=response.get("eval_count"),
            total_duration_ns=response.get("total_duration"),
            load_duration_ns=response.get("load_duration"),
            prompt_eval_duration_ns=response.get("prompt_eval_duration"),
            eval_duration_ns=response.get("eval_duration"),
        )
        return GenerationResult(
            text=content,
            model=str(response.get("model") or requested_model),
            finish_reason=response.get("done_reason"),
            usage=usage,
            raw_response=dict(response),
        )

    @staticmethod
    def _raise_for_provider_error(response: httpx.Response, model: str) -> None:
        if response.status_code == 404:
            raise ModelNotInstalledError(model)
        response.raise_for_status()
