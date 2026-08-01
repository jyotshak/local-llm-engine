from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from local_semantic_engine.config.models import OllamaSettings
from local_semantic_engine.core.errors import ModelNotInstalledError
from local_semantic_engine.core.models import ChatMessage, GenerationSettings, MessageRole
from local_semantic_engine.embeddings.ollama import OllamaEmbeddingProvider
from local_semantic_engine.llm.ollama import OllamaClient


def _settings() -> OllamaSettings:
    return OllamaSettings(generation_model="test-model", embedding_model="test-embedding")


def _http_client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, base_url="http://127.0.0.1:11434")


@pytest.mark.asyncio
async def test_generate_text_maps_settings_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 77
        assert payload["options"]["num_ctx"] == 2048
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )

    async with _http_client(httpx.MockTransport(handler)) as http_client:
        client = OllamaClient(_settings(), client=http_client)
        result = await client.generate_text(
            [ChatMessage(role=MessageRole.USER, content="hi")],
            GenerationSettings(model="test-model", max_output_tokens=77, context_tokens=2048),
        )

    assert result.text == "hello"
    assert result.usage.prompt_tokens == 11
    assert result.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_generate_structured_validates_json_schema_response() -> None:
    class Extracted(BaseModel):
        intent: str

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["format"]["properties"]["intent"]["type"] == "string"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"intent":"recommend"}'}},
        )

    async with _http_client(httpx.MockTransport(handler)) as http_client:
        client = OllamaClient(_settings(), client=http_client)
        result = await client.generate_structured(
            [ChatMessage(role=MessageRole.USER, content="help")],
            Extracted,
            GenerationSettings(model="test-model"),
        )

    assert result.intent == "recommend"


@pytest.mark.asyncio
async def test_stream_text_yields_ndjson_chunks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"hel"},"done":false}\n'
                b'{"message":{"content":"lo"},"done":true,"done_reason":"stop"}\n'
            ),
        )

    async with _http_client(httpx.MockTransport(handler)) as http_client:
        client = OllamaClient(_settings(), client=http_client)
        chunks = [
            chunk
            async for chunk in client.stream_text(
                [ChatMessage(role=MessageRole.USER, content="hi")],
                GenerationSettings(model="test-model"),
            )
        ]

    assert "".join(chunk.content for chunk in chunks) == "hello"
    assert chunks[-1].done is True


@pytest.mark.asyncio
async def test_embedding_provider_supports_batches() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        assert json.loads(request.content)["input"] == ["first", "second"]
        return httpx.Response(200, json={"embeddings": [[0.6, 0.8], [1.0, 0.0]]})

    async with _http_client(httpx.MockTransport(handler)) as http_client:
        provider = OllamaEmbeddingProvider(_settings(), client=http_client)
        batch = await provider.embed_texts(["first", "second"])

    assert batch.dimensions == 2
    assert batch.embeddings[0] == [0.6, 0.8]


@pytest.mark.asyncio
async def test_missing_model_is_a_controlled_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    async with _http_client(httpx.MockTransport(handler)) as http_client:
        client = OllamaClient(_settings(), client=http_client)
        with pytest.raises(ModelNotInstalledError):
            await client.generate_text(
                [ChatMessage(role=MessageRole.USER, content="hi")],
                GenerationSettings(model="test-model"),
            )
