from __future__ import annotations

import numpy as np
import pytest

from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.domains.documents.models import (
    DocumentChunk,
    DocumentQuestionRequest,
    GroundedAnswer,
)
from local_semantic_engine.domains.documents.qa import DocumentAnswerer
from local_semantic_engine.ingestion.documents.pdf import _split_text
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex


class FakeGenerator:
    async def generate_structured(self, messages, schema, settings) -> GroundedAnswer:
        return GroundedAnswer(
            answer="The evidence supports this answer.",
            cited_chunk_ids=["book:p2:c1", "appendix:p1:c1", "unknown"],
            insufficient_evidence=False,
        )


class FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_document_answerer_returns_validated_page_citations() -> None:
    chunks = [
        DocumentChunk(
            id="book:p2:c1",
            document_name="book.pdf",
            page_number=2,
            chunk_number=1,
            text="Evidence",
        ),
        DocumentChunk(
            id="appendix:p1:c1",
            document_name="appendix.pdf",
            page_number=1,
            chunk_number=1,
            text="More evidence",
        ),
    ]
    answerer = DocumentAnswerer(
        settings=AppSettings(),
        generator=FakeGenerator(),
        embedder=FakeEmbedder(),
        index=NumpyVectorIndex([chunk.id for chunk in chunks], np.array([[1.0, 0.0], [0.8, 0.2]])),
        chunks=chunks,
    )

    response = await answerer.answer(DocumentQuestionRequest(question="What is supported?"))

    assert [(item.document_name, item.page_number) for item in response.citations] == [
        ("book.pdf", 2),
        ("appendix.pdf", 1),
    ]
    assert not response.uncertain


def test_split_text_retains_overlapping_page_chunks() -> None:
    chunks = _split_text("one two three four five six", chunk_size=13, overlap=4)

    assert len(chunks) > 1
    assert "two" in chunks[0]
    assert "two" in chunks[1]
