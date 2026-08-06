"""Grounded local document question answering with page citations."""

from __future__ import annotations

from uuid import uuid4

from local_semantic_engine.config.models import AppSettings
from local_semantic_engine.core.models import ChatMessage, GenerationSettings
from local_semantic_engine.domains.documents.models import (
    DocumentAnswerResponse,
    DocumentChunk,
    DocumentCitation,
    DocumentQuestionRequest,
    GroundedAnswer,
)
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex


class DocumentAnswerer:
    def __init__(
        self,
        *,
        settings: AppSettings,
        generator,
        embedder,
        index: NumpyVectorIndex,
        chunks: list[DocumentChunk],
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.embedder = embedder
        self.index = index
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}

    async def answer(self, request: DocumentQuestionRequest) -> DocumentAnswerResponse:
        query_vector = await self.embedder.embed_query(request.question)
        matches = self.index.search(query_vector, top_k=request.max_chunks)
        chunks = [self.chunks_by_id[match.item_id] for match in matches]
        profile = self.settings.profile(request.profile.value)
        evidence = "\n\n".join(
            f"CHUNK_ID: {chunk.id}\nDOCUMENT: {chunk.document_name}\nPAGE: {chunk.page_number}\n"
            f"TEXT: {chunk.text[: profile.candidate_summary_characters * 2]}"
            for chunk in chunks
        )
        answer = await self.generator.generate_structured(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Answer only from the supplied evidence. If it does not support an "
                        "answer, set insufficient_evidence=true and explain what is missing. "
                        "Cite only supplied CHUNK_ID values that support the answer. Do not "
                        "use outside knowledge."
                    ),
                ),
                ChatMessage(
                    role="user", content=f"Question:\n{request.question}\n\nEvidence:\n{evidence}"
                ),
            ],
            GroundedAnswer,
            GenerationSettings(
                model=self.settings.ollama.generation_model,
                temperature=0.0,
                max_output_tokens=profile.max_output_tokens,
                context_tokens=profile.context_tokens,
                thinking=False,
                keep_alive=self.settings.ollama.keep_alive,
            ),
        )
        valid_ids = [
            chunk_id for chunk_id in answer.cited_chunk_ids if chunk_id in self.chunks_by_id
        ]
        cited = [self.chunks_by_id[chunk_id] for chunk_id in dict.fromkeys(valid_ids)]
        citations = list(
            {
                (chunk.document_name, chunk.page_number): DocumentCitation(
                    document_name=chunk.document_name, page_number=chunk.page_number
                )
                for chunk in cited
            }.values()
        )
        uncertain = answer.insufficient_evidence or not citations
        warnings = [] if citations else ["No validated source citation was returned."]
        return DocumentAnswerResponse(
            answer=answer.answer,
            citations=citations,
            uncertain=uncertain,
            warnings=warnings,
            trace_id=str(uuid4()),
        )
