"""Page-aware text extraction and indexing for local text-based PDFs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from local_semantic_engine.core.errors import CorpusNotReadyError
from local_semantic_engine.domains.documents.models import DocumentChunk
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex, new_manifest

DOCUMENT_REPRESENTATION_VERSION = "1"


def extract_pdf_chunks(
    directory: Path, *, chunk_size: int = 1200, overlap: int = 180
) -> list[DocumentChunk]:
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise CorpusNotReadyError(
            f"No PDFs found in {directory}. Add text-based PDFs and try again."
        )
    chunks: list[DocumentChunk] = []
    for pdf_path in pdfs:
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            for chunk_number, chunk_text in enumerate(
                _split_text(text, chunk_size, overlap), start=1
            ):
                chunk_id = f"{pdf_path.stem}:p{page_number}:c{chunk_number}"
                content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_name=pdf_path.name,
                        page_number=page_number,
                        chunk_number=chunk_number,
                        text=chunk_text,
                        content_hash=content_hash,
                    )
                )
    if not chunks:
        raise CorpusNotReadyError(
            "No extractable text was found. Version 1 requires text-based PDFs."
        )
    return chunks


async def build_document_index(
    *, chunks: list[DocumentChunk], embedding_provider, embedding_model: str, output_directory: Path
) -> int:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 16):
        batch = await embedding_provider.embed_texts(
            [chunk.text for chunk in chunks[start : start + 16]]
        )
        vectors.extend(batch.embeddings)
    index = NumpyVectorIndex([chunk.id for chunk in chunks], np.asarray(vectors, dtype=np.float32))
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "chunks.jsonl").write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    index.save(
        output_directory,
        new_manifest(
            embedding_model=embedding_model,
            dimensions=index.dimensions,
            representation_version=DOCUMENT_REPRESENTATION_VERSION,
            record_hashes={chunk.id: chunk.content_hash for chunk in chunks},
        ),
        prefix="document",
    )
    return len(chunks)


def load_document_chunks(path: Path) -> list[DocumentChunk]:
    try:
        chunks = [
            DocumentChunk.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line
        ]
    except FileNotFoundError as exc:
        raise CorpusNotReadyError(
            "Document index is unavailable. Run `lse corpus documents build` first."
        ) from exc
    if not chunks:
        raise CorpusNotReadyError("Document index has no chunks.")
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            end = boundary if boundary > start else end
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
