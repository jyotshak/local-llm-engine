"""A small, exact NumPy vector index for local catalogues."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from local_semantic_engine.core.errors import CorpusNotReadyError
from local_semantic_engine.core.models import ScoredId


class IndexManifest(BaseModel):
    embedding_model: str
    dimensions: int = Field(gt=0)
    representation_version: str
    record_hashes: dict[str, str]
    created_at: str


class NumpyVectorIndex:
    """Exact cosine search over normalized float32 vectors."""

    def __init__(self, item_ids: Sequence[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(item_ids):
            raise ValueError("Vector rows must match item IDs.")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("The index requires non-empty vectors.")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Vector index IDs must be unique.")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("Vector index cannot contain zero vectors.")
        self._item_ids = list(item_ids)
        self._vectors = matrix / norms[:, np.newaxis]
        self._id_to_row = {item_id: row for row, item_id in enumerate(self._item_ids)}

    @property
    def dimensions(self) -> int:
        return int(self._vectors.shape[1])

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int,
        eligible_ids: Collection[str] | None = None,
    ) -> list[ScoredId]:
        if top_k < 1:
            return []
        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.shape[0] != self.dimensions:
            raise ValueError("Query dimensions do not match the index.")
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector cannot be zero.")
        scores = self._vectors @ (query_vector / query_norm)
        if eligible_ids is None:
            rows = np.arange(len(self._item_ids))
        else:
            rows = np.array(
                sorted(
                    self._id_to_row[item_id]
                    for item_id in eligible_ids
                    if item_id in self._id_to_row
                ),
                dtype=np.int64,
            )
        if rows.size == 0:
            return []
        ordered = rows[np.argsort(-scores[rows], kind="stable")[:top_k]]
        return [ScoredId(item_id=self._item_ids[row], score=float(scores[row])) for row in ordered]

    def save(self, directory: Path, manifest: IndexManifest) -> None:
        """Persist matrix, ID mapping, and manifest through atomic replacements."""

        if manifest.dimensions != self.dimensions:
            raise ValueError("Manifest dimensions do not match the index.")
        if set(manifest.record_hashes) != set(self._item_ids):
            raise ValueError("Manifest record hashes do not match index IDs.")
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_numpy_save(directory / "movie_vectors.npy", self._vectors)
        _atomic_text_write(directory / "movie_ids.json", json.dumps(self._item_ids, indent=2))
        _atomic_text_write(
            directory / "movie_index_manifest.json",
            manifest.model_dump_json(indent=2),
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        embedding_model: str,
        representation_version: str,
        record_hashes: dict[str, str],
    ) -> NumpyVectorIndex:
        """Load a compatible index or raise a controlled readiness error."""

        try:
            ids = json.loads((directory / "movie_ids.json").read_text(encoding="utf-8"))
            manifest = IndexManifest.model_validate_json(
                (directory / "movie_index_manifest.json").read_text(encoding="utf-8")
            )
            vectors = np.load(directory / "movie_vectors.npy", allow_pickle=False)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
            raise CorpusNotReadyError(
                "The local movie index is unavailable or unreadable."
            ) from exc
        if manifest.embedding_model != embedding_model:
            raise CorpusNotReadyError("The movie index uses a different embedding model.")
        if manifest.representation_version != representation_version:
            raise CorpusNotReadyError(
                "The movie index uses a different text representation version."
            )
        if manifest.record_hashes != record_hashes:
            raise CorpusNotReadyError("The movie index does not match the current movie corpus.")
        if not isinstance(ids, list) or not all(isinstance(item_id, str) for item_id in ids):
            raise CorpusNotReadyError("The movie index ID mapping is invalid.")
        try:
            index = cls(ids, vectors)
        except ValueError as exc:
            raise CorpusNotReadyError("The movie index vectors are invalid.") from exc
        if index.dimensions != manifest.dimensions:
            raise CorpusNotReadyError("The movie index dimensions do not match its manifest.")
        return index


def new_manifest(
    *,
    embedding_model: str,
    dimensions: int,
    representation_version: str,
    record_hashes: dict[str, str],
) -> IndexManifest:
    return IndexManifest(
        embedding_model=embedding_model,
        dimensions=dimensions,
        representation_version=representation_version,
        record_hashes=record_hashes,
        created_at=datetime.now(UTC).isoformat(),
    )


def _atomic_text_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_numpy_save(path: Path, vectors: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.save(temporary_path, vectors)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
