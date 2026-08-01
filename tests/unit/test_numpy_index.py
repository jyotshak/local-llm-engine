from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from local_semantic_engine.core.errors import CorpusNotReadyError
from local_semantic_engine.retrieval.numpy_index import NumpyVectorIndex, new_manifest


def test_exact_search_and_eligible_filter() -> None:
    index = NumpyVectorIndex(
        ["a", "b", "c"],
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
    )

    results = index.search([1.0, 0.0], top_k=2, eligible_ids={"b", "c"})

    assert [result.item_id for result in results] == ["b", "c"]


def test_index_manifest_mismatch_is_controlled(tmp_path: Path) -> None:
    index = NumpyVectorIndex(["a"], np.array([[1.0, 0.0]]))
    manifest = new_manifest(
        embedding_model="embeddinggemma",
        dimensions=2,
        representation_version="1",
        record_hashes={"a": "hash"},
    )
    index.save(tmp_path, manifest)

    with pytest.raises(CorpusNotReadyError):
        NumpyVectorIndex.load(
            tmp_path,
            embedding_model="different-model",
            representation_version="1",
            record_hashes={"a": "hash"},
        )
