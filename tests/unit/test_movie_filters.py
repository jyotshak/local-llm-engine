from __future__ import annotations

from local_semantic_engine.domains.movies.filters import apply_hard_constraints
from local_semantic_engine.domains.movies.models import (
    EvidenceState,
    MissingDataPolicy,
    MovieHardConstraints,
    MovieRecord,
)


def test_strict_unknown_content_signal_excludes_movie() -> None:
    movie = MovieRecord(id="a", title="Unknown", runtime_minutes=90)
    constraints = MovieHardConstraints(excluded_content_attributes=["graphic_gore"])

    result = apply_hard_constraints(
        [movie], constraints, missing_data_policy=MissingDataPolicy.STRICT
    )

    assert result.eligible == []
    assert "unknown excluded content attribute: graphic_gore" in result.excluded_reasons["a"]


def test_permissive_unknown_content_signal_remains_eligible() -> None:
    movie = MovieRecord(id="a", title="Unknown", runtime_minutes=90)
    constraints = MovieHardConstraints(excluded_content_attributes=["graphic_gore"])

    result = apply_hard_constraints(
        [movie], constraints, missing_data_policy=MissingDataPolicy.ALLOW_WITH_WARNING
    )

    assert [item.id for item in result.eligible] == ["a"]
    assert result.missing_evidence == ["content attribute: graphic_gore"]


def test_present_excluded_content_signal_is_always_filtered() -> None:
    movie = MovieRecord(id="a", title="Gory")
    movie.semantic_attributes.content_warnings["graphic_gore"] = EvidenceState.PRESENT
    constraints = MovieHardConstraints(excluded_content_attributes=["graphic_gore"])

    result = apply_hard_constraints(
        [movie], constraints, missing_data_policy=MissingDataPolicy.ALLOW_WITH_WARNING
    )

    assert result.eligible == []
