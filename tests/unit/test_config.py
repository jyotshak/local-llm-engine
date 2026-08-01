from __future__ import annotations

from pathlib import Path

import pytest

from local_semantic_engine.config.loader import load_settings
from local_semantic_engine.core.errors import ConfigurationError


def test_environment_overrides_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        """
        [ollama]
        base_url = "http://127.0.0.1:19999"
        generation_model = "from-toml"

        [api]
        port = 9000
        """
    )

    settings = load_settings(
        config_path,
        environment={
            "LSE_OLLAMA__GENERATION_MODEL": "from-environment",
            "LSE_API__PORT": "9001",
        },
    )

    assert settings.ollama.base_url == "http://127.0.0.1:19999"
    assert settings.ollama.generation_model == "from-environment"
    assert settings.api.port == 9001


def test_non_loopback_host_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text('[api]\nhost = "0.0.0.0"\n')

    with pytest.raises(ConfigurationError):
        load_settings(config_path, environment={})


def test_standard_profiles_are_available() -> None:
    settings = load_settings(environment={})

    assert settings.profile("fast").rerank_candidate_count == 12
    assert settings.profile("balanced").rerank_candidate_count == 20
    assert settings.profile("quality").rerank_candidate_count == 30
