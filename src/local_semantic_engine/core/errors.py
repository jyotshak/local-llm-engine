"""Controlled application errors safe to expose through the API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LocalSemanticEngineError(Exception):
    """Base class for expected application failures."""

    code: str
    message: str
    retryable: bool = False
    details: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


class ConfigurationError(LocalSemanticEngineError):
    """Raised when a local configuration is invalid or unsafe."""

    def __init__(self, message: str, *, details: list[str] | None = None) -> None:
        super().__init__("CONFIGURATION_INVALID", message, False, details or [])


class CorpusNotReadyError(LocalSemanticEngineError):
    """Raised when the movie corpus or its index is unavailable."""

    def __init__(self, message: str = "The local movie corpus is not ready.") -> None:
        super().__init__("CORPUS_NOT_READY", message, False)


class OllamaUnavailableError(LocalSemanticEngineError):
    """Raised when the local Ollama server cannot be reached."""

    def __init__(self, message: str = "Ollama is unavailable on the configured local URL.") -> None:
        super().__init__("OLLAMA_UNAVAILABLE", message, True)


class ModelNotInstalledError(LocalSemanticEngineError):
    """Raised when Ollama does not have the requested local model."""

    def __init__(self, model: str) -> None:
        super().__init__("MODEL_NOT_INSTALLED", f"The configured model is not installed: {model}.")


class ModelTimeoutError(LocalSemanticEngineError):
    """Raised when a local model request exceeds its configured timeout."""

    def __init__(self) -> None:
        super().__init__("MODEL_TIMEOUT", "The local model request timed out.", True)


class ModelOutputInvalidError(LocalSemanticEngineError):
    """Raised when a structured model response does not validate."""

    def __init__(self, details: list[str]) -> None:
        super().__init__(
            "MODEL_OUTPUT_INVALID",
            "The local model returned invalid structured output.",
            False,
            details,
        )
