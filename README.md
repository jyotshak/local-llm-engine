# Local LLM Engine

This repository contains the implementation of the local semantic-processing framework described in [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md).

The current stage establishes the project foundation. It does not download models or movie data automatically.

## Local setup

Install Ollama for Windows, then pull the two configured local models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:8b
```

If you want model weights on a non-system drive, set `OLLAMA_MODELS` before starting
Ollama. This workspace uses `X:\LLM\models`.

## Available commands

```powershell
lse doctor
lse corpus movies build
lse index movies
lse search movies "thoughtful science fiction under two hours"
```

`lse search movies` is the currently available retrieval diagnostic. The next
milestone adds hard-constraint filtering, Qwen ranking/explanations, then the
localhost API and streaming chat endpoint.
