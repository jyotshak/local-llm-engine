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
lse recommend movies "thoughtful science fiction under two hours"
```

`lse search movies` is a retrieval diagnostic. `lse recommend movies` uses Qwen
locally to interpret and rank candidates, while the app enforces hard constraints
such as runtime against local records before and after Qwen's response.

## Local API

Start the loopback-only API:

```powershell
lse serve
```

Then use `http://127.0.0.1:8765/docs` for interactive local API documentation.
The main endpoints are `GET /health`, `POST /v1/movies/recommend`,
`POST /v1/movies/recommend/stream` (server-sent progress events followed by a
validated result), and `POST /v1/documents/answer`.

## Evaluation

Run the tracked smoke cases after changing prompts, models, or retrieval logic:

```powershell
lse evaluate movies --report data/evaluation/reports/movie_latest.json
```

The report measures recommendation count, configured factual constraints, and
per-case latency. It deliberately does not claim subjective recommendation quality.

## Document Q&A

Place text-based PDFs in `data/documents/raw/`, then build a separate local index:

```powershell
lse corpus documents build
lse ask documents "What does the author say about the main theme?"
```

Answers are grounded in retrieved page-aware chunks and return a de-duplicated list
of filename/page-number citations. Image-only or scanned PDFs are outside Version 1.
