# Local LLM Engine

This repository contains the implementation of the local semantic-processing framework described in [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md).

The current stage establishes the project foundation. It does not download models or movie data automatically.

## Planned commands

```powershell
lse doctor
lse corpus movies build
lse index movies build
lse recommend movies "thoughtful science fiction under two hours"
lse serve
```
