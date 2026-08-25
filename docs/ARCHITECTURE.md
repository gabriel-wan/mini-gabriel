# ARCHITECTURE.md — intended high-level architecture

This document describes the intended end-to-end architecture of mini-gabriel. Nothing described here is implemented yet unless explicitly marked otherwise. Specific technologies are deliberately **not** chosen at this stage.

## Pipeline overview

```
Telegram
   ↓
Data extraction
   ↓
Raw dataset
   ↓
Filtering / preprocessing
   ↓
Training examples
   ↓
Pre-trained LLM
   ↓
Parameter-efficient fine-tuning
   ↓
mini-gabriel model
   ↓
Chat interface
```

## Component status

| Component | Status |
|---|---|
| Data extraction (Telegram) | **implemented** (`extract.py`) — not yet run against real data |
| Raw dataset storage (`data/raw/`, local only) | **implemented** (`storage.py`) — JSONL per chat plus a manifest |
| Chat analysis / selection | **implemented** (`selection.py`, `analyze.py`) — not yet run against real data |
| Filtering / preprocessing into training examples | planned — not implemented yet |
| Training-example construction | planned — not implemented yet |
| Base pre-trained LLM | planned — model not selected |
| Parameter-efficient fine-tuning | planned — method/framework not selected |
| Fine-tuned mini-gabriel model | planned — not implemented yet |
| Chat interface | planned — not implemented yet |

## Why no RAG?

> RAG is not part of the core architecture because the objective is style imitation rather than factual memory retrieval.

The model is expected to learn *how* the author writes from fine-tuning, not to look up *what* the author knows at inference time.

## Notes

- The only implemented piece of the project so far is the repository structure itself (Python package skeleton, docs, data directories).
- Technology choices (model, fine-tuning framework, serving stack, interface) are tracked as open questions in [PROJECT.md](PROJECT.md) under "Not yet decided".
