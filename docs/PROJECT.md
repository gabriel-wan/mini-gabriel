# PROJECT.md — mini-gabriel project context

> This is the main project context document. It is written so that a coding agent (Claude Code, Codex, etc.) or a human collaborator can read it and quickly understand the project's purpose, current decisions, non-goals, and status. Treat the `docs/` directory as the project's source of truth.

## Objective

Investigate whether a relatively small amount of personal conversational data — the author's own Telegram messages — can be used to fine-tune a pre-trained open-source LLM to imitate the author's **conversational style**.

"Style" means things like:

- wording and vocabulary
- sentence structure
- punctuation and capitalization habits
- slang and abbreviations
- emoji usage
- typical response length
- conversational patterns and how the author naturally responds to different kinds of messages

## Non-goals

These are explicitly **out of scope**:

- Factual memory retrieval (a bot that "knows things about" the author)
- RAG or any vector-database-backed retrieval
- Training an LLM from scratch
- Building a general-purpose assistant
- Reproducing the author's identity/personality perfectly — the target is style imitation, not a digital clone

## Current hypothesis

> A sufficiently capable instruction-tuned open-source LLM, fine-tuned on examples derived from my own conversations, may learn aspects of my conversational style.

This is a **hypothesis to test**, not an established result. Nothing has been trained yet.

## Planned stages

1. **Repository setup** — structure, documentation, tooling foundation
2. **Telegram data extraction** — pull the author's messages from Telegram
3. **Dataset analysis** — inspect the raw data, understand its shape and quality
4. **Dataset construction** — filter and transform raw messages into training examples
5. **Baseline model** — pick a base model and establish baseline (un-fine-tuned) behavior
6. **Fine-tuning** — parameter-efficient fine-tuning on the constructed dataset
7. **Evaluation** — assess style similarity between model output and real messages
8. **Chatbot interface** — a way to converse with the fine-tuned model

## Current status

| Area | Status |
|---|---|
| Repository foundation | complete |
| Telegram extraction | implemented and validated on a sample; full run pending |
| Chat analysis / selection | implemented and validated on a sample |
| Dataset construction | not started |
| Model selection | not decided |
| Fine-tuning approach | not decided |
| Deployment | not decided |

## Decisions

Decisions already made:

- **Project name:** mini-gabriel (Python package: `mini_gabriel`)
- **Goal:** conversational style imitation
- **RAG / memory retrieval:** out of scope
- **Training from scratch:** out of scope
- **Initial data source:** Telegram
- **Initial data window:** the 2026 calendar year, Asia/Singapore
- **Ingestion shape:** extraction and chat selection are separate stages; all
  selection logic is kept free of Telethon so it is testable

## Not yet decided

Do **not** assume answers to these — they are open questions:

- Exact base model
- Model size
- Fine-tuning framework
- LoRA vs QLoRA (or another parameter-efficient method)
- Training hyperparameters
- GPU/hardware
- Inference/deployment approach
- Chatbot interface

## Related documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — intended high-level architecture
- [DATA.md](DATA.md) — data strategy, selection criteria, privacy rules
- [EXPERIMENTS.md](EXPERIMENTS.md) — ML experiment log
- [TODO.md](TODO.md) — concrete roadmap
