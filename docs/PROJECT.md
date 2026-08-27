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
7. **Evaluation** — assess style similarity between model output and real messages (harness built, see [EVALUATION.md](EVALUATION.md))
8. **Chatbot interface** — a way to converse with the fine-tuned model

## Current status

| Area | Status |
|---|---|
| Repository foundation | complete |
| Telegram extraction | complete — 432 chats, 113,053 messages |
| Chat analysis / selection | complete — 58 of 432 chats qualify |
| Dataset construction | complete — 17,001 examples, 15,567 train / 1,434 holdout |
| Model selection | `unsloth/Qwen3-8B-Base` |
| Fine-tuning approach | LoRA decided; framework not decided |
| Evaluation | harness complete — style metric with a calibrated floor |
| Deployment | not decided |
| Training hardware | decided — A100-40 on the NUS SoC cluster |

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
- **Training hardware:** NUS SoC Compute Cluster, A100-40 on the `gpu`
  partition (see [CLUSTER.md](CLUSTER.md))
- **Fine-tuning method:** plain LoRA in bf16, not QLoRA - the memory budget
  does not require quantisation at 8B
- **Base model:** `unsloth/Qwen3-8B-Base`. A base model rather than an instruct
  one, because instruct tuning pushes exactly the polite, capitalised,
  single-paragraph style the evaluation scores furthest from mine, so
  fine-tuning would be spent undoing it. 8.19B parameters, 16.4 GB in bf16,
  roughly 20 GB with activations - comfortable on an A100-40.

  Qwen3.5-9B-Base was chosen first and rejected on inspection: its architecture
  is `Qwen3_5ForConditionalGeneration`, it ships a `processor_config.json`, and
  HuggingFace tags it image-text-to-text. The Qwen3.5 family is natively
  multimodal, which is why every Qwen3.5 notebook Unsloth publishes is a vision
  notebook. `FastLanguageModel` is the wrong loader for it. Qwen3-8B-Base is
  `Qwen3ForCausalLM`, text-generation, with no processor - and is the same
  family as the notebook the training code was taken from.

## Not yet decided

Do **not** assume answers to these — they are open questions:

- Whether a larger size beats 9B
- Fine-tuning framework
- Training hyperparameters
- Inference/deployment approach
- Chatbot interface

## Related documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — intended high-level architecture
- [DATA.md](DATA.md) — data strategy, selection criteria, privacy rules
- [EVALUATION.md](EVALUATION.md) — how style similarity is judged
- [CLUSTER.md](CLUSTER.md) — training hardware and its decisions
- [EXPERIMENTS.md](EXPERIMENTS.md) — ML experiment log
- [TODO.md](TODO.md) — concrete roadmap
