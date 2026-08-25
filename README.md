# mini-gabriel

An experiment in teaching an open-source LLM to talk like me.

## What is this?

mini-gabriel is a personal ML/GenAI project exploring whether a relatively small amount of personal conversational data (my own Telegram messages) can be used to fine-tune a pre-trained open-source LLM so that it imitates my **conversational style** — wording, vocabulary, sentence structure, punctuation and capitalization habits, slang, abbreviations, emoji usage, typical response length, and how I naturally respond to different kinds of messages.

## The problem being explored

Large language models write like large language models. This project asks a narrower question: given a modest, personal dataset of real conversations, can parameter-efficient fine-tuning make a model *sound* like a specific person?

## Style imitation, not memory

This is a key distinction:

- **In scope:** learning *how* I write — tone, phrasing, rhythm, habits.
- **Out of scope:** learning *what* I know — facts about my life, my contacts, my history.

RAG and memory-retrieval systems are explicitly **not** part of the core project. The goal is not a chatbot that remembers things about me; it is a chatbot that responds the way I would.

Training an LLM from scratch is also out of scope — the starting point is a pre-trained open-source model.

## Planned pipeline

```
Telegram messages
    → data extraction
    → preprocessing
    → training dataset
    → pre-trained open-source LLM
    → parameter-efficient fine-tuning
    → personalized chatbot
```

The specific base model, fine-tuning method, framework, and deployment environment have not been chosen yet.

## Current status

The project is at the foundation stage: repository structure and documentation only. No data extraction, training, or evaluation has happened yet, so no claims are made about results or model quality.

See [docs/PROJECT.md](docs/PROJECT.md) for the full project context, decisions, and open questions.

## ⚠️ Privacy note

This project works with **private personal data** (real Telegram conversations). Raw Telegram data, API credentials, and session files live only on the local machine and must **never** be committed to this repository. The repository contains code and documentation only — never private conversations. See [docs/DATA.md](docs/DATA.md) for the full privacy rules.
