# DATA.md — data strategy

This document describes the planned data strategy for mini-gabriel. Telegram extraction is **not implemented yet**.

## Data source

The initial (and currently only) data source is the author's own Telegram message history.

## Initial dataset-selection criteria

The criteria currently being considered for selecting which messages/chats enter the dataset:

- Messages from **2026 only**
- **Exclude channels**
- **Exclude group chats with more than 20 members**
- A candidate chat must contain **at least 100 messages authored by me**

These are **initial** criteria, not final ones. The dataset-selection process may be revised after inspecting the actual extracted data — e.g. thresholds may change, or additional quality filters may be added.

## Data layout

- `data/raw/` — raw extracted Telegram data. Local only, git-ignored.
- `data/processed/` — filtered/preprocessed data and constructed training examples. Local only, git-ignored.

Generated/processed datasets are **not** assumed safe to commit: they are derived from private conversations and stay local by default.

## Privacy rules

These rules are non-negotiable:

- Raw messages remain **local** to the author's machine.
- Raw messages must **never** be committed to Git.
- Telegram API credentials must **never** be committed.
- Session files (`*.session`, `*.session-journal`) must **never** be committed.
- The repository contains code and documentation — **not private conversations**.

The `.gitignore` at the repository root enforces these rules for the standard paths (`data/raw/`, `data/processed/`, `.env`, session files). Any new location that holds private data must be added to `.gitignore` before data is written there.
