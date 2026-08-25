"""Dataset builder.

Reads the raw extraction, applies the turn/session/example logic, and writes
training examples to the git-ignored processed directory.

This module does the I/O; every decision lives in `turns.py` and `examples.py`,
which are pure and tested. Nothing here prints or writes message text outside
the dataset files themselves.

The output is deliberately model-agnostic: examples are structured context and
target, not a formatted chat template. No base model has been chosen yet, and
formatting for a specific one is a later, cheap step.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

from . import config as project_config
from .analyze import analyse_chats, load_manifest_or_fail
from .examples import (
    BuildConfig,
    assign_pseudonyms,
    build_examples,
    select_holdout_chats,
)
from .selection import SelectionCriteria
from .storage import chat_jsonl_path, iter_records
from .turns import build_turns, split_sessions

TRAIN_NAME = "train.jsonl"
HOLDOUT_NAME = "holdout.jsonl"
PSEUDONYM_MAP_NAME = "pseudonym_map.json"
SUMMARY_NAME = "dataset_summary.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]

    return {
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 1),
    }


def build_dataset(
    build_config: Optional[BuildConfig] = None,
    criteria: Optional[SelectionCriteria] = None,
    holdout_count: int = 4,
    output_dir: Optional[Path] = None,
) -> dict:
    """Build the dataset. Returns a summary containing counts only."""
    build_config = build_config or BuildConfig()
    criteria = criteria or SelectionCriteria()
    output_dir = output_dir or project_config.PROCESSED_DIR

    manifest = load_manifest_or_fail(project_config.MANIFEST_PATH)
    results = analyse_chats(manifest, project_config.CHATS_DIR, criteria)
    qualifying = [stats for stats, decision in results if decision.included]

    if not qualifying:
        raise RuntimeError(
            "No chats qualify under the current criteria, so there is nothing to build."
        )

    # analyse_chats returns chats ranked by volume, which is what the holdout
    # selector expects.
    holdout_ids = set(select_holdout_chats([s.chat_id for s in qualifying], holdout_count))

    train: list[dict] = []
    holdout: list[dict] = []
    pseudonym_map: dict[str, dict[str, str]] = {}
    totals = {"turns_mine": 0, "trivial_seen": 0, "trivial_kept": 0, "no_context": 0}
    target_lengths: list[int] = []
    messages_per_target: list[int] = []
    per_chat: list[dict] = []

    for stats in qualifying:
        records = iter_records(chat_jsonl_path(project_config.CHATS_DIR, stats.chat_id))
        turns = build_turns(records, build_config.burst_gap_seconds)
        sessions = split_sessions(turns, build_config.session_gap_seconds)
        pseudonyms = assign_pseudonyms(turns)

        examples, counters = build_examples(
            stats.chat_id, stats.chat_type, sessions, pseudonyms, build_config
        )

        for key, value in counters.items():
            totals[key] += value

        # The map holds real sender ids and stays local; data/processed is
        # git-ignored, like everything else derived from private conversations.
        pseudonym_map[str(stats.chat_id)] = {
            str(sender): label for sender, label in pseudonyms.items()
        }

        bucket = holdout if stats.chat_id in holdout_ids else train
        bucket.extend(examples)

        for example in examples:
            target_lengths.append(len("\n".join(example["target"])))
            messages_per_target.append(len(example["target"]))

        per_chat.append(
            {
                "chat_id": stats.chat_id,
                "chat_type": stats.chat_type,
                "split": "holdout" if stats.chat_id in holdout_ids else "train",
                "sessions": len(sessions),
                "turns_total": len(turns),
                "examples": len(examples),
            }
        )

    _write_jsonl(output_dir / TRAIN_NAME, train)
    _write_jsonl(output_dir / HOLDOUT_NAME, holdout)

    with (output_dir / PSEUDONYM_MAP_NAME).open("w", encoding="utf-8") as handle:
        json.dump(pseudonym_map, handle, ensure_ascii=False, indent=2)

    summary = {
        "config": {
            "burst_gap_seconds": build_config.burst_gap_seconds,
            "session_gap_seconds": build_config.session_gap_seconds,
            "context_turns": build_config.context_turns,
            "context_token_budget": build_config.context_token_budget,
            "trivial_keep_rate": build_config.trivial_keep_rate,
            "seed": build_config.seed,
            "min_my_text_messages": criteria.min_my_text_messages,
            "max_participants": criteria.max_participants,
            "holdout_count": holdout_count,
        },
        "totals": {
            "chats": len(qualifying),
            "chats_train": len(qualifying) - len(holdout_ids),
            "chats_holdout": len(holdout_ids),
            "examples_train": len(train),
            "examples_holdout": len(holdout),
            "my_turns_seen": totals["turns_mine"],
            "trivial_turns_seen": totals["trivial_seen"],
            "trivial_turns_kept": totals["trivial_kept"],
            "dropped_no_context": totals["no_context"],
        },
        "target_length_chars": _percentiles(target_lengths),
        "messages_per_target": _percentiles(messages_per_target),
        "per_chat": per_chat,
    }

    with (output_dir / SUMMARY_NAME).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def render_summary(summary: dict) -> str:
    """Console rendering. Counts only - never any message text."""
    totals = summary["totals"]
    config = summary["config"]
    lengths = summary.get("target_length_chars", {})
    per_target = summary.get("messages_per_target", {})

    kept = totals["trivial_turns_kept"]
    seen = totals["trivial_turns_seen"]
    rate = f"{kept * 100 // seen}%" if seen else "n/a"

    return "\n".join(
        [
            "",
            "Dataset build",
            "-------------",
            f"  chats            : {totals['chats']} "
            f"({totals['chats_train']} train, {totals['chats_holdout']} holdout)",
            f"  examples         : {totals['examples_train']:,} train, "
            f"{totals['examples_holdout']:,} holdout",
            "",
            f"  my turns seen    : {totals['my_turns_seen']:,}",
            f"  trivial turns    : {seen:,} seen, {kept:,} kept ({rate} "
            f"at rate {config['trivial_keep_rate']})",
            f"  dropped, no ctx  : {totals['dropped_no_context']:,}",
            "",
            f"  target length    : p50 {lengths.get('p50')} chars, "
            f"p90 {lengths.get('p90')}, max {lengths.get('max')}",
            f"  msgs per target  : mean {per_target.get('mean')}, "
            f"p90 {per_target.get('p90')}, max {per_target.get('max')}",
            "",
        ]
    )
