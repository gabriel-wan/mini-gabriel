#!/usr/bin/env python
"""CLI for the dataset-construction stage.

Reads the raw extraction and writes training examples to data/processed/.
Safe to re-run: output files are rewritten from scratch each time, and the
build is deterministic, so the same inputs and settings give the same dataset.

Never prints message text.
"""

import argparse
import sys

from mini_gabriel import config
from mini_gabriel.build import build_dataset, render_summary
from mini_gabriel.examples import BuildConfig
from mini_gabriel.selection import SelectionCriteria


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--burst-gap", type=int, default=300, metavar="SECONDS",
                        help="longest silence still counted as one turn (default: 300)")
    parser.add_argument("--session-gap", type=int, default=10800, metavar="SECONDS",
                        help="silence that starts a new conversation (default: 10800)")
    parser.add_argument("--context-turns", type=int, default=10,
                        help="turns of history per example (default: 10)")
    parser.add_argument("--context-tokens", type=int, default=1024,
                        help="approximate token budget for context (default: 1024)")
    parser.add_argument("--trivial-keep-rate", type=float, default=0.33,
                        help="share of all-acknowledgement turns to keep (default: 0.33)")
    parser.add_argument("--holdout-chats", type=int, default=4,
                        help="whole chats reserved for evaluation (default: 4)")
    parser.add_argument("--seed", default="mini-gabriel",
                        help="seed for deterministic downsampling")
    parser.add_argument("--min-my-text-messages", type=int, default=config.MIN_MY_TEXT_MESSAGES)
    parser.add_argument("--max-participants", type=int, default=config.MAX_PARTICIPANTS)
    args = parser.parse_args()

    build_config = BuildConfig(
        burst_gap_seconds=args.burst_gap,
        session_gap_seconds=args.session_gap,
        context_turns=args.context_turns,
        context_token_budget=args.context_tokens,
        trivial_keep_rate=args.trivial_keep_rate,
        seed=args.seed,
    )
    criteria = SelectionCriteria(
        max_participants=args.max_participants,
        min_my_text_messages=args.min_my_text_messages,
    )

    try:
        summary = build_dataset(build_config, criteria, args.holdout_chats)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_summary(summary))
    print(f"  train   : {config.PROCESSED_DIR / 'train.jsonl'}")
    print(f"  holdout : {config.PROCESSED_DIR / 'holdout.jsonl'}")
    print(f"  summary : {config.PROCESSED_DIR / 'dataset_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
