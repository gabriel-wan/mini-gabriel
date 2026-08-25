#!/usr/bin/env python
"""CLI for the chat-analysis stage.

Reads the raw extraction and reports which chats qualify as training data.
Safe to re-run with different thresholds; it never modifies raw data and never
prints or writes message text.
"""

import argparse
import sys

from mini_gabriel import config
from mini_gabriel.analyze import (
    analyse_chats,
    build_report,
    load_manifest_or_fail,
    render_terminal_summary,
    write_report,
)
from mini_gabriel.selection import SelectionCriteria


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-participants", type=int, default=config.MAX_PARTICIPANTS)
    parser.add_argument("--min-my-text-messages", type=int, default=config.MIN_MY_TEXT_MESSAGES)
    parser.add_argument("--top", type=int, default=25,
                        help="how many chats to show in the console summary")
    args = parser.parse_args()

    criteria = SelectionCriteria(
        max_participants=args.max_participants,
        min_my_text_messages=args.min_my_text_messages,
    )

    try:
        manifest = load_manifest_or_fail(config.MANIFEST_PATH)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = analyse_chats(manifest, config.CHATS_DIR, criteria)
    report = build_report(manifest, results, criteria)
    json_path, markdown_path = write_report(report, config.PROCESSED_DIR)

    print(render_terminal_summary(report, limit=args.top))
    print(f"  full report: {json_path}")
    print(f"               {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
