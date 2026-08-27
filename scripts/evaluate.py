#!/usr/bin/env python
"""CLI for the evaluation harness.

Two modes, neither of which needs a GPU:

  prompts   write the held-out conversations for a model to reply to,
            and record my own style profile as the reference
  score     compare a file of generated replies against what I actually said

Generation happens between the two, on a GPU, and is not part of this script.
Never prints message text.
"""

import argparse
import sys
from pathlib import Path

from mini_gabriel import config
from mini_gabriel.evaluate import render_score, score_generated, write_prompts
from mini_gabriel.style import render_comparison


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("prompts", help="write held-out prompts and the reference profile")
    p.add_argument("--limit", type=int, default=None,
                   help="only use the first N held-out examples")

    s = sub.add_parser("score", help="score generated replies")
    s.add_argument("generated", type=Path, help="JSONL of generated replies")
    s.add_argument("--label", default="model", help="name for this run in the output")

    args = parser.parse_args()

    try:
        if args.mode == "prompts":
            result = write_prompts(limit=args.limit)
            profile = result["profile"]
            print(f"\n  wrote {result['prompts']:,} prompts -> {result['prompts_path']}")
            print(f"  reference profile     -> {result['reference_path']}")
            print(f"\n  your style, measured on the held-out replies:")
            for key, value in profile.items():
                if key != "n":
                    print(f"    {key:<22}{value:>10}")
            print(f"\n  floor (you vs you)    : {round(result['floor'], 4)}")
            print("  a model scoring near this is indistinguishable on these metrics\n")
        else:
            print(render_score(score_generated(args.generated, label=args.label)))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
