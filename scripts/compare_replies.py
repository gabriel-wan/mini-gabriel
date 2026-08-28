#!/usr/bin/env python
"""Read generated replies next to the real ones, for the same conversation.

The style metric is blind to whether a reply makes sense. It measures length,
capitalisation, burst structure and slang, so a model producing well-shaped
nonsense scores well. This is the layer that catches that: a person reading the
output.

Two modes:

  default   labelled, for reading and forming a judgement
  --blind   the two replies are shuffled and unlabelled, so you have to guess.
            An answer key prints at the end. This is the quiz to give friends.

Unlike everything else in this project, this prints message text - that is its
entire purpose. It writes nothing to disk, so nothing here can be committed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from mini_gabriel import config
from mini_gabriel.evaluate import example_id, load_holdout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("generated", type=Path, help="a replies JSONL from generate.py")
    p.add_argument("--count", type=int, default=10, help="how many to show")
    p.add_argument("--context-turns", type=int, default=3,
                   help="turns of conversation to show above each reply")
    p.add_argument("--blind", action="store_true",
                   help="hide which reply is which; print an answer key at the end")
    p.add_argument("--seed", type=int, default=0, help="which sample you get")
    p.add_argument("--min-length", type=int, default=0,
                   help="only show examples where your real reply is at least this long, "
                        "to skip the one-word ones")
    return p.parse_args()


def show(messages: list[str], indent: str = "         ") -> str:
    """Render a reply's messages, one per line, as they would arrive."""
    if not messages:
        return indent + "(nothing)"
    return ("\n" + indent).join(messages)


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = [
        json.loads(line)
        for line in args.generated.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = {str(row["id"]): row.get("reply") or [] for row in rows}

    pairs = []
    for index, example in enumerate(load_holdout()):
        key = example_id(example, index)
        if key not in generated:
            continue
        if len("\n".join(example["target"])) < args.min_length:
            continue
        pairs.append((example, generated[key]))

    if not pairs:
        print("error: no generated replies matched the holdout set.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.count, len(pairs)))

    answer_key = []
    for number, (example, model_reply) in enumerate(sample, start=1):
        print(f"\n{'─' * 66}")
        print(f"  {number}.  ({example['chat_type']})")
        print(f"{'─' * 66}")

        for turn in example["context"][-args.context_turns :]:
            who = "me " if turn["speaker"] == "me" else f"{turn['speaker']}  "
            for line in turn["text"].split("\n"):
                print(f"  {who:<5} {line}")

        real = list(example["target"])
        if args.blind:
            options = [("you", real), ("model", model_reply)]
            rng.shuffle(options)
            answer_key.append("A" if options[0][0] == "you" else "B")
            print(f"\n     A   {show(options[0][1], '         ')}")
            print(f"     B   {show(options[1][1], '         ')}")
        else:
            print(f"\n     you    {show(real, '            ')}")
            print(f"     model  {show(model_reply, '            ')}")

    if args.blind:
        print(f"\n{'─' * 66}")
        print("  answer key (which letter was really you):")
        print("  " + " ".join(f"{i}{letter}" for i, letter in enumerate(answer_key, 1)))
        print(f"{'─' * 66}\n")
    else:
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
