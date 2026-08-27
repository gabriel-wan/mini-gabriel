#!/usr/bin/env python
"""Convert the built dataset into chat messages for training.

Reads train.jsonl / holdout.jsonl and writes *_messages.jsonl, where each row is
{"messages": [{"role": ..., "content": ...}, ...]}.

Deliberately emits roles and content rather than a rendered chat template: the
tokenizer's apply_chat_template handles model-specific formatting at training
time, and stays correct when the model changes.

Never prints message text.
"""

import argparse
import json
import sys
from pathlib import Path

from mini_gabriel import config
from mini_gabriel.chatformat import DEFAULT_SYSTEM_PROMPT, convert_all


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"{path} not found. Run scripts/build_dataset.py first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="system message prepended to every example")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="omit the system message entirely")
    args = parser.parse_args()

    system_prompt = None if args.no_system_prompt else args.system_prompt
    out = config.PROCESSED_DIR

    try:
        for split in ("train", "holdout"):
            examples = load(out / f"{split}.jsonl")
            rows, counters = convert_all(examples, system_prompt)
            target = out / f"{split}_messages.jsonl"
            with target.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"\n  {split}: {len(examples):,} examples -> {counters['converted']:,} rows")
            print(f"    dropped, context was all mine : {counters['dropped_no_other_speaker']:,}")
            print(f"    dropped, context ended on me  : {counters['dropped_trailing_self']:,}")
            print(f"    -> {target}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
