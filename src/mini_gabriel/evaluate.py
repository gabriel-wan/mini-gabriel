"""Evaluation harness.

Two halves, both GPU-free:

* ``write_prompts`` extracts the conversations from the held-out chats so a
  model can be asked to reply to them.
* ``score_generated`` compares those replies against what I actually said, using
  the style metrics in `style.py`.

Generation itself happens elsewhere and needs a GPU. Nothing here loads a model,
so the evaluation can be built and tested before any training has run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import config as project_config
from .style import compare, render_comparison, self_distance, style_profile

PROMPTS_NAME = "eval_prompts.jsonl"
REFERENCE_NAME = "eval_reference.json"
HOLDOUT_NAME = "holdout.jsonl"


def load_holdout(processed_dir: Optional[Path] = None) -> list[dict]:
    processed_dir = processed_dir or project_config.PROCESSED_DIR
    path = processed_dir / HOLDOUT_NAME
    if not path.exists():
        raise RuntimeError(f"No holdout set at {path}. Run scripts/build_dataset.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def example_id(example: dict, index: int) -> str:
    """Stable identifier so generated replies can be matched back."""
    return f"{example['chat_id']}:{index}"


def write_prompts(processed_dir: Optional[Path] = None, limit: Optional[int] = None) -> dict:
    """Write the held-out conversations, and the reference style profile.

    The reference profile is what I actually replied to these conversations, and
    the floor is that profile compared against itself. A model's score is only
    interpretable against the floor: two samples of the same writing do not
    score zero, so zero is not the target.
    """
    processed_dir = processed_dir or project_config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    holdout = load_holdout(processed_dir)
    if limit:
        holdout = holdout[:limit]

    prompts_path = processed_dir / PROMPTS_NAME
    with prompts_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(holdout):
            handle.write(
                json.dumps(
                    {
                        "id": example_id(example, index),
                        "chat_type": example["chat_type"],
                        "context": example["context"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    replies = [example["target"] for example in holdout]
    reference = style_profile(replies)
    floor = self_distance(replies)

    reference_path = processed_dir / REFERENCE_NAME
    with reference_path.open("w", encoding="utf-8") as handle:
        json.dump({"profile": reference, "floor": floor}, handle, ensure_ascii=False, indent=2)

    return {
        "prompts": len(holdout),
        "prompts_path": prompts_path,
        "reference_path": reference_path,
        "profile": reference,
        "floor": floor,
    }


def _reply_of(row: dict) -> list[str]:
    """Accept either a list of messages or a single newline-joined string."""
    if isinstance(row.get("reply"), list):
        return [str(part) for part in row["reply"]]
    text = row.get("reply") or row.get("text") or ""
    return [part for part in str(text).split("\n") if part.strip()]


def score_generated(
    generated_path: Path,
    processed_dir: Optional[Path] = None,
    label: str = "model",
) -> dict:
    """Score a file of generated replies against the held-out originals.

    The generated file is JSONL, one row per reply, with an ``id`` matching the
    prompts file and either ``reply`` (list of messages) or ``text`` (a single
    string, split on newlines).
    """
    processed_dir = processed_dir or project_config.PROCESSED_DIR

    if not generated_path.exists():
        raise RuntimeError(f"No generated replies at {generated_path}.")

    rows = [
        json.loads(line)
        for line in generated_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = {str(row.get("id")): _reply_of(row) for row in rows}

    holdout = load_holdout(processed_dir)
    reference_replies: list[list[str]] = []
    candidate_replies: list[list[str]] = []
    missing = 0

    for index, example in enumerate(holdout):
        key = example_id(example, index)
        if key not in generated:
            missing += 1
            continue
        reference_replies.append(example["target"])
        candidate_replies.append(generated[key])

    if not candidate_replies:
        raise RuntimeError(
            "No generated replies matched the holdout ids. Check that the "
            "generation step preserved the 'id' field from the prompts file."
        )

    # Compare against only the examples that were actually generated for, so a
    # partial generation run is not penalised for the replies it never made.
    result = compare(style_profile(reference_replies), style_profile(candidate_replies))
    result["label"] = label
    result["floor"] = self_distance(reference_replies)
    result["missing"] = missing
    result["matched"] = len(candidate_replies)
    return result


def render_score(result: dict) -> str:
    """Console rendering. Contains no message text."""
    text = render_comparison(result, title=f"style comparison: {result.get('label', 'model')}")
    floor = result.get("floor")
    distance = result.get("style_distance")

    lines = [text.rstrip("\n")]
    if floor is not None:
        lines.append(f"  floor (you vs you): {round(floor, 4)}")
        if distance is not None:
            ratio = distance / floor if floor else float("inf")
            verdict = (
                "indistinguishable on these metrics"
                if ratio <= 1.5
                else "close" if ratio <= 3
                else "measurably different"
            )
            lines.append(f"  ratio to floor    : {ratio:.1f}x  ->  {verdict}")
    if result.get("missing"):
        lines.append(f"  note: {result['missing']:,} holdout examples had no generated reply")
    lines.append("")
    return "\n".join(lines)
