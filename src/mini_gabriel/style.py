"""Style profiling and comparison.

Measures the surface habits that make writing recognisable - length, burst
structure, capitalisation, punctuation, emoji, slang - and compares two sets of
replies on those measures.

This exists because training loss cannot answer the question the project is
actually asking. A model that memorised the training set scores excellent loss
and is worthless; a model that writes fluent, tidy, correctly punctuated
sentences scores well on every generic benchmark and sounds nothing like the
author. The metrics here are deliberately shallow, because shallow is exactly
what "style" means at this level.

Pure functions: no I/O, no model, no GPU.
"""

from __future__ import annotations

import random
import re
import statistics
from typing import Iterable, Mapping, Sequence

# Reply = the messages of one turn, matching the "target" field of an example.
Reply = Sequence[str]

TRIVIAL_MAX_CHARS = 5

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
TERMINAL_PUNCT = (".", "!", "?")
SINGLISH = re.compile(
    r"\b(lah|leh|lor|sia|hor|meh|liao|walao|wah|aiya|aiyo|siao|shiok|chio|"
    r"paiseh|kiasu|bojio|abit|alr|damn|confirm)\b",
    re.IGNORECASE,
)

# Metrics expressed as a rate in [0, 1]; compared by absolute difference.
RATE_METRICS = (
    "trivial_rate",
    "emoji_rate",
    "lowercase_start_rate",
    "terminal_punct_rate",
    "singlish_rate",
    "question_rate",
    "multi_message_rate",
)
# Metrics on an open numeric scale; compared by relative difference.
SCALE_METRICS = ("mean_chars", "median_chars", "mean_messages")

ALL_METRICS = RATE_METRICS + SCALE_METRICS


def _joined(reply: Reply) -> str:
    return "\n".join(message.strip() for message in reply if message is not None)


def style_profile(replies: Iterable[Reply]) -> dict:
    """Measure the style of a set of replies.

    Each reply is the list of messages in one turn, so burst structure survives
    into the measurement rather than being flattened away.
    """
    replies = [list(reply) for reply in replies if list(reply)]
    count = len(replies)
    if not count:
        return {"n": 0}

    joined = [_joined(reply) for reply in replies]
    lengths = [len(text) for text in joined]
    message_counts = [len(reply) for reply in replies]

    def rate(predicate) -> float:
        return sum(1 for item in replies if predicate(item)) / count

    return {
        "n": count,
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": statistics.median(lengths),
        "mean_messages": round(statistics.mean(message_counts), 2),
        "multi_message_rate": round(rate(lambda r: len(r) > 1), 4),
        "trivial_rate": round(
            rate(lambda r: all(len(m.strip()) <= TRIVIAL_MAX_CHARS for m in r)), 4
        ),
        "emoji_rate": round(rate(lambda r: bool(EMOJI.search(_joined(r)))), 4),
        "lowercase_start_rate": round(
            rate(lambda r: _joined(r)[:1].islower()), 4
        ),
        "terminal_punct_rate": round(
            rate(lambda r: _joined(r).endswith(TERMINAL_PUNCT)), 4
        ),
        "singlish_rate": round(rate(lambda r: bool(SINGLISH.search(_joined(r)))), 4),
        "question_rate": round(rate(lambda r: "?" in _joined(r)), 4),
    }


def _normalised_difference(metric: str, reference: float, candidate: float) -> float:
    """Difference on a comparable scale, so metrics can be averaged."""
    if metric in RATE_METRICS:
        return abs(reference - candidate)
    if reference == 0:
        return 0.0 if candidate == 0 else 1.0
    # Relative error, capped so one wild metric cannot dominate the average.
    return min(abs(reference - candidate) / abs(reference), 1.0)


def compare(reference: Mapping, candidate: Mapping) -> dict:
    """Compare two style profiles.

    Returns per-metric detail plus ``style_distance``: the mean normalised
    difference across metrics, where 0 is identical and larger is further away.

    The number is only meaningful against a baseline. Use ``self_distance`` to
    establish the floor - two samples of genuinely the same writing do not score
    0, and a model cannot be expected to beat that.
    """
    metrics = {}
    differences = []

    for metric in ALL_METRICS:
        if metric not in reference or metric not in candidate:
            continue
        ref_value = reference[metric]
        cand_value = candidate[metric]
        difference = _normalised_difference(metric, ref_value, cand_value)
        differences.append(difference)
        metrics[metric] = {
            "reference": ref_value,
            "candidate": cand_value,
            "difference": round(cand_value - ref_value, 4),
            "normalised": round(difference, 4),
        }

    return {
        "style_distance": round(statistics.mean(differences), 4) if differences else None,
        "metrics": metrics,
        "reference_n": reference.get("n", 0),
        "candidate_n": candidate.get("n", 0),
    }


def self_distance(replies: Sequence[Reply], seed: int = 0) -> float:
    """The distance between two halves of the same set of replies.

    This is the floor. Real writing compared against itself does not score 0,
    because the metrics are estimates from finite samples. A model scoring at or
    near this value is as close as the measurement can detect; a model scoring
    far above it is measurably different.

    The halves are split by a seeded shuffle. Splitting chronologically would
    separate the halves by any drift over time, and taking alternate items would
    separate them perfectly whenever the input has an even-period repeating
    structure - which is exactly the case where the floor must stay low. The
    shuffle is seeded so the floor is reproducible.
    """
    if len(replies) < 4:
        return 0.0
    shuffled = list(replies)
    random.Random(seed).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    first, second = shuffled[:midpoint], shuffled[midpoint:]
    result = compare(style_profile(first), style_profile(second))
    return result["style_distance"] or 0.0


def render_comparison(result: Mapping, title: str = "style comparison") -> str:
    """Human-readable table. Contains no message text."""
    lines = [
        "",
        title,
        "-" * len(title),
        f"  reference: {result.get('reference_n', 0):,} replies   "
        f"candidate: {result.get('candidate_n', 0):,} replies",
        "",
        f"  {'metric':<22}{'you':>10}{'model':>10}{'diff':>10}",
    ]
    for metric, values in result.get("metrics", {}).items():
        lines.append(
            f"  {metric:<22}{values['reference']:>10}{values['candidate']:>10}"
            f"{values['difference']:>10}"
        )
    lines += ["", f"  style distance: {result.get('style_distance')}  (lower is closer)", ""]
    return "\n".join(lines)
