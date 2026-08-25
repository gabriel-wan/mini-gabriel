"""Training-example construction.

Pure functions: turns in, examples out. No I/O and no randomness that would
make two runs disagree.

An example is one turn I authored, paired with the conversation that preceded
it. Every turn I authored becomes the target of exactly one example, so all of
the data is used; capping context limits how much history each example carries,
not how much data is kept.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from .turns import Turn

ME = "me"

# No base model has been chosen, so no tokenizer is available. Context is
# budgeted in characters using this ratio and converted for reporting. Replace
# with a real tokenizer once a model is selected.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class BuildConfig:
    """Every knob in dataset construction. See docs/DATA.md for the rationale."""

    burst_gap_seconds: int = 300
    session_gap_seconds: int = 3 * 60 * 60
    context_turns: int = 10
    context_token_budget: int = 1024
    trivial_keep_rate: float = 0.33
    seed: str = "mini-gabriel"

    @property
    def context_char_budget(self) -> int:
        return self.context_token_budget * CHARS_PER_TOKEN


# --------------------------------------------------------------------------
# Pseudonymisation
# --------------------------------------------------------------------------


def _label_for(index: int) -> str:
    """A, B, ... Z, AA, AB, ... for however many participants a chat has."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def assign_pseudonyms(turns: Sequence[Turn]) -> dict[Optional[int], str]:
    """Map each participant to a stable placeholder.

    I am always ``me``. Everyone else is lettered in order of first appearance,
    so the mapping is deterministic for a given chat and carries no real
    identity into the training data.
    """
    mapping: dict[Optional[int], str] = {}
    others = 0
    for turn in turns:
        if turn.is_me:
            mapping.setdefault(turn.sender_id, ME)
            continue
        if turn.sender_id not in mapping:
            mapping[turn.sender_id] = _label_for(others)
            others += 1
    return mapping


# --------------------------------------------------------------------------
# Deterministic downsampling
# --------------------------------------------------------------------------


def keep_trivial_turn(chat_id: int, target_key: str, keep_rate: float, seed: str) -> bool:
    """Decide whether to keep an all-acknowledgement turn.

    Uses a stable hash rather than a random number generator so that rebuilding
    the dataset produces exactly the same examples. A dataset you cannot
    reproduce is one you cannot attribute a training result to.
    """
    if keep_rate >= 1:
        return True
    if keep_rate <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{chat_id}:{target_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < keep_rate


# --------------------------------------------------------------------------
# Context windowing
# --------------------------------------------------------------------------


def take_context(
    history: Sequence[Turn],
    pseudonyms: Mapping[Optional[int], str],
    config: BuildConfig,
) -> list[dict]:
    """Take the most recent turns that fit within both budgets.

    Walks backwards from the target so the turns nearest the reply are kept,
    then restores chronological order.
    """
    picked: list[Turn] = []
    used = 0
    for turn in reversed(history[-config.context_turns :] if config.context_turns else []):
        cost = len(turn.text)
        if picked and used + cost > config.context_char_budget:
            break
        picked.append(turn)
        used += cost
    picked.reverse()
    return [{"speaker": pseudonyms.get(t.sender_id, "?"), "text": t.text} for t in picked]


# --------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------


def build_examples(
    chat_id: int,
    chat_type: str,
    sessions: Iterable[Sequence[Turn]],
    pseudonyms: Mapping[Optional[int], str],
    config: Optional[BuildConfig] = None,
) -> tuple[list[dict], dict]:
    """Turn sessions into examples. Returns (examples, counters)."""
    config = config or BuildConfig()
    examples: list[dict] = []
    counters = {"turns_mine": 0, "trivial_seen": 0, "trivial_kept": 0, "no_context": 0}

    for session in sessions:
        for position, turn in enumerate(session):
            if not turn.is_me:
                continue
            counters["turns_mine"] += 1

            if turn.is_trivial:
                counters["trivial_seen"] += 1
                target_key = f"{turn.start_utc}:{position}"
                if not keep_trivial_turn(
                    chat_id, target_key, config.trivial_keep_rate, config.seed
                ):
                    continue
                counters["trivial_kept"] += 1

            context = take_context(session[:position], pseudonyms, config)
            if not context:
                # Nothing preceding it in this session: there is no conversation
                # to respond to, so there is nothing to learn from.
                counters["no_context"] += 1
                continue

            examples.append(
                {
                    "chat_id": chat_id,
                    "chat_type": chat_type,
                    "date_utc": turn.start_utc,
                    "context": context,
                    "target": list(turn.messages),
                    "target_is_trivial": turn.is_trivial,
                }
            )

    return examples, counters


# --------------------------------------------------------------------------
# Held-out chats
# --------------------------------------------------------------------------


def select_holdout_chats(
    ranked_chat_ids: Sequence[int],
    count: int,
    skip_largest: int = 3,
) -> list[int]:
    """Choose whole chats to hold out, spread across the size distribution.

    Whole chats rather than random examples, because neighbouring examples share
    overlapping context: a random split would leak the validation set into
    training. The largest few chats are skipped so that holding out does not
    remove a large share of the training data.
    """
    if count <= 0 or not ranked_chat_ids:
        return []

    pool = list(ranked_chat_ids[skip_largest:]) or list(ranked_chat_ids)
    count = min(count, len(pool))
    # Evenly spaced through the ranking, so the holdout spans busy and quiet chats.
    step = len(pool) / count
    return [pool[min(int(i * step), len(pool) - 1)] for i in range(count)]
