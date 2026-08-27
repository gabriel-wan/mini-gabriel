"""Convert training examples into chat messages.

Emits the standard ``[{"role": ..., "content": ...}]`` structure that every
training framework accepts, rather than a rendered chat template.

That distinction is deliberate. Templates are model-specific and change between
releases, so hardcoding Qwen's special tokens here would silently rot. The
tokenizer's own ``apply_chat_template`` renders these messages correctly for
whichever model is loaded, and stays correct when the model changes.

Pure functions: no tokenizer, no model, no I/O.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

ME = "me"
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"

DEFAULT_SYSTEM_PROMPT = (
    "You are replying as yourself in a casual Telegram chat. "
    "Write the way you normally write."
)


def _speaker_prefixed(speaker: str, text: str, prefix: bool) -> str:
    """Label who is talking, when more than one other person is present."""
    return f"{speaker}: {text}" if prefix else text


def needs_speaker_prefix(context: Sequence[Mapping]) -> bool:
    """True when the conversation has more than one other participant.

    In a one-to-one chat the labels add noise; in a group they are the only way
    to tell who said what once everyone collapses into the user role.
    """
    others = {turn["speaker"] for turn in context if turn["speaker"] != ME}
    return len(others) > 1


def to_messages(
    example: Mapping,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    force_speaker_prefix: Optional[bool] = None,
) -> Optional[list[dict]]:
    """Convert one example into chat messages.

    Returns None for examples that cannot form a valid exchange - specifically
    those whose context contains nothing from anyone else, leaving nothing to
    reply to.

    Consecutive turns from the same role are merged, because chat templates
    expect user and assistant to alternate and several break outright when they
    do not.
    """
    context = list(example.get("context") or [])
    target = list(example.get("target") or [])

    if not context or not target:
        return None
    if all(turn["speaker"] == ME for turn in context):
        # Only my own earlier messages: no incoming message to respond to.
        return None

    prefix = (
        needs_speaker_prefix(context) if force_speaker_prefix is None else force_speaker_prefix
    )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": SYSTEM, "content": system_prompt})

    for turn in context:
        speaker = turn["speaker"]
        role = ASSISTANT if speaker == ME else USER
        content = turn["text"] if speaker == ME else _speaker_prefixed(speaker, turn["text"], prefix)

        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})

    reply = "\n".join(target)

    # The target must be its own final assistant message, never merged into a
    # preceding one, or the model would be trained to predict its own context.
    if messages and messages[-1]["role"] == ASSISTANT:
        return None

    messages.append({"role": ASSISTANT, "content": reply})
    return messages


def convert_all(
    examples: Sequence[Mapping],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> tuple[list[dict], dict]:
    """Convert a dataset. Returns (rows, counters)."""
    rows: list[dict] = []
    counters = {"converted": 0, "dropped_no_other_speaker": 0, "dropped_trailing_self": 0}

    for example in examples:
        context = list(example.get("context") or [])
        messages = to_messages(example, system_prompt)
        if messages is None:
            if context and all(turn["speaker"] == ME for turn in context):
                counters["dropped_no_other_speaker"] += 1
            else:
                counters["dropped_trailing_self"] += 1
            continue
        rows.append({"messages": messages})
        counters["converted"] += 1

    return rows, counters
