"""Pure selection logic: date windows, dialog eligibility, chat qualification.

Nothing in this module imports Telethon or touches the network. Everything
operates on plain dataclasses and dictionaries, so the whole decision surface
is testable with fictional data.

Two distinct stages are represented here, and they must not be conflated:

* ``evaluate_dialog`` decides whether a dialog is *extracted at all*. It only
  excludes things that can never be useful (broadcast channels, bots, Saved
  Messages). It never applies size or volume thresholds.
* ``evaluate_chat`` decides whether an *already extracted* chat qualifies as
  training data. This is where the participant and message thresholds live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

# Chat types. "broadcast" is a Telegram channel; "supergroup" is a megagroup.
# Both are Channel objects in Telethon, which is why the distinction is made
# explicitly here rather than by object type.
PRIVATE = "private"
GROUP = "group"
SUPERGROUP = "supergroup"
BROADCAST = "broadcast"

EXTRACTABLE_CHAT_TYPES = frozenset({PRIVATE, GROUP, SUPERGROUP})
GROUP_CHAT_TYPES = frozenset({GROUP, SUPERGROUP})


# --------------------------------------------------------------------------
# Date window
# --------------------------------------------------------------------------


def year_window_utc(year: int, tz_name: str) -> tuple[datetime, datetime]:
    """Return the UTC half-open interval [start, end) for a calendar year.

    The year boundaries are interpreted in ``tz_name``, not in UTC. For
    Asia/Singapore (UTC+8, no DST) the 2026 window therefore starts at
    2025-12-31T16:00:00Z and ends at 2026-12-31T16:00:00Z.
    """
    tz = ZoneInfo(tz_name)
    start_local = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)
    end_local = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def is_in_window(moment: datetime, start: datetime, end: datetime) -> bool:
    """True if ``moment`` falls in the half-open interval [start, end)."""
    if moment.tzinfo is None:
        raise ValueError("naive datetime; extraction records are always timezone-aware")
    return start <= moment < end


def has_text(message: Mapping) -> bool:
    """True if the message carries non-whitespace text.

    Media-only messages, stickers and service messages are retained in the raw
    extraction but must not count toward the training-data threshold.
    """
    text = message.get("text") or ""
    return bool(text.strip())


# --------------------------------------------------------------------------
# Stage 1: which dialogs get extracted
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogDescriptor:
    """Platform-neutral description of a Telegram dialog."""

    chat_id: int
    name: str
    chat_type: str
    is_bot: bool = False
    is_self_chat: bool = False
    is_migrated: bool = False
    participant_count: Optional[int] = None
    participant_count_known: bool = True


@dataclass(frozen=True)
class Decision:
    """Outcome of a filtering step, with the reasons it was excluded."""

    included: bool
    reasons: tuple[str, ...] = ()

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons) if self.reasons else ""


def evaluate_dialog(dialog: DialogDescriptor) -> Decision:
    """Decide whether a dialog should be extracted at all.

    Size and volume thresholds are deliberately *not* applied here: they cannot
    be known before extraction, and applying them here would collapse the two
    stages into one.
    """
    reasons: list[str] = []

    if dialog.chat_type == BROADCAST:
        reasons.append("broadcast channel")
    elif dialog.chat_type not in EXTRACTABLE_CHAT_TYPES:
        reasons.append("unsupported chat type: " + str(dialog.chat_type))

    if dialog.is_bot:
        reasons.append("bot conversation")

    if dialog.is_self_chat:
        reasons.append("saved messages")

    if dialog.is_migrated:
        # When a legacy group is upgraded to a supergroup, Telegram leaves a
        # deactivated stub behind in the dialog list. Its messages now live in
        # the supergroup, so fetching the stub duplicates work and produces a
        # bogus zero participant count.
        reasons.append("legacy group migrated to a supergroup")

    # A group containing only me is a private notepad, not a conversation, and
    # carries no conversational style. The count is not always known when this
    # runs, so callers re-evaluate once it has been resolved.
    if (
        dialog.chat_type in GROUP_CHAT_TYPES
        and dialog.participant_count_known
        and dialog.participant_count is not None
        and 0 < dialog.participant_count <= 1
    ):
        reasons.append("group has no other participants")

    return Decision(included=not reasons, reasons=tuple(reasons))


# --------------------------------------------------------------------------
# Stage 2: which extracted chats qualify as training data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatStats:
    """Aggregated counts for one extracted chat. Contains no message text."""

    chat_id: int
    name: str
    chat_type: str
    participant_count: Optional[int]
    participant_count_known: bool
    total_messages: int
    my_messages: int
    my_text_messages: int
    other_messages: int
    other_text_messages: int
    first_message_utc: Optional[str]
    last_message_utc: Optional[str]
    extraction_complete: bool = True

    def to_dict(self) -> dict:
        return {
            "chat_id": self.chat_id,
            "name": self.name,
            "chat_type": self.chat_type,
            "participant_count": self.participant_count,
            "participant_count_known": self.participant_count_known,
            "total_messages": self.total_messages,
            "my_messages": self.my_messages,
            "my_text_messages": self.my_text_messages,
            "other_messages": self.other_messages,
            "other_text_messages": self.other_text_messages,
            "first_message_utc": self.first_message_utc,
            "last_message_utc": self.last_message_utc,
            "extraction_complete": self.extraction_complete,
        }


@dataclass(frozen=True)
class SelectionCriteria:
    """Chat-selection thresholds. Initial values, expected to be revised."""

    max_participants: int = 20
    min_my_text_messages: int = 100


def aggregate_chat_stats(
    dialog: DialogDescriptor,
    messages: Iterable[Mapping],
    extraction_complete: bool = True,
) -> ChatStats:
    """Fold a stream of raw message records into per-chat counts.

    Accepts any iterable of mappings, so it can be fed from a JSONL file or
    from a fictional list in the tests.
    """
    total = my_total = my_text = other_total = other_text = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    for message in messages:
        total += 1
        textual = has_text(message)

        if message.get("is_outgoing"):
            my_total += 1
            if textual:
                my_text += 1
        else:
            other_total += 1
            if textual:
                other_text += 1

        stamp = message.get("date_utc")
        if stamp:
            if first_seen is None or stamp < first_seen:
                first_seen = stamp
            if last_seen is None or stamp > last_seen:
                last_seen = stamp

    return ChatStats(
        chat_id=dialog.chat_id,
        name=dialog.name,
        chat_type=dialog.chat_type,
        participant_count=dialog.participant_count,
        participant_count_known=dialog.participant_count_known,
        total_messages=total,
        my_messages=my_total,
        my_text_messages=my_text,
        other_messages=other_total,
        other_text_messages=other_text,
        first_message_utc=first_seen,
        last_message_utc=last_seen,
        extraction_complete=extraction_complete,
    )


def evaluate_chat(stats: ChatStats, criteria: Optional[SelectionCriteria] = None) -> Decision:
    """Decide whether an extracted chat qualifies as training data."""
    criteria = criteria or SelectionCriteria()
    reasons: list[str] = []

    if stats.chat_type == BROADCAST:
        reasons.append("broadcast channel")

    if stats.chat_type in GROUP_CHAT_TYPES:
        # An unknown participant count is never treated as passing: the whole
        # point of the rule is to keep large groups out. A reported count of
        # zero is not a real count either - it appears on migrated or emptied
        # legacy groups - so it counts as undetermined rather than as
        # comfortably below the limit.
        if (
            not stats.participant_count_known
            or stats.participant_count is None
            or stats.participant_count <= 0
        ):
            reasons.append("participant count could not be determined")
        elif stats.participant_count <= 1:
            reasons.append("group has no other participants")
        elif stats.participant_count > criteria.max_participants:
            reasons.append(
                "{} participants exceeds limit of {}".format(
                    stats.participant_count, criteria.max_participants
                )
            )

    if stats.my_text_messages < criteria.min_my_text_messages:
        reasons.append(
            "only {} text messages authored by me (need {})".format(
                stats.my_text_messages, criteria.min_my_text_messages
            )
        )

    if not stats.extraction_complete:
        reasons.append("extraction incomplete; counts are partial")

    return Decision(included=not reasons, reasons=tuple(reasons))


def descriptor_from_manifest_entry(entry: Mapping) -> DialogDescriptor:
    """Rebuild a DialogDescriptor from a manifest record."""
    return DialogDescriptor(
        chat_id=entry["chat_id"],
        name=entry.get("name", ""),
        chat_type=entry.get("chat_type", PRIVATE),
        is_bot=entry.get("is_bot", False),
        is_self_chat=entry.get("is_self_chat", False),
        is_migrated=entry.get("is_migrated", False),
        participant_count=entry.get("participant_count"),
        participant_count_known=entry.get("participant_count_known", True),
    )
