"""Turn and session segmentation.

Pure functions over the plain message records written by the extraction stage.
No I/O, no Telethon, no network, so the whole thing is testable with fictional
data.

Two levels of structure are built here:

* A **turn** is a run of consecutive messages from the same author, sent close
  together in time. This is the unit the model is asked to produce, because
  writing several short messages in a row is the dominant pattern in the data
  (54% of turns contain more than one message).
* A **session** is a stretch of conversation separated from its neighbours by a
  long silence. Sessions exist so that an example never draws context from an
  unrelated exchange days earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Optional, Sequence

DEFAULT_BURST_GAP_SECONDS = 300  # 5 minutes
DEFAULT_SESSION_GAP_SECONDS = 3 * 60 * 60  # 3 hours

# A message this short carries no style signal on its own: "ok", "lol", "yeah".
TRIVIAL_MAX_CHARS = 5


@dataclass(frozen=True)
class Turn:
    """One author's uninterrupted run of messages."""

    sender_id: Optional[int]
    is_me: bool
    messages: tuple[str, ...]
    start_utc: str
    end_utc: str

    @property
    def text(self) -> str:
        """The turn as a single string, message boundaries kept as newlines."""
        return "\n".join(self.messages)

    @property
    def is_trivial(self) -> bool:
        """True when every message in the turn is a bare acknowledgement.

        A turn that pairs "ok" with something substantive is not trivial; only
        turns that are nothing but acknowledgements count.
        """
        return all(len(message.strip()) <= TRIVIAL_MAX_CHARS for message in self.messages)


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def has_usable_text(record: Mapping) -> bool:
    """True if the record carries text worth learning from."""
    return bool((record.get("text") or "").strip())


def build_turns(
    records: Iterable[Mapping],
    burst_gap_seconds: int = DEFAULT_BURST_GAP_SECONDS,
) -> list[Turn]:
    """Group messages into turns, oldest first.

    Messages without text are skipped entirely rather than treated as turn
    boundaries. A photo sent in the middle of a burst is part of the same
    stretch of typing, so splitting the burst around it would invent a turn
    boundary that did not exist.

    Records are sorted by message id, so callers need not pre-sort.
    """
    ordered = sorted(
        (record for record in records if has_usable_text(record)),
        key=lambda record: record.get("message_id", 0),
    )

    turns: list[Turn] = []
    buffer: list[str] = []
    identity: Optional[tuple] = None
    started: Optional[str] = None
    last_stamp: Optional[str] = None

    def flush() -> None:
        if buffer and identity is not None and started is not None and last_stamp is not None:
            turns.append(
                Turn(
                    sender_id=identity[1],
                    is_me=identity[0],
                    messages=tuple(buffer),
                    start_utc=started,
                    end_utc=last_stamp,
                )
            )

    for record in ordered:
        text = (record["text"] or "").strip()
        who = (bool(record.get("is_outgoing")), record.get("sender_id"))
        stamp = record["date_utc"]

        continues = (
            identity == who
            and last_stamp is not None
            and (_parse(stamp) - _parse(last_stamp)).total_seconds() <= burst_gap_seconds
        )

        if continues:
            buffer.append(text)
        else:
            flush()
            buffer = [text]
            identity = who
            started = stamp

        last_stamp = stamp

    flush()
    return turns


def split_sessions(
    turns: Sequence[Turn],
    session_gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS,
) -> list[list[Turn]]:
    """Cut a chat's turns into conversation sessions on long silences."""
    if not turns:
        return []

    sessions: list[list[Turn]] = [[turns[0]]]
    for previous, current in zip(turns, turns[1:]):
        silence = (_parse(current.start_utc) - _parse(previous.end_utc)).total_seconds()
        if silence > session_gap_seconds:
            sessions.append([current])
        else:
            sessions[-1].append(current)
    return sessions
