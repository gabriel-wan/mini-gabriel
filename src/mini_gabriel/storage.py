"""Raw-data storage: JSONL message files and the extraction manifest.

Filesystem I/O only, no Telethon and no network. Raw messages are appended to
one JSONL file per chat so that extraction can be interrupted and resumed
without rewriting or discarding anything already fetched.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional

SCHEMA_VERSION = 1


def chat_jsonl_path(chats_dir: Path, chat_id: int) -> Path:
    """Path of the raw message file for one chat.

    Negative ids (groups and channels) are encoded with a 'n' prefix so the
    filename stays portable across filesystems.
    """
    stem = f"n{abs(chat_id)}" if chat_id < 0 else str(chat_id)
    return chats_dir / f"{stem}.jsonl"


def append_records(path: Path, records: Iterable[Mapping]) -> int:
    """Append records as JSONL. Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def iter_records(path: Path) -> Iterator[dict]:
    """Stream records from a JSONL file, skipping a truncated final line.

    A partially written last line can occur if extraction was killed mid-write;
    it is skipped rather than treated as corruption of the whole file.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def last_message_id(path: Path) -> int:
    """Highest message id already stored for a chat; 0 if none.

    This is the resume point: extraction restarts from just after it.
    """
    highest = 0
    for record in iter_records(path):
        message_id = record.get("message_id")
        if isinstance(message_id, int) and message_id > highest:
            highest = message_id
    return highest


def empty_manifest(target_year: int, tz_name: str, window_start: str, window_end: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "target_year": target_year,
        "timezone": tz_name,
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "chats": {},
        "skipped": {},
    }


def load_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def save_manifest(path: Path, manifest: Mapping) -> None:
    """Write the manifest atomically so an interrupted run cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=".manifest-", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def manifest_chat_entry(
    dialog_dict: Mapping,
    last_id: int,
    message_count: int,
    complete: bool,
    error: Optional[str] = None,
) -> dict:
    entry = dict(dialog_dict)
    entry.update(
        {
            "last_message_id": last_id,
            "message_count": message_count,
            "extraction_complete": complete,
            "error": error,
        }
    )
    return entry


def count_records(path: Path) -> int:
    """Number of well-formed records currently stored for a chat."""
    return sum(1 for _ in iter_records(path))
