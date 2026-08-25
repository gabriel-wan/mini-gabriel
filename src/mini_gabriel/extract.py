"""Telegram extraction via Telethon.

This is the only module that talks to Telegram. It fetches messages inside the
target date window and writes them verbatim to JSONL; it applies no selection
thresholds of its own beyond the window and the dialog-eligibility rules in
``selection.py``. Deciding which chats are usable is the analysis stage's job.

Operational properties:

* Resumable. Progress is recorded per chat in the manifest and re-derived from
  the JSONL on disk, so an interrupted run continues where it stopped.
* Rate-limit tolerant. ``FloodWaitError`` pauses and retries that chat rather
  than aborting the run.
* Quiet. Chat names and counts are logged; message text never is.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from telethon import TelegramClient
from telethon.errors import (
    ApiIdInvalidError,
    FloodWaitError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
)
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.messages import GetFullChatRequest
from telethon.tl.types import Channel, Chat, User

from . import config
from .selection import (
    BROADCAST,
    GROUP,
    GROUP_CHAT_TYPES,
    PRIVATE,
    SUPERGROUP,
    DialogDescriptor,
    evaluate_dialog,
    year_window_utc,
)
from .storage import (
    append_records,
    chat_jsonl_path,
    count_records,
    empty_manifest,
    last_message_id,
    load_manifest,
    manifest_chat_entry,
    save_manifest,
)

logger = logging.getLogger("mini_gabriel.extract")

# Records are flushed to disk in batches so an interrupted run loses very little.
FLUSH_EVERY = 200


def classify_entity(entity) -> tuple[str, bool, bool]:
    """Map a Telethon entity to (chat_type, is_bot, is_self_chat).

    Note that both broadcast channels and supergroups are ``Channel`` objects;
    the ``broadcast`` flag is what separates them. Testing ``isinstance(entity,
    Channel)`` alone would wrongly discard the supergroups we want to keep.
    """
    if isinstance(entity, User):
        return PRIVATE, bool(getattr(entity, "bot", False)), bool(getattr(entity, "is_self", False))
    if isinstance(entity, Chat):
        return GROUP, False, False
    if isinstance(entity, Channel):
        if getattr(entity, "broadcast", False):
            return BROADCAST, False, False
        return SUPERGROUP, False, False
    return type(entity).__name__.lower(), False, False


def is_migrated_group(entity) -> bool:
    """True for a legacy group that has been upgraded to a supergroup.

    Telegram keeps the old ``Chat`` in the dialog list as a deactivated stub
    once its messages have moved to the new supergroup. Fetching it duplicates
    the supergroup's work and yields a meaningless zero participant count.
    """
    if not isinstance(entity, Chat):
        return False
    return getattr(entity, "migrated_to", None) is not None or bool(
        getattr(entity, "deactivated", False)
    )


async def resolve_participant_count(client, entity, chat_type: str) -> tuple[Optional[int], bool]:
    """Best-effort participant count.

    Returns ``(count, known)``. When the count cannot be established the caller
    records that fact; it is never assumed to be small enough to qualify.
    """
    if chat_type == PRIVATE:
        return 2, True

    try:
        if isinstance(entity, Channel):
            full = await client(GetFullChannelRequest(channel=entity))
            count = getattr(full.full_chat, "participants_count", None)
        elif isinstance(entity, Chat):
            count = getattr(entity, "participants_count", None)
            if count is None:
                full = await client(GetFullChatRequest(chat_id=entity.id))
                count = getattr(full.full_chat, "participants_count", None)
        else:
            return None, False
    except FloodWaitError:
        raise
    except Exception as exc:  # private/restricted chats, missing rights, etc.
        logger.debug("participant count unavailable: %s", type(exc).__name__)
        return None, False

    # A zero count is never genuine for a chat that appears in the dialog list;
    # it means Telegram did not report one. Treat it as unknown so it cannot
    # slip under the participant limit.
    if isinstance(count, int) and count > 0:
        return count, True
    return None, False


def message_to_record(message, chat_id: int, local_tz: ZoneInfo) -> dict:
    """Convert a Telethon message into a plain, JSON-serialisable record.

    ``raw_text`` is used rather than ``text`` so the message is stored exactly
    as received, without markdown re-rendering of entities.
    """
    stamp_utc = message.date.astimezone(timezone.utc)
    media = getattr(message, "media", None)
    action = getattr(message, "action", None)

    return {
        "chat_id": chat_id,
        "message_id": message.id,
        "date_utc": stamp_utc.isoformat(),
        "date_local": stamp_utc.astimezone(local_tz).isoformat(),
        "sender_id": getattr(message, "sender_id", None),
        "is_outgoing": bool(getattr(message, "out", False)),
        "text": message.raw_text or "",
        "reply_to_msg_id": getattr(message, "reply_to_msg_id", None),
        "is_forwarded": getattr(message, "forward", None) is not None,
        "has_media": media is not None,
        "media_type": type(media).__name__ if media is not None else None,
        "is_service": action is not None,
    }


async def extract_chat(
    client,
    entity,
    descriptor: DialogDescriptor,
    window_start: datetime,
    window_end: datetime,
    chats_dir: Path,
    local_tz: ZoneInfo,
) -> tuple[int, int, bool]:
    """Extract one chat's in-window messages. Returns (last_id, new, complete)."""
    path = chat_jsonl_path(chats_dir, descriptor.chat_id)
    last_id = last_message_id(path)
    new_records = 0
    complete = False

    while True:
        buffer: list[dict] = []
        try:
            # min_id resumes an interrupted chat; offset_date positions a fresh
            # one at the start of the window. Iteration is oldest-first so the
            # highest id seen is always a safe resume point.
            kwargs = {"reverse": True}
            if last_id:
                kwargs["min_id"] = last_id
            else:
                kwargs["offset_date"] = window_start

            async for message in client.iter_messages(entity, **kwargs):
                if message.date >= window_end:
                    complete = True
                    break
                if message.date < window_start:
                    last_id = max(last_id, message.id)
                    continue

                buffer.append(message_to_record(message, descriptor.chat_id, local_tz))
                last_id = max(last_id, message.id)

                if len(buffer) >= FLUSH_EVERY:
                    new_records += append_records(path, buffer)
                    buffer = []
            else:
                # Iterator exhausted: the chat has no more messages at all.
                complete = True

            if buffer:
                new_records += append_records(path, buffer)
            return last_id, new_records, complete

        except FloodWaitError as exc:
            if buffer:
                new_records += append_records(path, buffer)
            wait_for = int(getattr(exc, "seconds", 60)) + 5
            logger.warning(
                "rate limited on %s; waiting %ss then resuming", descriptor.name, wait_for
            )
            await asyncio.sleep(wait_for)


async def run_extraction(
    year: int = config.TARGET_YEAR,
    tz_name: str = config.TIMEZONE,
    limit: Optional[int] = None,
    dry_run: bool = False,
    max_members_to_fetch: Optional[int] = None,
) -> dict:
    """Run the extraction stage. Returns the manifest."""
    credentials = config.load_credentials()
    config.ensure_data_dirs()

    window_start, window_end = year_window_utc(year, tz_name)
    local_tz = ZoneInfo(tz_name)

    logger.info(
        "window: %s .. %s (%s %s)",
        window_start.isoformat(),
        window_end.isoformat(),
        year,
        tz_name,
    )

    manifest = load_manifest(config.MANIFEST_PATH) or empty_manifest(
        year, tz_name, window_start.isoformat(), window_end.isoformat()
    )
    manifest.setdefault("chats", {})
    manifest.setdefault("skipped", {})

    client = TelegramClient(str(credentials.session_path), credentials.api_id, credentials.api_hash)

    try:
        # start() runs Telethon's interactive login on first use: phone number,
        # the code Telegram sends, and a 2FA password if the account has one.
        # Login failures are ordinary user-input mistakes, so they are turned
        # into actionable messages rather than tracebacks.
        logger.info(
            "login: enter your phone in international format, country code first "
            "and no spaces or dashes (a Singapore number looks like +6591234567)"
        )
        try:
            await client.start()
        except PhoneNumberInvalidError as exc:
            raise RuntimeError(
                "Telegram rejected that phone number. It needs international "
                "format: country code first, no spaces or dashes. A Singapore "
                "mobile looks like +6591234567."
            ) from exc
        except (PhoneCodeInvalidError, PhoneCodeExpiredError) as exc:
            raise RuntimeError(
                "That login code was wrong or had already expired. Re-run the "
                "command to request a fresh one."
            ) from exc
        except ApiIdInvalidError as exc:
            raise RuntimeError(
                "TELEGRAM_API_ID and TELEGRAM_API_HASH are not a valid pair. "
                "Check them against my.telegram.org (API development tools)."
            ) from exc

        me = await client.get_me()
        logger.info("authenticated as user id %s", getattr(me, "id", "unknown"))

        processed = 0
        async for dialog in client.iter_dialogs():
            if limit is not None and processed >= limit:
                logger.info("reached --limit of %s dialogs", limit)
                break

            entity = dialog.entity
            chat_type, is_bot, is_self_chat = classify_entity(entity)
            descriptor = DialogDescriptor(
                chat_id=dialog.id,
                name=dialog.name or "",
                chat_type=chat_type,
                is_bot=is_bot,
                is_self_chat=is_self_chat,
                is_migrated=is_migrated_group(entity),
            )

            decision = evaluate_dialog(descriptor)
            if not decision.included:
                manifest["skipped"][str(descriptor.chat_id)] = {
                    "chat_id": descriptor.chat_id,
                    "name": descriptor.name,
                    "chat_type": descriptor.chat_type,
                    "reason": decision.reason_text,
                }
                continue

            try:
                count, known = await resolve_participant_count(client, entity, chat_type)
            except FloodWaitError as exc:
                wait_for = int(getattr(exc, "seconds", 60)) + 5
                logger.warning("rate limited resolving members; waiting %ss", wait_for)
                await asyncio.sleep(wait_for)
                count, known = await resolve_participant_count(client, entity, chat_type)

            descriptor = replace(descriptor, participant_count=count, participant_count_known=known)
            if not known:
                logger.info("participant count unavailable for %s", descriptor.name)

            # Re-evaluate now that the participant count is known: it rules out
            # groups that contain only me, which the first pass could not see.
            decision = evaluate_dialog(descriptor)
            if not decision.included:
                manifest["skipped"][str(descriptor.chat_id)] = {
                    "chat_id": descriptor.chat_id,
                    "name": descriptor.name,
                    "chat_type": descriptor.chat_type,
                    "participant_count": count,
                    "reason": decision.reason_text,
                }
                continue

            # Optional fetch-scope limit, off by default. The selection criteria
            # still live entirely in the analysis stage; this only avoids paying
            # to download chats that the participant rule will certainly reject.
            # The message-count rule cannot be short-circuited this way, because
            # it is unknowable before fetching.
            if (
                max_members_to_fetch is not None
                and descriptor.chat_type in GROUP_CHAT_TYPES
                and known
                and count is not None
                and count > max_members_to_fetch
            ):
                manifest["skipped"][str(descriptor.chat_id)] = {
                    "chat_id": descriptor.chat_id,
                    "name": descriptor.name,
                    "chat_type": descriptor.chat_type,
                    "participant_count": count,
                    "reason": (
                        f"not fetched: {count} members exceeds the "
                        f"--max-members-to-fetch limit of {max_members_to_fetch}"
                    ),
                }
                logger.info("skipping %s (%s members)", descriptor.name, count)
                continue

            processed += 1

            if dry_run:
                manifest["chats"].setdefault(
                    str(descriptor.chat_id),
                    manifest_chat_entry(descriptor.__dict__, 0, 0, False, None),
                )
                logger.info(
                    "eligible: %s (%s, members=%s)",
                    descriptor.name,
                    descriptor.chat_type,
                    count if known else "unknown",
                )
                continue

            existing = manifest["chats"].get(str(descriptor.chat_id), {})
            if existing.get("extraction_complete"):
                # Refresh metadata but do not refetch messages.
                manifest["chats"][str(descriptor.chat_id)] = manifest_chat_entry(
                    descriptor.__dict__,
                    existing.get("last_message_id", 0),
                    existing.get("message_count", 0),
                    True,
                    None,
                )
                logger.info("already complete, skipping: %s", descriptor.name)
                continue

            logger.info("extracting %s (%s)", descriptor.name, descriptor.chat_type)
            error: Optional[str] = None
            complete = False
            last_id = existing.get("last_message_id", 0)

            try:
                last_id, new_records, complete = await extract_chat(
                    client, entity, descriptor, window_start, window_end, config.CHATS_DIR, local_tz
                )
                logger.info("  %s new messages", new_records)
            except Exception as exc:  # noqa: BLE001 - one bad chat must not kill the run
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("  failed: %s", error)

            stored = count_records(chat_jsonl_path(config.CHATS_DIR, descriptor.chat_id))
            manifest["chats"][str(descriptor.chat_id)] = manifest_chat_entry(
                descriptor.__dict__, last_id, stored, complete, error
            )
            save_manifest(config.MANIFEST_PATH, manifest)

    finally:
        await client.disconnect()

    save_manifest(config.MANIFEST_PATH, manifest)
    logger.info(
        "done: %s dialogs %s, %s skipped",
        len(manifest["chats"]),
        "eligible (dry run - nothing fetched)" if dry_run else "extracted",
        len(manifest["skipped"]),
    )
    return manifest
