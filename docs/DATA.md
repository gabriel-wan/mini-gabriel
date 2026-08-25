# DATA.md — data strategy and ingestion

This document describes the data strategy and the ingestion implementation as
it actually exists. Extraction and analysis are implemented; neither has been
run against real data yet.

## Data source

The initial (and currently only) data source is the author's own Telegram
message history, read through Telethon with the author's own API credentials.

## Two stages, deliberately separate

Ingestion is split into two stages that must not be collapsed into one:

1. **Extraction** (`src/mini_gabriel/extract.py`) fetches messages and writes
   them verbatim to disk. It applies only the date window and the
   dialog-eligibility rules. It discards nothing else.
2. **Analysis / selection** (`src/mini_gabriel/analyze.py`) reads what was
   extracted and decides which chats qualify as training data.

The separation exists because the selection criteria *cannot* be evaluated
before extraction: neither "at least 100 messages authored by me" nor the
participant count is knowable until the data has been fetched. Keeping the
stages apart also means thresholds can be revised and the analysis re-run
without re-downloading anything.

All decision logic lives in `src/mini_gabriel/selection.py`, which imports no
Telethon and touches no network. That is what makes it testable.

## Date window

The target window is the **2026 calendar year in Asia/Singapore**, converted to
UTC for comparison:

- start: `2025-12-31T16:00:00Z` (inclusive)
- end:   `2026-12-31T16:00:00Z` (exclusive)

The timezone matters at the boundaries. A message sent at 00:30 on 1 January
2026 Singapore time occurs on 31 December 2025 in UTC; treating the window as
UTC would wrongly exclude it. The window is the *fetch scope* — it bounds what
extraction requests from Telegram, rather than being a filter applied after the
fact.

Windows note: `zoneinfo` has no IANA database on Windows, so `tzdata` is a
declared dependency there.

## Dialog eligibility (applied during extraction)

Extracted:

- private chats with people
- groups and supergroups, **regardless of size**

Skipped, with the reason recorded in the manifest:

- broadcast channels
- bot conversations
- Saved Messages (the chat with oneself)
- legacy groups that have been migrated to a supergroup
- groups containing only me (a private notepad, not a conversation)

The migrated-group case is not hypothetical: the first dry run against the real
account found 41 such stubs out of 578 eligible dialogs. When a legacy group is
upgraded, Telegram leaves the old `Chat` in the dialog list as a deactivated
shell whose messages now live in the supergroup. Fetching it duplicates work
and reports a participant count of zero.

Telethon represents both broadcast channels and supergroups as `Channel`
objects; they are distinguished by the `broadcast` flag, not by type. Excluding
channels by `isinstance` alone would silently drop the supergroups we want.

Size and message-volume thresholds are **not** applied here.

## Chat-selection criteria (applied during analysis)

A chat qualifies as training data when:

- it is a private chat, group, or supergroup (never a broadcast channel), and
- for groups and supergroups, the participant count is **known** and is
  **at most 20**, and
- it contains **at least 100 messages authored by me that have non-empty text**.

Two deliberate refinements:

- **Media-only, sticker, and service messages are retained in the raw
  extraction but do not count toward the 100.** A chat that is a hundred
  stickers is not a hundred style samples.
- **An unknown participant count never silently qualifies.** When the count
  cannot be determined, the fact is recorded and the chat is excluded, since
  the purpose of the rule is to keep large groups out.
- **A reported count of zero is treated as unknown, not as "small".** Zero is
  never a genuine count for a chat that appears in the dialog list; it means
  Telegram declined to report one. Taking it at face value would let dead
  groups sail under the 20-member limit.

Chats whose extraction did not complete are reported with partial counts and
are not treated as qualifying.

These remain **initial** criteria. They are expected to change once real data
has been inspected; both thresholds are command-line flags on the analysis
stage precisely so they can be varied without re-extracting.

## Storage layout

```
data/
├── mini_gabriel.session      Telethon session (a credential; git-ignored)
├── raw/
│   ├── manifest.json         per-chat extraction state and metadata
│   └── chats/
│       ├── 1001.jsonl        one JSONL file per chat, one message per line
│       └── n2001.jsonl       negative (group) ids are written with an 'n' prefix
└── processed/
    ├── chat_analysis.json    full analysis report
    └── chat_analysis.md      the same report, human-readable
```

### Raw message record

One JSON object per line:

| field | meaning |
|---|---|
| `chat_id` | Telegram chat id |
| `message_id` | Telegram message id (also the resume point) |
| `date_utc` | timestamp in UTC, ISO 8601 |
| `date_local` | the same instant in Asia/Singapore |
| `sender_id` | author's Telegram user id |
| `is_outgoing` | true when authored by me |
| `text` | message text, preserved exactly as received |
| `reply_to_msg_id` | id of the message being replied to, if any |
| `is_forwarded` | whether the message was forwarded |
| `has_media` / `media_type` | media presence and Telethon media class name |
| `is_service` | whether this is a service event (joins, renames, and similar) |

Text is taken from Telethon's `raw_text`, not `text`, so it is stored exactly
as received rather than re-rendered with markdown for entities.

### Manifest

`data/raw/manifest.json` records the window, and for every dialog: its
metadata, participant count and whether that count is known, the highest
message id stored, how many messages were stored, whether extraction completed,
and any error. Skipped dialogs are recorded separately with their reason.

## Operational behaviour

- **Resumable.** Progress is derived from the highest message id already in
  each chat's JSONL, so an interrupted run continues where it stopped and never
  refetches or rewrites existing data. Chats already marked complete are
  skipped entirely.
- **Rate-limit tolerant.** `FloodWaitError` pauses for the requested interval
  and then resumes that chat, rather than aborting the run. Buffered records
  are flushed before sleeping.
- **Fault-isolated.** A chat that fails (private, restricted, missing rights)
  records the error in the manifest and the run continues.
- **Atomic manifest writes.** The manifest is written to a temporary file and
  renamed, so an interrupted run cannot corrupt it.
- **Truncation-tolerant reads.** A partially written final JSONL line is
  skipped rather than invalidating the file.

## Privacy rules

These rules are non-negotiable:

- Raw messages remain **local** to the author's machine.
- Raw messages must **never** be committed to Git.
- Telegram API credentials must **never** be committed or printed. They are
  read from `.env` and never appear in logs or error messages.
- Session files (`*.session`, `*.session-journal`) must **never** be committed.
  The session file is itself a credential: anyone holding it can act as the
  account.
- No Telegram data is uploaded anywhere. Extraction writes to local disk only.
- **Message contents are never printed to the terminal** and never appear in
  the analysis report. The report carries chat names, ids, types, participant
  counts, message counts, date ranges, and qualification status — counts and
  metadata, never text. A test (`tests/test_analyze.py`) asserts this by
  planting a canary string in fictional messages and checking it cannot reach
  any rendered or written report.
- The repository contains code and documentation — **not private
  conversations**.

The `.gitignore` at the repository root covers `.env`, `*.session`,
`*.session-journal`, `data/raw/`, and `data/processed/`. Any new location that
holds private data must be added to `.gitignore` before data is written there.

## Running the stages

```bash
python scripts/extract_telegram.py --dry-run    # list eligible dialogs, fetch nothing
python scripts/extract_telegram.py              # extract; resumable
python scripts/analyze_chats.py                 # report on what qualifies
```

### Optional fetch-scope limit

`--max-members-to-fetch N` skips *fetching* groups larger than N. It is off by
default, so the stages stay separate as designed.

It exists because the participant count is one criterion that genuinely is
knowable before fetching, and the cost asymmetry is severe: the first dry run
found 104 groups above the 20-member limit, the largest with 24,028 members.
Downloading a year of traffic from those only to discard all of it is expensive
in time and rate limits. The message-count criterion cannot be short-circuited
this way, since it is unknowable until the messages exist.

The trade-off is that raising the participant threshold later means re-running
extraction for the chats that were skipped. Extraction is resumable and skips
completed chats, so that re-run only fetches what is genuinely missing.

The first extraction run performs Telethon's interactive login (phone number,
the code Telegram sends, and a 2FA password if the account has one). The
session is then reused, so later runs need no interaction.
