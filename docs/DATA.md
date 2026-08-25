# DATA.md — data strategy and ingestion

This document describes the data strategy and the ingestion implementation as
it actually exists. Extraction and analysis are implemented and validated
end-to-end against real data, and the full extraction is complete: 113,053
messages across 432 chats, of which 58 chats qualify. Dataset construction is
designed but not yet implemented.

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
    ├── chat_analysis.md      the same report, human-readable
    ├── train.jsonl           training examples
    ├── holdout.jsonl         examples from the held-out chats
    ├── pseudonym_map.json    sender id to placeholder, per chat (private)
    └── dataset_summary.json  counts and configuration for the last build
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

## Dataset construction

Implemented in `turns.py` (segmentation), `examples.py` (example construction)
and `build.py` (I/O), and run against the complete extraction: 113,053 messages
across 432 chats, of which 58 qualify.

### What a training example is

The chat history is not handed to the model as one block. It is cut into many
examples, each shaped as:

    conversation so far  ->  what I said next

Every turn I authored becomes the target of one example, so all of the data is
used. A single message is used many times over: once as a target, and again as
context for the examples that follow it.

Worked example. Given:

    1. Friend:  eh you going gym later
    2. Me:      ya
    3. Me:      6pm
    4. Friend:  ok see you
    5. Me:      nice

two examples are produced:

    context: [1]          ->  target: "ya" + "6pm"
    context: [1, 2, 3, 4] ->  target: "nice"

### 1. Bursts are merged, with message boundaries preserved

Consecutive messages from the same author separated by less than `burst_gap`
form a single turn. They are joined with a newline so the boundaries survive
into the target. At generation time the output is split on that delimiter and
sent as separate messages.

Evidence: 54% of my turns contain more than one message, mean 1.93 messages per
turn. 82% of same-author gaps are under 60 seconds, with p90 at 8 minutes.
Treating each message as its own example would train the model to stop after
one, discarding the most characteristic feature of the style.

### 2. Conversations are split on a time gap

A silence longer than `session_gap` begins a new conversation, so context is
never drawn from an unrelated exchange days earlier.

Evidence: only 4% of speaker transitions exceed three hours and 7% exceed one
hour. Conversations are dense, so this boundary rarely fires and is not worth
refining further. Reply chains (`reply_to_msg_id`) are kept in the raw data but
are not used for segmentation yet, since explicit replies are sparse in casual
chat.

### 3. Context is capped, not discarded

Each example carries the preceding `context_turns` turns, subject to
`context_token_budget`. Capping context does not drop data: older messages have
already served as targets in their own examples and as context for nearer ones.

Evidence: the median message is 15 characters, so ten turns is roughly 120
tokens. Context is unusually cheap here and can afford to be generous.

### 4. Other participants are pseudonymised

Each participant is mapped to a stable placeholder within a chat. The mapping is
written to a local, git-ignored file so results stay debuggable.

This preserves multi-speaker structure in group chats while keeping real
identities out of the training data, which matters because models memorise and
reproduce what they are trained on.

Known limitation: this replaces the speaker label only. Names typed inside
message text are not removed. Reliable in-text scrubbing is substantially harder
and carries real false-positive risk, so residual names will remain in the data.

### 5. Validation is held out by whole chats

A small number of entire chats, spanning different registers, are reserved for
evaluation.

Splitting randomly within a chat would leak: neighbouring examples share
overlapping context, so a held-out example would already have been seen during
training. Held-out loss is in any case a weak proxy for style; the meaningful
evaluation is comparing generated replies against the real ones for the same
context.

### 6. Trivial turns are downsampled

Turns consisting only of very short acknowledgements are retained at
`trivial_keep_rate` rather than in full.

Evidence: **11% of my turns are entirely trivial** - every message in them is
five characters or fewer. Trained on all of them, the highest-probability output
becomes an acknowledgement, and the result imitates the style faithfully while
being useless to converse with. Downsampling rather than removing preserves the
ability to be terse when terseness is right.

Note that the per-message figure is 22%, twice the per-turn figure. Merging
bursts absorbs many acknowledgements into turns that also contain something
substantive, so the per-message number materially overstates the problem. The
per-turn figure is the one that matters, because a turn is what the model is
asked to produce.

### Rejected: capping turns per chat

The dataset is concentrated: the largest chat is 18.1% of all training examples
(3,751 turns), the top five are 45.8%, and the top ten are 60%. This looked like
a problem worth fixing with a `max_turns_per_chat` cap, on the theory that the
model would learn one relationship's register rather than a general style.

**The evidence does not support it, so no cap is applied.** Comparing the
largest chat against the aggregate of the other fifty:

| metric | largest chat | other 50 chats |
|---|---:|---:|
| median turn length | 26 chars | 24 chars |
| messages per turn | 1.88 | 1.89 |
| trivial turns | 11.8% | 11.4% |
| ends with punctuation | 9.2% | 9.8% |

My structural style is essentially identical across conversation partners, so
the concentrated chats are not stylistically distinct and there is nothing for a
cap to correct. Capping at 1,500 turns would discard 3,055 real examples - 15% of
the dataset - to address a skew that is not present.

The one dimension that genuinely varies by relationship is emoji rate: 1.9% and
2.5% in the two largest chats, against 9.7% and 15.3% in others. Capping barely
moves this (roughly 5.2% to 5.8% overall), so it is not a fix for it either. The
consequence to be aware of is that the model will learn a single average emoji
rate and apply it uniformly, where I modulate it per person. That is an inherent
limit of relationship-agnostic style imitation; addressing it would mean
conditioning on the chat, not deleting data.

If a trained model turns out to sound like one specific conversation, the cap is
a parameter that can be enabled and the dataset rebuilt in seconds. Deleting
data preemptively, against the evidence, is the wrong order.

### Built dataset

The first build, at the parameter values below:

| | |
|---|---:|
| turns I authored | 20,415 |
| less trivial turns dropped | −1,544 |
| less conversation openers (no context to reply to) | −1,870 |
| **training examples** | **17,001** |
| split | 15,567 train / 1,434 holdout, 54 / 4 chats |

Three things worth noting about those numbers.

The turn count is 20,415 rather than the 20,697 an earlier ad-hoc count gave.
The difference is media: the implementation skips text-free messages instead of
letting them end a turn, so bursts interrupted by a photo now merge into one
turn, as intended.

Turns are not examples. A turn that opens a conversation has nothing preceding
it to respond to, so it cannot become an example; 1,870 turns are dropped for
that reason. This means the dataset teaches the model to *reply*, not to
*initiate*. That is the right trade for a chatbot, but it is a real omission.

Downsampling worked as designed: trivial turns fall from 11% of turns to 4% of
training examples.

### Parameters

Every value above is a parameter rather than a constant, so variants can be
tried without editing code.

| parameter | initial value | controls |
|---|---|---|
| `burst_gap` | 5 minutes | longest silence still counted as one turn |
| `session_gap` | 3 hours | silence that starts a new conversation |
| `context_turns` | 10 | turns of history included per example |
| `context_token_budget` | 1024 | hard cap on context size |
| `trivial_keep_rate` | 0.33 | share of all-trivial turns retained (11% of turns are trivial) |
| `holdout_chats` | 3-4 | chats reserved for evaluation |

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
