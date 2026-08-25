#!/usr/bin/env python
"""CLI for the Telegram extraction stage.

Fetches messages inside the target year window and writes them to
data/raw/chats/*.jsonl. Applies no selection thresholds: use analyze_chats.py
afterwards to see which chats qualify.

The first run performs Telethon's interactive login (phone number, the code
Telegram sends, and a 2FA password if the account has one). The session is
then reused, so later runs are non-interactive.
"""

import argparse
import asyncio
import logging
import sys

from mini_gabriel import config
from mini_gabriel.extract import run_extraction


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # Chat names routinely contain emoji; the Windows console default
        # encoding cannot render them.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=config.TARGET_YEAR)
    parser.add_argument("--timezone", default=config.TIMEZONE)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N eligible dialogs (useful for a first look)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list eligible dialogs without fetching any messages")
    parser.add_argument("--max-members-to-fetch", type=int, default=None,
                        help="skip fetching groups larger than this (off by default). "
                             "Only a fetch-scope shortcut: selection still happens "
                             "in the analysis stage")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    try:
        asyncio.run(
            run_extraction(
                year=args.year,
                tz_name=args.timezone,
                limit=args.limit,
                dry_run=args.dry_run,
                max_members_to_fetch=args.max_members_to_fetch,
            )
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; progress is saved and the next run will resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
