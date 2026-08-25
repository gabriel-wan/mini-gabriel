"""Configuration: credentials and filesystem paths.

This module deliberately does not import Telethon so that it stays usable from
the pure selection/analysis code and from the test suite.

Credentials are read from a local ``.env`` file and are never logged, printed,
or written to any output file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHATS_DIR = RAW_DIR / "chats"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# The session file lives under data/ and is matched by the "*.session" rule in
# .gitignore. It is a credential: anyone holding it can act as the account.
DEFAULT_SESSION_PATH = DATA_DIR / "mini_gabriel.session"

# Extraction scope. The window is a calendar year in TIMEZONE, converted to UTC.
TARGET_YEAR = 2026
TIMEZONE = "Asia/Singapore"

# Chat-selection thresholds. Applied by the analysis stage, never by extraction.
MAX_PARTICIPANTS = 20
MIN_MY_TEXT_MESSAGES = 100


@dataclass(frozen=True)
class TelegramCredentials:
    """Telegram API credentials. Never log or serialise this object."""

    api_id: int
    api_hash: str
    session_path: Path


def load_credentials(env_path: Optional[Path] = None) -> TelegramCredentials:
    """Read credentials from ``.env``.

    Raises RuntimeError with an actionable message if anything is missing, and
    never includes the credential values in that message.
    """
    load_dotenv(env_path or PROJECT_ROOT / ".env")

    api_id_raw = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    missing = [
        name
        for name, value in (("TELEGRAM_API_ID", api_id_raw), ("TELEGRAM_API_HASH", api_hash))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing {' and '.join(missing)} in .env. "
            f"Create {PROJECT_ROOT / '.env'} with your credentials from https://my.telegram.org "
            "(API development tools). The .env file is git-ignored."
        )

    try:
        api_id = int(str(api_id_raw).strip())
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_API_ID must be an integer.") from exc

    session_override = os.getenv("TELEGRAM_SESSION_PATH")
    session_path = Path(session_override) if session_override else DEFAULT_SESSION_PATH

    return TelegramCredentials(api_id=api_id, api_hash=str(api_hash).strip(), session_path=session_path)


def ensure_data_dirs() -> None:
    """Create the git-ignored data directories if they do not exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, CHATS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
