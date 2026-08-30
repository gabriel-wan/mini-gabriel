#!/usr/bin/env python
"""A Telegram bot that replies in my style, using the fine-tuned model.

Runs on a laptop against a local Ollama server. The cluster cannot host this:
its compute nodes are not reachable from the internet and jobs die after three
hours.

The conversation is built with the same rules the training data used - same
system prompt, same 10-turn window, same user/assistant mapping - by importing
`chatformat` rather than reimplementing it. If inference formatting drifts from
training formatting the model sees something unfamiliar and quality drops with
no visible error.

Replies are split on newlines and sent as separate messages, which is what
turns the burst structure the model learned into the bursts you actually see in
a chat.

Setup:
    1. ollama create mini-gabriel -f Modelfile      (see docs/DEPLOYMENT.md)
    2. get a bot token from @BotFather on Telegram
    3. put TELEGRAM_BOT_TOKEN in .env
    4. python scripts/telegram_bot.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv

from mini_gabriel.chatformat import ASSISTANT, DEFAULT_SYSTEM_PROMPT, SYSTEM, USER
from mini_gabriel.config import PROJECT_ROOT

logger = logging.getLogger("mini_gabriel.bot")

# Roughly what a real burst looks like, so messages do not all land at once.
PAUSE_BETWEEN_MESSAGES = 1.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="mini-gabriel", help="the Ollama model name")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--context-turns", type=int, default=10,
                   help="must match what the dataset was built with")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="0.5 chosen by sweep - see docs/EXPERIMENTS.md 002")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return p.parse_args()


def build_messages(history, system_prompt: str, context_turns: int) -> list[dict]:
    """Assemble the prompt exactly as training examples were assembled.

    Consecutive turns sharing a role are merged, because the chat template
    expects user and assistant to alternate - the same rule chatformat applies
    when building the dataset.
    """
    messages = [{"role": SYSTEM, "content": system_prompt}] if system_prompt else []

    for role, text in list(history)[-context_turns:]:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})

    return messages


async def generate(session, url: str, model: str, messages: list[dict],
                   temperature: float, max_tokens: int) -> list[str]:
    """Ask Ollama for a reply, and split it into messages on newlines."""
    async with session.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=120,
    ) as response:
        response.raise_for_status()
        payload = await response.json()

    text = (payload.get("message") or {}).get("content", "").strip()
    return [line.strip() for line in text.split("\n") if line.strip()]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("error: TELEGRAM_BOT_TOKEN is not set in .env.\n"
              "Get one from @BotFather on Telegram, then add:\n"
              "  TELEGRAM_BOT_TOKEN=...\n"
              "to your .env file. It is git-ignored.", file=sys.stderr)
        return 1

    try:
        import aiohttp
        from telegram import Update
        from telegram.constants import ChatAction
        from telegram.ext import Application, ContextTypes, MessageHandler, filters
    except ImportError:
        print("error: bot dependencies missing. Install them with:\n"
              "  pip install -e \".[bot]\"", file=sys.stderr)
        return 1

    # One history per chat, in memory. Restarting the bot forgets everything,
    # which is fine: this is a demo, and not persisting private conversations
    # is the safer default.
    histories = defaultdict(lambda: deque(maxlen=args.context_turns * 2))

    async def on_message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return

        chat_id = update.effective_chat.id
        histories[chat_id].append((USER, update.message.text))
        messages = build_messages(histories[chat_id], args.system_prompt, args.context_turns)

        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            async with aiohttp.ClientSession() as session:
                replies = await generate(
                    session, args.ollama_url, args.model, messages,
                    args.temperature, args.max_tokens,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("generation failed: %s", exc)
            await update.message.reply_text("(model unreachable - is ollama running?)")
            return

        if not replies:
            return

        histories[chat_id].append((ASSISTANT, "\n".join(replies)))

        # Send each message separately, with a pause, so it arrives as a burst
        # rather than one block of text.
        for index, reply in enumerate(replies):
            if index:
                await update.message.chat.send_action(ChatAction.TYPING)
                await asyncio.sleep(PAUSE_BETWEEN_MESSAGES)
            await update.message.reply_text(reply)

        logger.info("chat %s: %s messages in reply", chat_id, len(replies))

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("model=%s via %s", args.model, args.ollama_url)
    logger.info("bot running; message it on Telegram. Ctrl-C to stop.")
    app.run_polling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
