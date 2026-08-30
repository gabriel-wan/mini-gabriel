# DEPLOYMENT.md — running the model as a Telegram bot

The model runs on a laptop, not the cluster. Compute nodes are not reachable
from the internet and jobs are killed after three hours, so nothing there can
answer a message.

That is workable because the model is small once quantised and the replies are
short: a 4-bit 8B model is about 5 GB, and a typical reply is around ten tokens,
so even CPU generation answers in a couple of seconds.

## 1. Export a GGUF (on the cluster)

Merging needs the whole 16 GB model in memory, so it happens where there is a
GPU. The result comes back down by `scp`.

```bash
cd ~/mini-gabriel && sbatch export_ephemeral.sbatch runs/<jobid>/adapter
```

`q4_k_m` is the default quantisation - the usual quality/size trade. `q8_0` is
larger and closer to the original if the 4-bit version reads badly.

GGUF conversion builds llama.cpp on first run, so this takes longer than a
training job. If it fails, `--merged-only` at least preserves the merged model
so it can be converted separately.

Then, from the laptop:

```bash
scp -r xlogin:~/mini-gabriel/export .
```

## 2. Load it into Ollama (on the laptop)

Install Ollama, then write a `Modelfile` next to the `.gguf`:

```
FROM ./mini-gabriel.q4_k_m.gguf
```

```bash
ollama create mini-gabriel -f Modelfile
ollama run mini-gabriel
```

That last command is worth doing before touching Telegram: it confirms the
model loads and replies at all, with nothing else in the way.

## 3. Create the bot

Message **@BotFather** on Telegram, send `/newbot`, follow the prompts, and it
returns a token. Put it in `.env`, which is git-ignored:

```
TELEGRAM_BOT_TOKEN=...
```

The bot is a **separate account**, not you. People message it directly and know
they are talking to a bot.

## 4. Run it

```bash
pip install -e ".[bot]"
python scripts/telegram_bot.py
```

It polls Telegram, so no hosting, no webhook, no public IP. Stop it with Ctrl-C
and it is gone.

## How the bot builds a prompt

It imports `chatformat` rather than reimplementing the rules, so a conversation
is assembled exactly as the training examples were: same system prompt, same
ten-turn window, same user/assistant mapping, consecutive same-role turns
merged. Drift between inference formatting and training formatting degrades
output with no error to explain it.

Replies are split on newlines and sent as separate messages with a pause
between them. That is what turns the burst structure the model learned into
bursts you actually see arriving in a chat.

History is per-chat and in memory only. Restarting forgets everything, which is
the safer default for private conversations.

## Before sharing it

**Check for memorisation.** The model trained on real conversations, and names
typed inside messages were never pseudonymised - only speaker labels were. A
model that has memorised training data can reproduce them.

Prompt it with the opening of a few real conversations and see whether it
completes them verbatim. If it does, that is private content reaching whoever
is chatting, and lower rank or fewer epochs is the fix.

`scripts/compare_replies.py` is the quickest way to look for this: replies that
reference things that actually happened are the signal.

## What was deliberately not done

**No public web deployment.** A stranger cannot judge whether the replies sound
like me, so the audience for this is people who know me, and Telegram is where
they already are.

**No vLLM, no Docker.** vLLM is built for serving many concurrent users and
pre-allocates GPU memory aggressively. For one person demonstrating a project it
is the wrong tool, and the reference project lost two weeks to exactly that
before abandoning deployment entirely.
