# EXPERIMENTS.md — ML experiment log

Log of fine-tuning and evaluation experiments. Append one entry per experiment
using the template at the bottom.

Style distance is measured by `scripts/evaluate.py` against 1,434 held-out
replies from four chats no model trained on. See [EVALUATION.md](EVALUATION.md)
for what it measures and why the floor matters.

---

## Summary

| run | model | rank | epochs | style distance | vs floor |
|---|---|---:|---:|---:|---|
| — | *floor (me vs me)* | — | — | **0.0237** | 1.0x |
| 762734 | Qwen3-8B-Base + LoRA | 16 | 2 | **0.0637** | 2.7x |
| — | Qwen3-8B-Base, no fine-tuning | — | — | 0.3639 | 15.4x |

Run 762734 is listed at its generation default of temperature 0.8. Sampling
temperature was swept in 002 and moves this by less than 0.007 in either
direction; 0.5 is the adopted default.

---

## 001 — First fine-tune, and the baseline it is measured against

**Date:** 2026-08-27 (training), 2026-08-28 (evaluation)

**Objective:** Establish whether fine-tuning changes conversational style at
all, and by how much, against a measured baseline rather than an assumed one.

**Dataset:** 14,939 training rows from 54 chats. 1,434 held-out replies from 4
chats reserved for evaluation and never trained on. Built with `burst_gap=300`,
`session_gap=10800`, `context_turns=10`, `trivial_keep_rate=0.33`.

**Model:** `unsloth/Qwen3-8B-Base` (8.19B parameters, bf16). A base model rather
than an instruct one; ChatML applied via `get_chat_template`, since base models
ship none.

**Training method:** LoRA in bf16, not QLoRA. Response-only training, verified
by decoding the labels before the run rather than assumed - see
`scripts/check_response_masking.py`. 43,646,976 trainable parameters, 0.53% of
the model.

**Hyperparameters:** rank 16, alpha 32, lr 2e-4, 2 epochs, batch 8 x grad-accum
2, max_seq_length 1024, seed 3407.

**Hardware:** NUS SoC cluster, one A100-40 (a 3g.40gb MIG slice of an 80GB
card). 91.4 minutes. Slower than it should have been: `padding_free` was pinned
off, which for a 104-token mean sequence wastes most of the compute on padding.
Fixed afterwards, not yet re-measured.

**Result:**

| | style distance | ratio to floor |
|---|---:|---|
| floor (me vs me) | 0.0237 | 1.0x |
| **fine-tuned** | **0.0637** | **2.7x** |
| baseline, no fine-tuning | 0.3639 | 15.4x |

Fine-tuning reduced style distance by 83%.

Per-metric, fine-tuned against me:

| metric | me | model |
|---|---:|---:|
| lowercase_start_rate | 0.948 | 0.933 |
| multi_message_rate | 0.596 | 0.540 |
| singlish_rate | 0.131 | 0.120 |
| terminal_punct_rate | 0.093 | 0.081 |
| mean_messages | 1.98 | 1.90 |
| mean_chars | 43.9 | 32.9 |
| **trivial_rate** | **0.042** | **0.133** |

**Observations:**

The base model was writing 30-message, 410-character replies to messages like
"eh you free later". Fine-tuning brought that to 1.9 messages and 32.9
characters. The generated files differ 7x in size, which was visible before any
metric was computed.

The habits that most distinguish my writing transferred well. Lowercase starts
land within 1.5 points, and burst structure within 0.06 messages per reply.

The one clear gap is `trivial_rate`: the model produces bare acknowledgements
three times as often as I do (13.3% against 4.2%). It over-learned terseness.
This is the largest single contributor to the remaining 0.0637.

A caution about the baseline's `trivial_rate` of 0.041, which almost exactly
matches mine: it is right for the wrong reason. The base model never wrote a
short reply at all, so it never wrote a trivial one. A single metric matching
does not mean the behaviour matches.

The metric is also blind to coherence. It measures surface habits, so a model
producing well-shaped nonsense would score well. Reading the outputs is still
required.

**Next step:** Lower `trivial_keep_rate` from 0.33 toward 0.15 and re-measure -
a direct, testable fix for the largest remaining gap. Then sweep rank and
epochs now that a ranking metric exists.

---

## 002 — Inference temperature sweep

**Date:** 2026-08-31

**Objective:** Test whether sampling temperature explains replies that read well
in some conversations and badly in others. Temperature is an inference-time
setting, so this costs one generation job and no retraining - worth exhausting
before touching the training data.

**Model:** Run 762734's adapter, unchanged. No weights were modified.

**Method:** `scripts/sweep_temperature.sbatch` generates all temperatures in one
job. Building the environment and loading a 16 GB model costs far more than the
generating does, so one job per temperature would have been mostly setup. Seed
pinned at 3407 and `top_p` at 0.9 across every run, so temperature is the only
thing varying.

**Hardware:** One A100-40. 4.1 minutes per temperature, 1,434 replies each.

**Result:**

| temperature | style distance | ratio to floor |
|---:|---:|---|
| 0.4 | **0.0570** | 2.4x |
| 0.5 | 0.0582 | 2.5x |
| 0.6 | 0.0624 | 2.6x |
| 0.8 | 0.0637 | 2.7x |

**Observations:**

Style distance improves monotonically as temperature falls, but the whole spread
is 0.0067 - against a floor of 0.0237 and an untrained baseline of 0.3639. The
metric cannot separate these four runs, and reading the ranking as a trend would
be over-reading it.

The per-metric columns show why it is a wash rather than a trend. Temperature
pulls different habits in opposite directions. At 0.4 the model is closer on
Singlish (0.0007 off), lowercase starts, burst structure and length; at 0.8 it
is closer on questions (0.112 against my 0.138, versus 0.081 at 0.4), emoji and
terminal punctuation. Questions and emoji live in the tail of the distribution,
and low temperature stops reaching for them. Which end wins depends on how the
metrics are weighted.

The finding that matters is what does *not* move. `trivial_rate` is 0.1353,
0.1339, 0.1353, 0.1332 across the four runs - flat, against my 0.042. So is
`mean_chars`, at roughly 33 against my 43.9. Those two are most of the remaining
distance, and sampling does not touch either. They come from the training data
mix, which confirms from a second direction that `trivial_keep_rate` is the
lever and temperature is not.

**Decision:** 0.5 adopted as the default for `scripts/telegram_bot.py`, chosen
by reading the replies rather than by the table. The 0.0012 of style distance it
concedes to 0.4 is well inside the noise this experiment just established.

**Next step:** Unchanged by this result - lower `trivial_keep_rate` from 0.33
toward 0.15, rebuild, and retrain. This experiment rules out a cheaper fix
rather than providing one.

---

## Template

```
Experiment:
Date:
Objective:
Dataset:
Model:
Training method:
Hyperparameters:
Hardware:
Result:
Observations:
Next step:
```
