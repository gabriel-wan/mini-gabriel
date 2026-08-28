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
