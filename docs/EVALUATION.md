# EVALUATION.md — judging whether it worked

Implemented in `src/mini_gabriel/style.py` (the metric) and
`src/mini_gabriel/evaluate.py` (the harness). Neither loads a model, so both run
without a GPU.

## Why loss is not enough

Training loss cannot answer the question this project asks. A model that
memorised the training set scores excellent loss and is worthless. A model that
writes fluent, tidy, correctly punctuated prose scores well on every generic
benchmark and sounds nothing like me.

So the evaluation measures style directly, on conversations the model has never
seen.

## Three layers, cheapest first

1. **Automatic style metrics** — implemented. Objective, instant, catches gross
   failures without anyone reading anything.
2. **Side-by-side reading** — my real reply next to the generated one for the
   same context, from the held-out chats.
3. **Blind test** — mix real and generated replies, see whether a friend can
   tell them apart. The actual bar.

Only the first is automated. It is a filter, not a verdict: it can prove a model
is wrong, and cannot prove one is right.

## The metrics

Measured over replies, where a reply is one turn and keeps its message
boundaries:

| metric | what it catches |
|---|---|
| `mean_chars`, `median_chars` | writing at the wrong length |
| `mean_messages`, `multi_message_rate` | one long paragraph instead of a burst |
| `trivial_rate` | too many or too few bare acknowledgements |
| `emoji_rate` | over- or under-using emoji |
| `lowercase_start_rate` | capitalising sentences |
| `terminal_punct_rate` | ending on a full stop |
| `singlish_rate` | losing the slang |
| `question_rate` | asking more or fewer questions |

Rate metrics are compared by absolute difference, scale metrics by relative
difference capped at 1.0, so one wild value cannot dominate. The mean across all
of them is the **style distance**.

## The floor

A style distance is meaningless on its own, because two samples of genuinely the
same writing do not score zero — the metrics are estimates from finite samples.

`self_distance` establishes that floor by splitting the real replies in two and
comparing the halves. **A model is judged against the floor, not against zero.**

Measured on the held-out chats:

| | style distance | ratio to floor |
|---|---:|---|
| floor (me vs me) | **0.0237** | — |
| my replies echoed back | 0.000 | indistinguishable |
| synthetic generic-assistant prose | **0.357** | **15.1x — measurably different** |

The synthetic assistant is caught by three markers in particular:
lowercase starts 0.95 → 0.0, terminal punctuation 0.09 → 1.0, multi-message
replies 0.60 → 0.0.

## My style, measured on the held-out replies

| metric | value |
|---|---:|
| mean chars | 43.9 |
| median chars | 28.5 |
| messages per reply | 1.98 |
| multi-message replies | 59.6% |
| trivial replies | 4.2% |
| emoji | 4.2% |
| **starts lowercase** | **94.8%** |
| ends with punctuation | 9.3% |
| Singlish markers | 13.1% |
| contains a question | 13.8% |

## Running it

```bash
python scripts/evaluate.py prompts                      # held-out prompts + reference
# ... generate replies with a model, on a GPU ...
python scripts/evaluate.py score replies.jsonl --label qwen-8b-r16
```

`prompts` writes `data/processed/eval_prompts.jsonl`, one row per held-out
conversation with an `id` and its `context`.

Generation must preserve that `id`. The scorer accepts either `reply` (a list of
messages) or `text` (one string, split on newlines).

## The baseline that matters

Before reading any fine-tuned result, score the **un-fine-tuned** base model
prompted to imitate me. If plain prompting already lands close to the floor,
fine-tuning has a high bar to clear, and that is worth knowing before spending
weeks on it. Without this control, no later claim means much.

## Known limits

- The metrics are shallow by design — they measure surface habits, not whether a
  reply makes sense. A model emitting lowercase gibberish of the right length
  would score well.
- A model that memorises and regurgitates training replies scores perfectly.
  Only the held-out split guards against that, which is why the split is by whole
  chat.
- Nothing here measures coherence, relevance, or factual sanity. Layers 2 and 3
  exist for that.
