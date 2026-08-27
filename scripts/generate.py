#!/usr/bin/env python
"""Generate replies to the held-out conversations, for scoring.

Loads a trained adapter, feeds it each conversation from eval_prompts.jsonl, and
writes what it says to a JSONL that `scripts/evaluate.py score` can read.

This is the step that turns "training finished" into "did it work". The replies
are generated for conversations from chats the model never trained on, so they
test whether it learned a style rather than memorised a dataset.

Never prints message text: the replies go to the output file, not the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", type=Path, required=True,
                   help="a trained adapter directory, or a bare model name for a baseline")
    p.add_argument("--prompts", type=Path, required=True, help="eval_prompts.jsonl")
    p.add_argument("--output", type=Path, required=True, help="where to write the replies")
    p.add_argument("--limit", type=int, default=None,
                   help="only generate for the first N prompts")

    # Your replies have a median of 28 characters, so the cap is generous, not
    # tight. Greedy decoding collapses into repetition, so sample instead.
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--chat-template", default="chatml")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Unsloth before trl/transformers/peft, or its optimisations are skipped.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch

    torch.manual_seed(args.seed)

    prompts = [
        json.loads(line)
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"prompts: {len(prompts):,}")

    # Unsloth resolves the base model from the adapter's config, so an adapter
    # directory and a bare model name both work here. Passing a bare name is how
    # the un-fine-tuned baseline is generated.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        print(f"no chat template; applying {args.chat_template!r}")
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    FastLanguageModel.for_inference(model)

    # Left padding, so every sequence in a batch ends at the generation point.
    # With right padding the model would be continuing from padding tokens.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id < 0:
        eos_id = tokenizer.eos_token_id
    print(f"stopping on token id {eos_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]

            texts = [
                tokenizer.apply_chat_template(
                    [{"role": turn["speaker"] if turn["speaker"] == "system" else
                      ("assistant" if turn["speaker"] == "me" else "user"),
                      "content": turn["text"]}
                     for turn in row["context"]],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_seq_length,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Keep only what was generated, not the prompt fed back.
            for row, output in zip(batch, outputs):
                generated = output[inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                messages = [line for line in text.split("\n") if line.strip()]
                handle.write(
                    json.dumps({"id": row["id"], "reply": messages}, ensure_ascii=False) + "\n"
                )
                written += 1

            done = min(start + args.batch_size, len(prompts))
            rate = done / max(time.time() - started, 1e-6)
            print(f"  {done:>5}/{len(prompts)}  {rate:.1f}/s", flush=True)

    print(f"\nwrote {written:,} replies in {(time.time() - started) / 60:.1f} min")
    print(f"  -> {args.output}")
    print("\nScore them with:")
    print(f"  python scripts/evaluate.py score {args.output} --label <name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
