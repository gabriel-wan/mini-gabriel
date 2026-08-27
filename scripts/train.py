#!/usr/bin/env python
"""Fine-tune a base model on my conversational style with LoRA.

The Unsloth calls here follow their official Qwen3 conversational notebook
(unslothai/notebooks, nb/Qwen3_(14B)-Reasoning-Conversational.ipynb) rather than
being written from memory. Where this departs from that notebook it is
deliberate and commented, because silently drifting from a working example is
how these runs waste days.

Departures from the notebook:

* ``load_in_4bit=False``. The notebook uses QLoRA; an A100-40 holds an 8B model
  in bf16 with room to spare, so there is no reason to accept 4-bit precision
  loss and a slower forward pass. See docs/CLUSTER.md.
* ``max_seq_length=1024`` instead of 2048. The p99 example is 484 tokens and
  only 37 of 15,567 exceed 768, so 2048 would be padding nobody needs.
* ``num_train_epochs`` instead of ``max_steps=30``. Thirty steps is a demo
  value; it would touch a fraction of the dataset.

Never prints message text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True, help="train_messages.jsonl")
    p.add_argument("--output", type=Path, required=True, help="where to write the adapter")
    p.add_argument("--model", default="unsloth/Qwen3-8B-Base")

    # The sweep knobs. See docs/EVALUATION.md for how to compare the results.
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (primary knob)")
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default: 2x rank)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--load-in-4bit", action="store_true",
                   help="use QLoRA; only needed for models larger than ~13B on a 40GB card")
    p.add_argument("--response-only", action="store_true",
                   help="train only on my replies, masking the other side out of the loss. "
                        "Run scripts/check_response_masking.py first; the job wrapper does "
                        "this and only passes this flag once that check passes.")
    # These must match what check_response_masking.py verified against the chat
    # template. Passing them explicitly rather than relying on defaults means the
    # check and the real run cannot silently disagree about turn boundaries.
    p.add_argument("--instruction-part", default="<|im_start|>user\n")
    p.add_argument("--response-part", default="<|im_start|>assistant\n")
    p.add_argument("--chat-template", default="chatml",
                   help="applied only if the tokenizer has none of its own, which is "
                        "the usual case for a base model (default: chatml)")
    p.add_argument("--label", default=None, help="name for this run in the summary")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Unsloth must be imported before trl, transformers and peft or its
    # optimisations are not applied - it warns about this at runtime. Imported
    # inside main so --help still works without the training stack present.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    label = args.label or f"r{args.rank}-lr{args.lr}-e{args.epochs}"
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"=== run: {label} ===", flush=True)

    # ---------------------------------------------------------------- model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Base models generally ship no chat template, and Qwen3-8B-Base is no
    # exception. ChatML is what Qwen's own instruct models use and its control
    # tokens are already in this tokenizer's vocabulary, so applying it teaches
    # the model a format it can already represent rather than inventing one.
    if not getattr(tokenizer, "chat_template", None):
        print(f"no chat template on this tokenizer; applying {args.chat_template!r}")
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    if not getattr(tokenizer, "chat_template", None):
        print("error: still no chat template after get_chat_template; the dataset "
              "cannot be rendered.", file=sys.stderr)
        return 1

    # The failure that made the reference project emit endless emoji: when pad
    # and eos are the same id, the padding mask also masks the real end-of-turn
    # token, so the model never learns to stop.
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    print(f"pad={tokenizer.pad_token!r} ({pad_id})  eos={tokenizer.eos_token!r} ({eos_id})")
    if pad_id is not None and pad_id == eos_id:
        print("WARNING: pad_token_id == eos_token_id. The model may never learn to stop. "
              "Set a distinct pad token before training.", file=sys.stderr)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.alpha or args.rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # ---------------------------------------------------------------- data
    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def render(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(messages, tokenize=False)
                for messages in batch["messages"]
            ]
        }

    dataset = dataset.map(render, batched=True, remove_columns=dataset.column_names)
    print(f"examples: {len(dataset):,}")

    # ---------------------------------------------------------------- train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(args.output / "checkpoints"),
            report_to="none",
            padding_free=False,
        ),
    )

    if args.response_only:
        # Masks everything but my replies out of the loss, so the other person's
        # messages are read as context but never scored. The markers are passed
        # explicitly and are the same ones check_response_masking.py verified
        # against this model's chat template - relying on library defaults here
        # would let the check and the real run disagree without either failing.
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part=args.instruction_part,
            response_part=args.response_part,
        )
        print(f"training on responses only "
              f"(instruction={args.instruction_part!r} response={args.response_part!r})")

    started = time.time()
    stats = trainer.train()
    minutes = (time.time() - started) / 60

    # ---------------------------------------------------------------- save
    adapter_dir = args.output / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "label": label,
        "model": args.model,
        "examples": len(dataset),
        "minutes": round(minutes, 1),
        "train_loss": getattr(stats, "training_loss", None),
        "config": {
            "rank": args.rank,
            "alpha": args.alpha or args.rank * 2,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_length": args.max_seq_length,
            "load_in_4bit": args.load_in_4bit,
            "response_only": args.response_only,
            "seed": args.seed,
        },
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\ndone in {minutes:.1f} min, loss {summary['train_loss']}")
    print(f"adapter -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
