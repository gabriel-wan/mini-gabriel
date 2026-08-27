#!/usr/bin/env python
"""Verify that response-only training masks the right tokens, before trusting it.

`train_on_responses_only` sets the loss labels of everything except my replies
to -100, so the model reads the other person's messages as context but is only
scored on producing mine. That is what this project wants. But it works by
string-matching turn markers against the chat template, so a marker that does
not match the template fails in one of two silent ways:

* nothing is masked, and the model is trained to write both sides
* everything is masked, and the model is trained on nothing at all

Neither raises an error. Both produce a plausible loss curve. So this checks the
labels directly and exits non-zero if they are wrong.

The conversation used here is fictional. Masking depends on the template's
markers rather than on content, so a synthetic example proves the same thing
without writing real messages into a job log.
"""

from __future__ import annotations

import argparse
import sys

# Fictional, and deliberately distinctive so it is easy to find in the decode.
FIXTURE = [
    {"role": "system", "content": "SYSTEMPROMPTMARKER"},
    {"role": "user", "content": "USERALPHA are you coming later"},
    {"role": "assistant", "content": "GABRIELALPHA\nya gimme 10"},
    {"role": "user", "content": "USERBRAVO ok see you"},
    {"role": "assistant", "content": "GABRIELBRAVO ok"},
]

USER_MARKERS = ["USERALPHA", "USERBRAVO"]
REPLY_MARKERS = ["GABRIELALPHA", "GABRIELBRAVO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="unsloth/Qwen3.5-9B-Base")
    p.add_argument("--instruction-part", default="<|im_start|>user\n")
    p.add_argument("--response-part", default="<|im_start|>assistant\n")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--show-template", action="store_true",
                   help="print the rendered conversation and exit, to read the real markers")
    return p.parse_args()


def fail(message: str) -> int:
    print(f"\nFAILED: {message}", file=sys.stderr)
    print("Do not enable --response-only until this passes.", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    print(f"=== response-masking check: {args.model} ===\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    if not getattr(tokenizer, "chat_template", None):
        return fail("tokenizer has no chat template")

    rendered = tokenizer.apply_chat_template(FIXTURE, tokenize=False)

    print("--- rendered by the model's own chat template ---")
    print(repr(rendered))
    print()

    if args.show_template:
        return 0

    # 1. The markers must actually occur in what the template produces. This is
    #    the check that catches a template whose turn markers differ from ChatML.
    for name, marker in (("instruction_part", args.instruction_part),
                         ("response_part", args.response_part)):
        if marker not in rendered:
            print(f"--- chat_template ---\n{tokenizer.chat_template}\n", file=sys.stderr)
            return fail(
                f"{name}={marker!r} does not appear in the rendered conversation. "
                "Read the rendered output above and pass the correct markers."
            )
    print(f"markers found: instruction={args.instruction_part!r} "
          f"response={args.response_part!r}")

    # 2. Build a minimal trainer and apply the masking.
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16,
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
        random_state=3407, use_rslora=False, loftq_config=None,
    )

    dataset = Dataset.from_list([{"text": rendered}] * 2)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=args.max_seq_length,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            num_train_epochs=1, learning_rate=2e-4, logging_steps=1,
            output_dir="/tmp/masking-check", report_to="none", seed=3407,
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part=args.instruction_part,
        response_part=args.response_part,
    )

    # 3. Pull a real batch and read the labels the trainer would actually use.
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    kept = [i for i, label in enumerate(labels.tolist()) if label != -100]
    masked = [i for i, label in enumerate(labels.tolist()) if label == -100]

    trained_text = tokenizer.decode([input_ids[i] for i in kept])
    masked_text = tokenizer.decode([input_ids[i] for i in masked])

    print(f"\ntokens: {len(labels)} total, {len(kept)} trained on, {len(masked)} masked")
    print(f"\n--- tokens the loss is computed on ---\n{trained_text!r}")
    print(f"\n--- tokens masked out ---\n{masked_text!r}\n")

    # 4. The assertions that matter.
    if not kept:
        return fail("every token is masked; the model would train on nothing")
    if not masked:
        return fail("nothing is masked; the model would train on both sides")

    for marker in REPLY_MARKERS:
        if marker not in trained_text:
            return fail(f"my reply {marker!r} is not in the trained tokens; "
                        "the response is being masked out")

    for marker in USER_MARKERS:
        if marker in trained_text:
            return fail(f"the other person's message {marker!r} is in the trained "
                        "tokens; their text is not being masked")
        if marker not in masked_text:
            return fail(f"the other person's message {marker!r} is missing from the "
                        "input entirely; context is being dropped, not masked")

    print("PASSED")
    print("  - my replies are in the loss")
    print("  - the other person's messages are visible as context but not in the loss")
    print("  - neither everything nor nothing is masked")
    print("\nSafe to run training with --response-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
