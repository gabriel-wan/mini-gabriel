#!/usr/bin/env python
"""Export a trained adapter as a single GGUF file that runs on a laptop CPU.

Right now the model is two pieces: a 16 GB base model and a 175 MB adapter, and
running it needs a GPU. This merges them and quantises the result to 4-bit,
producing roughly 5 GB that llama.cpp or Ollama can run on an ordinary machine.

That is what makes a Telegram bot possible. The cluster cannot host one - its
compute nodes are not reachable from the internet and jobs die after three
hours - so the model has to come to a machine that is always on.

Run this on the cluster: merging needs to load the full model, which will not
fit on a laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", type=Path, required=True, help="a trained adapter directory")
    p.add_argument("--output", type=Path, required=True, help="where to write the GGUF")
    p.add_argument("--quantization", default="q4_k_m",
                   help="q4_k_m is the usual quality/size trade (~5 GB for 8B). "
                        "q8_0 is bigger and closer to the original; f16 is unquantised.")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--chat-template", default="chatml")
    p.add_argument("--merged-only", action="store_true",
                   help="stop after merging, skipping GGUF conversion")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Unsloth before trl/transformers/peft, or its optimisations are skipped.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"loading {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    # The template has to travel with the model. Without it the bot would have
    # to reconstruct ChatML by hand and any drift from the training format
    # silently degrades output.
    if not getattr(tokenizer, "chat_template", None):
        print(f"no chat template; applying {args.chat_template!r}")
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.merged_only:
        print("merging to 16-bit")
        model.save_pretrained_merged(str(args.output), tokenizer, save_method="merged_16bit")
        print(f"merged -> {args.output}")
        return 0

    # GGUF conversion builds llama.cpp on first use, which takes a while and is
    # the step most likely to fail. Merging is reported separately so a failure
    # here does not look like a failure to merge.
    print(f"merging and quantising to {args.quantization} (this is the slow part)")
    try:
        model.save_pretrained_gguf(
            str(args.output), tokenizer, quantization_method=args.quantization
        )
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        print(f"\nGGUF conversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Re-run with --merged-only to at least keep the merged model, then "
              "convert it separately with llama.cpp's convert_hf_to_gguf.py.",
              file=sys.stderr)
        return 1

    # Unsloth writes the GGUF to a sibling directory with "_gguf" appended and
    # leaves the intermediate merged model in the directory that was asked for.
    # Reporting the requested path sends you to 16 GB of safetensors instead of
    # the 5 GB file you actually want.
    gguf_dir = args.output.with_name(args.output.name + "_gguf")
    found = sorted(gguf_dir.glob("*.gguf")) or sorted(args.output.glob("*.gguf"))

    if not found:
        print(f"\nno .gguf found under {gguf_dir} or {args.output}", file=sys.stderr)
        return 1

    print(f"\nwrote GGUF to {found[0].parent}")
    for path in found:
        print(f"  {path.name}  {path.stat().st_size / 1e9:.1f} GB")
    print("\nCopy that file - not the .safetensors shards - to your laptop:")
    print(f"  scp xlogin:{found[0]} .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
