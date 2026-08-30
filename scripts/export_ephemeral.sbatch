#!/bin/bash
# Export a trained adapter as a laptop-runnable GGUF file.
#
#   sbatch scripts/export_ephemeral.sbatch runs/762734/adapter
#
# Merging needs the full 16 GB model in memory, so it happens here rather than
# on a laptop. The result is roughly 5 GB and comes back down by scp.
#
# GGUF conversion builds llama.cpp on first run, so give this more time than a
# training job would need.

#SBATCH --job-name=mg-export
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ADAPTER="${1:?usage: sbatch export_ephemeral.sbatch <adapter-dir> [quantization]}"
QUANT="${2:-q4_k_m}"

HERE="${HOME}/mini-gabriel"
WORK="/tmp/${USER}-${SLURM_JOB_ID}"
OUT="${HERE}/export"

mkdir -p "$WORK" "$OUT"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export PIP_CACHE_DIR="$WORK/pipcache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "adapter : $ADAPTER"
echo "quant   : $QUANT"
echo "start   : $(date -Is)"

echo "--- building environment ---"
python3 -m venv "$WORK/env"
source "$WORK/env/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet unsloth

echo "--- exporting ---"
# The GGUF lands in home, which has a quota. A 4-bit 8B model is about 5 GB,
# which has fitted before, but check the result rather than assuming.
python "${HERE}/export_gguf.py" \
    --adapter "$ADAPTER" \
    --output "$OUT" \
    --quantization "$QUANT"

echo "--- result ---"
ls -la "$OUT"
du -sh "$OUT" "$HOME"

rm -rf "$WORK"
echo "finish  : $(date -Is)"
