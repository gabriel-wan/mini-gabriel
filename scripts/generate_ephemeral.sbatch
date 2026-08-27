#!/bin/bash
# Generate held-out replies from a trained adapter, for scoring.
#
#   sbatch scripts/generate_ephemeral.sbatch runs/762734/adapter fine-tuned
#   sbatch scripts/generate_ephemeral.sbatch unsloth/Qwen3-8B-Base baseline
#
# The second form passes a bare model name instead of an adapter, which produces
# the un-fine-tuned baseline. Without that comparison a fine-tuned score means
# nothing: it could be that the base model was already close.
#
# Builds its environment on node-local NVMe for the same reason the training job
# does - see docs/CLUSTER.md.

#SBATCH --job-name=mg-generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ADAPTER="${1:?usage: sbatch generate_ephemeral.sbatch <adapter-dir-or-model-name> <label>}"
LABEL="${2:-model}"
shift 2 || true

HERE="${HOME}/mini-gabriel"
WORK="/tmp/${USER}-${SLURM_JOB_ID}"
OUT="${HERE}/replies/${LABEL}.jsonl"

mkdir -p "$WORK" "$(dirname "$OUT")"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export PIP_CACHE_DIR="$WORK/pipcache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "adapter : $ADAPTER"
echo "label   : $LABEL"
echo "start   : $(date -Is)"

echo "--- building environment ---"
python3 -m venv "$WORK/env"
source "$WORK/env/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet unsloth

echo "--- generating ---"
python "${HERE}/generate.py" \
    --adapter "$ADAPTER" \
    --prompts "${HERE}/eval_prompts.jsonl" \
    --output "$OUT" \
    "$@"

echo "--- done ---"
wc -l "$OUT"
rm -rf "$WORK"
echo "finish  : $(date -Is)"
