#!/bin/bash
# Fine-tuning run that needs no persistent cluster storage.
#
#   sbatch scripts/train_ephemeral.sbatch --rank 16 --epochs 2
#
# The home quota is ~120 MB, far too small for PyTorch (see docs/CLUSTER.md).
# This builds the whole environment on the GPU node's local NVMe instead, which
# is 3.1 TB and unquota'd, and downloads the model there directly - compute
# nodes have outbound internet, so nothing needs pre-staging.
#
# Everything under /tmp is wiped when the job ends. Only the adapter is copied
# back, and it is small.

#SBATCH --job-name=mini-gabriel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

HERE="${HOME}/mini-gabriel"
MODEL="${MG_MODEL:-unsloth/Qwen3-8B-Base}"

WORK="/tmp/${USER}-${SLURM_JOB_ID}"
OUT="${HERE}/runs/${SLURM_JOB_ID}"
mkdir -p "$WORK" "$OUT"

# Keep pip, HuggingFace and every temp file off the quota'd home directory.
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export PIP_CACHE_DIR="$WORK/pipcache"
mkdir -p "$TMPDIR" "$HF_HOME"

echo "job   : ${SLURM_JOB_ID} on $(hostname)"
echo "model : $MODEL"
echo "work  : $WORK  ($(df -h /tmp | awk 'NR==2{print $4}') free)"
echo "start : $(date -Is)"

echo "--- building environment (once per job) ---"
python3 -m venv "$WORK/env"
source "$WORK/env/bin/activate"
pip install --quiet --upgrade pip
time pip install --quiet unsloth

echo "--- gpu ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Response-only training masks the other person's messages out of the loss, so
# the model reads them as context but is scored only on my replies. It works by
# string-matching turn markers against the chat template, and a marker that does
# not match the template fails silently in one of two ways: nothing masked, so
# the model learns to write both sides, or everything masked, so it learns
# nothing at all. Both produce a plausible loss curve, which is why the labels
# are inspected directly before the real run is allowed to start.
echo "--- verifying response masking ---"
if python "${HERE}/check_response_masking.py" --model "$MODEL"; then
    RESPONSE_ONLY="--response-only"
else
    echo "masking check failed; refusing to train rather than silently learning" >&2
    echo "the wrong thing. Re-run that script with --show-template to read the" >&2
    echo "chat template's real turn markers, then pass them explicitly." >&2
    exit 1
fi

echo "--- training ---"
# Run every configuration inside this one job: the environment is already built,
# so additional runs cost only their own training time.
python "${HERE}/train.py" \
    --data "${HERE}/train_messages.jsonl" \
    --output "$OUT" \
    --model "$MODEL" \
    $RESPONSE_ONLY \
    "$@"

echo "--- results ---"
du -sh "$OUT"
ls -la "$OUT"

# /tmp is wiped by the scheduler, but be explicit rather than relying on it.
rm -rf "$WORK"
echo "finish: $(date -Is)"
