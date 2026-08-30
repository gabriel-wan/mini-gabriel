#!/bin/bash
# Generate held-out replies at several temperatures, in one job.
#
#   sbatch scripts/sweep_temperature.sbatch runs/762734/adapter 0.4 0.5 0.6
#
# Temperature is how much the model gambles when picking each next word. The
# first run used 0.8, which is high, and is a plausible cause of replies that
# read well sometimes and badly other times.
#
# One job rather than one per temperature: building the environment and loading
# a 16 GB model costs far more than the generating does, and doing it three
# times over is most of the wall clock for none of the answer.
#
# The seed is held fixed across temperatures on purpose. Vary two things at once
# and a difference cannot be attributed to either.

#SBATCH --job-name=mg-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ADAPTER="${1:?usage: sbatch sweep_temperature.sbatch <adapter-dir> [temp ...]}"
shift
TEMPS=("$@")
if [ ${#TEMPS[@]} -eq 0 ]; then
    TEMPS=(0.4 0.5 0.6)
fi

HERE="${HOME}/mini-gabriel"
WORK="/tmp/${USER}-${SLURM_JOB_ID}"

mkdir -p "$WORK" "${HERE}/replies"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export PIP_CACHE_DIR="$WORK/pipcache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "adapter : $ADAPTER"
echo "temps   : ${TEMPS[*]}"
echo "start   : $(date -Is)"

echo "--- building environment ---"
python3 -m venv "$WORK/env"
source "$WORK/env/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet unsloth

for TEMP in "${TEMPS[@]}"; do
    LABEL="t${TEMP/./}"
    OUT="${HERE}/replies/${LABEL}.jsonl"
    echo "--- temperature ${TEMP} -> ${LABEL}.jsonl ---"
    python "${HERE}/generate.py" \
        --adapter "$ADAPTER" \
        --prompts "${HERE}/eval_prompts.jsonl" \
        --output "$OUT" \
        --temperature "$TEMP" \
        --seed 3407
done

echo "--- done ---"
wc -l "${HERE}"/replies/*.jsonl
rm -rf "$WORK"
echo "finish  : $(date -Is)"
