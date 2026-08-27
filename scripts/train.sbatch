#!/bin/bash
# Slurm wrapper for a fine-tuning run on the NUS SoC cluster.
#
#   sbatch scripts/train.sbatch
#
# The gpu partition caps jobs at 3 hours, which is ample: a run over this
# dataset is roughly 25 minutes. See docs/CLUSTER.md for why A100-40.

#SBATCH --job-name=mini-gabriel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "job     : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started : $(date -Is)"

source "${MG_ENV:-$HOME/mg-env}/bin/activate"

echo "--- gpu ---"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "--- versions ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

# Training entry point. Not yet written: it needs Unsloth's current API, which
# is taken from their official notebook rather than reconstructed.
python scripts/train.py "$@"

echo "finished: $(date -Is)"
