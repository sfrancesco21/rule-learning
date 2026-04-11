#!/bin/bash
#SBATCH --job-name=cake-fit
#SBATCH --output=logs/cake-fit-%A_%a.out
#SBATCH --error=logs/cake-fit-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

MANIFEST_PATH="${1:?Usage: sbatch --array=1-N slurm/fit_array.sbatch <manifest.tsv> [output_dir] [results_dir]}"
OUTPUT_DIR="${2:-model_fits}"
RESULTS_DIR="${3:-Results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set." >&2
  exit 1
fi

mkdir -p logs

# Optional environment setup:
# module load python/3.12
# source ~/path/to/venv/bin/activate

ROW="$(tail -n +2 "$MANIFEST_PATH" | sed -n "${SLURM_ARRAY_TASK_ID}p")"
if [[ -z "$ROW" ]]; then
  echo "No manifest row found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r MODEL PARTICIPANT_ID SEED <<< "$ROW"

echo "Running model=${MODEL} participant_id=${PARTICIPANT_ID} seed=${SEED}"
"${PYTHON_BIN}" fit.py \
  --model "${MODEL}" \
  --participant-id "${PARTICIPANT_ID}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --results-dir "${RESULTS_DIR}"
