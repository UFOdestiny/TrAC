#!/bin/bash
#SBATCH --job-name=trac-stage
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Run one GPU stage over all 24 main cells, split across two GPUs (finished cells are skipped).
# Account and partition are site-specific, e.g.:
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" scripts/slurm/gpu_stage.sbatch prefix_consistency
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
STAGE="${1:?usage: sbatch scripts/slurm/gpu_stage.sbatch <stage>}"
TAGS=()
for d in gsm8k math500 minerva olympiad aime3 gpqa_diamond mmlupro triviaqa; do
  for m in qwen8b llama8b ministral8b; do TAGS+=("${d}_${m}_k8"); done
done
A=(); B=()
for i in "${!TAGS[@]}"; do if (( i % 2 == 0 )); then A+=("${TAGS[$i]}"); else B+=("${TAGS[$i]}"); fi; done
bash scripts/run_stage.sh "$STAGE" 0 "${A[@]}" &
bash scripts/run_stage.sh "$STAGE" 1 "${B[@]}" &
wait
