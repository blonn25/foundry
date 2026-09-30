#!/bin/bash
#SBATCH --job-name=substrate_sweep
#SBATCH --output=logs/rfd3_substrate_sweep_%A_%a.out
#SBATCH --error=logs/rfd3_substrate_sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[13-19]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
module purge
module load nvidia-hpc
module load nvhpc/26.5
ROOT=/project/outputs/foundry/rfd3_system_early_cut/substrate_sweep_001
SEED="${SLURM_ARRAY_TASK_ID:?Submit --array=101 for pilot, then --array=102-110}"
# Only the pilot creates the immutable campaign manifest, before other seeds run.
if [[ "${SEED}" == 101 ]]; then
  srun scripts/rfd3_substrate_sweep_exec.sh python -m rfd3_system_early_cut.experiments.substrate_sweep prepare --root "${ROOT}"
fi
srun scripts/rfd3_substrate_sweep_exec.sh --gpu python -m rfd3_system_early_cut.experiments.substrate_sweep run \
  --root "${ROOT}" --seed "${SEED}" --attempt "${SLURM_ARRAY_JOB_ID}_${SEED}"
