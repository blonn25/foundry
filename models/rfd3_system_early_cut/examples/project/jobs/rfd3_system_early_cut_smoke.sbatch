#!/bin/bash
#SBATCH --job-name=rfd3early_smoke
#SBATCH --output=logs/rfd3_system_early_cut_smoke_%j.out
#SBATCH --error=logs/rfd3_system_early_cut_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[13-19]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT_DIR="/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design"
cd "${PROJECT_DIR}"
# The Foundry image requires L40S/H100 rather than Blackwell GPUs.
module purge
module load nvidia-hpc
module load nvhpc/26.5
command -v apptainer >/dev/null 2>&1 || module load SCS/apptainer/apptainer
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
OUTPUT_REL="outputs/foundry/rfd3_system_early_cut/smoke_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_REL}"
git -C software/foundry rev-parse HEAD > "${OUTPUT_REL}/../source_${SLURM_JOB_ID}.txt"
srun scripts/foundry_exec.sh --gpu env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/rfd3_system_early_cut/src:/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src" \
  python software/foundry/models/rfd3_system_early_cut/scripts/validate_early_cut_gpu.py \
  --out-dir "/project/${OUTPUT_REL}"
# Plotting uses the existing ESM environment; the Foundry image lacks Matplotlib.
srun scripts/esm_exec.sh python \
  software/foundry/models/rfd3_system_early_cut/scripts/plot_coupling_diagnostics.py \
  "${OUTPUT_REL}" --recursive
