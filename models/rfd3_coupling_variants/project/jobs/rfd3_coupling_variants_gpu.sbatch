#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[13-19]
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rfd3_coupling_gpu_%j.out
#SBATCH --error=logs/rfd3_coupling_gpu_%j.err
set -euo pipefail
module purge
module load CBI
module load nvidia-hpc
module load nvhpc/26.5
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
scripts/foundry_exec.sh --gpu env PYTHONDONTWRITEBYTECODE=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=/project/software/foundry/models/rfd3_coupling_variants/src:/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src \
  python /project/software/foundry/models/rfd3_coupling_variants/tests/validate_gpu.py \
  /project/outputs/pipeline/coupling_diversity_001/validation_${SLURM_JOB_ID}
