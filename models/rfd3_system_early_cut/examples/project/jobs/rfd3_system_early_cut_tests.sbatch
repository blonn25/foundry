#!/bin/bash
#SBATCH --job-name=rfd3early_tests
#SBATCH --output=logs/rfd3_system_early_cut_tests_%j.out
#SBATCH --error=logs/rfd3_system_early_cut_tests_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
PROJECT_DIR="/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design"
cd "${PROJECT_DIR}"
command -v apptainer >/dev/null 2>&1 || module load SCS/apptainer/apptainer
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun scripts/foundry_exec.sh env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/rfd3_system_early_cut/src:/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src" \
  python -m unittest discover \
  -s software/foundry/models/rfd3_system_early_cut/tests -p test_early_cut.py -v
