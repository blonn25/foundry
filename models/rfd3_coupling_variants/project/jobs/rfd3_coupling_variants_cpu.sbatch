#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rfd3_coupling_cpu_%j.out
#SBATCH --error=logs/rfd3_coupling_cpu_%j.err
set -euo pipefail
module purge
module load CBI
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
scripts/foundry_exec.sh env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/project/software/foundry/models/rfd3_coupling_variants/src:/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src \
  python -m unittest discover -s /project/software/foundry/models/rfd3_coupling_variants/tests -v
scripts/esm_exec.sh env \
  PYTHONPATH="$PWD/software/foundry/models/rfd3_coupling_variants/src" \
  python pipeline/coupling_diversity_001/test_pipeline.py
