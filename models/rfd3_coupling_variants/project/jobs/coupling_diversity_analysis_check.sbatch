#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=logs/coupling_analysis_check_%j.out
#SBATCH --error=logs/coupling_analysis_check_%j.err
set -euo pipefail
module purge
module load CBI
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
scripts/esm_exec.sh python pipeline/coupling_diversity_001/validate_analysis.py \
  outputs/pipeline/coupling_diversity_001/validation_2126955 \
  outputs/pipeline/coupling_diversity_001/analysis_check_${SLURM_JOB_ID}
