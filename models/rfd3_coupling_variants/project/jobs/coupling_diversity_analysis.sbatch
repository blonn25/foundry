#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/coupling_analysis_%j.out
#SBATCH --error=logs/coupling_analysis_%j.err
set -euo pipefail
module purge
module load CBI
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "${2:-}" == calibration ]]; then
  scripts/esm_exec.sh python pipeline/coupling_diversity_001/collect.py "$1" --calibrate
else
  scripts/esm_exec.sh python pipeline/coupling_diversity_001/collect.py "$1"
  scripts/esm_exec.sh python pipeline/coupling_diversity_001/diversity.py "$1"
  scripts/esm_exec.sh python pipeline/coupling_diversity_001/plot.py "$1"
fi
