#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/coupling_caliby_%A_%a.out
#SBATCH --error=logs/coupling_caliby_%A_%a.err
set -euo pipefail
module purge
module load CBI
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python pipeline/coupling_diversity_001/worker.py prepare_caliby "$1"
scripts/caliby_exec.sh python pipeline/coupling_diversity_001/worker.py caliby "$1"
