#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/coupling_fold_%A_%a.out
#SBATCH --error=logs/coupling_fold_%A_%a.err
set -euo pipefail
module purge
module load CBI
module load nvidia-hpc
module load nvhpc/26.5
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
scripts/esm_exec.sh python pipeline/coupling_diversity_001/worker.py fold "$1"
