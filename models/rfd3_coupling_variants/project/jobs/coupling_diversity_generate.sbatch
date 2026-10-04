#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[13-19]
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/coupling_generate_%A_%a.out
#SBATCH --error=logs/coupling_generate_%A_%a.err
set -euo pipefail
module purge
module load CBI
module load nvidia-hpc
module load nvhpc/26.5
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python pipeline/coupling_diversity_001/worker.py generate "$1"
