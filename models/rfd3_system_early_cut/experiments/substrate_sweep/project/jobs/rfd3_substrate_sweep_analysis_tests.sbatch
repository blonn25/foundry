#!/bin/bash
#SBATCH --job-name=substrate_analysis_tests
#SBATCH --output=logs/rfd3_substrate_sweep_analysis_tests_%j.out
#SBATCH --error=logs/rfd3_substrate_sweep_analysis_tests_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
srun scripts/rfd3_substrate_sweep_exec.sh python -m unittest discover \
  -s software/foundry/models/rfd3_system_early_cut/tests -p test_substrate_sweep.py -v
srun scripts/esm_exec.sh python software/foundry/models/rfd3_system_early_cut/experiments/substrate_sweep/check_plots.py
