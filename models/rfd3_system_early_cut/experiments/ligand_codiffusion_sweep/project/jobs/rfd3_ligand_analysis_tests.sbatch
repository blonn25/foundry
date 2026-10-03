#!/bin/bash
#SBATCH --job-name=ligand_analysis_tests
#SBATCH --output=logs/rfd3_ligand_analysis_tests_%j.out
#SBATCH --error=logs/rfd3_ligand_analysis_tests_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
MODEL=software/foundry/models/rfd3_system_early_cut
OUTPUT="outputs/foundry/rfd3_system_early_cut/ligand_analysis_validation_${SLURM_JOB_ID}"
srun scripts/rfd3_substrate_sweep_exec.sh python -m unittest discover -s "${MODEL}/tests" -p test_ligand_analysis.py -v
srun scripts/rfd3_substrate_sweep_exec.sh python -m unittest discover -s "${MODEL}/tests" -p test_substrate_sweep.py -v
srun scripts/rfd3_substrate_sweep_exec.sh python "${MODEL}/tests/test_ligand_analysis.py" --fixture-output "/project/${OUTPUT}"
srun scripts/esm_exec.sh python "${MODEL}/src/rfd3_system_early_cut/experiments/ligand_codiffusion_plot.py" "${OUTPUT}/analysis"
