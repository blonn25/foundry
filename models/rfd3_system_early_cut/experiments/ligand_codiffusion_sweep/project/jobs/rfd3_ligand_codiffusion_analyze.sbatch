#!/bin/bash
#SBATCH --job-name=ligand_analysis
#SBATCH --output=logs/rfd3_ligand_codiffusion_analyze_%j.out
#SBATCH --error=logs/rfd3_ligand_codiffusion_analyze_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
ROOT=outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001
OUTPUT="${ROOT}/analysis_${SLURM_JOB_ID}"
# The pilot uses --seeds 101; all comparisons use only those same baseline seeds.
srun scripts/rfd3_substrate_sweep_exec.sh python -m rfd3_system_early_cut.experiments.ligand_codiffusion_analysis \
  --root "/project/${ROOT}" --output "/project/${OUTPUT}" "$@"
srun scripts/esm_exec.sh python \
  software/foundry/models/rfd3_system_early_cut/src/rfd3_system_early_cut/experiments/ligand_codiffusion_plot.py "${OUTPUT}"
