#!/bin/bash
#SBATCH --job-name=ligand_coupling_tests
#SBATCH --output=logs/rfd3_ligand_codiffusion_tests_%j.out
#SBATCH --error=logs/rfd3_ligand_codiffusion_tests_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
set -euo pipefail
cd /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
for TEST in test_early_cut.py test_substrate_sweep.py test_ligand_coupling.py; do
  srun scripts/rfd3_substrate_sweep_exec.sh python -m unittest discover \
    -s software/foundry/models/rfd3_system_early_cut/tests -p "${TEST}" -v
done
srun scripts/rfd3_substrate_sweep_exec.sh python -m rfd3_system_early_cut.experiments.ligand_codiffusion_sweep validate \
  --root "/project/outputs/foundry/rfd3_system_early_cut/ligand_preflight_${SLURM_JOB_ID}"
