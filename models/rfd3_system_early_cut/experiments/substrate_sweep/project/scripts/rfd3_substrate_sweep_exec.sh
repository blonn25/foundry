#!/usr/bin/env bash
# One environment entry point for the isolated early-cut experiment.
set -euo pipefail
PROJECT_DIR="/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design"
cd "${PROJECT_DIR}"
GPU=()
if [[ "${1:-}" == --gpu ]]; then GPU=(--gpu); shift; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FOUNDRY_REVISION
FOUNDRY_REVISION=$(git -C software/foundry rev-parse HEAD)
exec scripts/foundry_exec.sh "${GPU[@]}" env \
  PYTHONDONTWRITEBYTECODE=1 FOUNDRY_REVISION="${FOUNDRY_REVISION}" \
  PYTHONPATH="/project/software/foundry/models/rfd3_system_early_cut/src:/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src" \
  "$@"
