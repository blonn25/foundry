#!/usr/bin/env bash
# Submit a fixed DAG; SLURM success dependencies are the only controller.
set -euo pipefail
PROJECT_DIR=/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design
cd "${PROJECT_DIR}"
PILOT_JOB=""
if [[ $# -eq 2 && "$1" == --pilot-job && "$2" =~ ^[0-9]+$ ]]; then
  PILOT_JOB="$2"
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--pilot-job EXISTING_SEED_101_ARRAY_ID]" >&2
  exit 64
fi
if [[ -n "$(git -C software/foundry status --porcelain)" ]]; then
  echo "Foundry execution checkout must be clean before submission." >&2
  exit 65
fi
git -C software/foundry pull --ff-only origin system_design_v0
PREFLIGHT_JOB=already_validated
if [[ -z "${PILOT_JOB}" ]]; then
  PREFLIGHT_JOB=$(sbatch --parsable jobs/rfd3_substrate_sweep_tests.sbatch)
  PILOT_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${PREFLIGHT_JOB}" \
    --array=101 jobs/rfd3_substrate_sweep.sbatch)
fi
PILOT_ANALYSIS_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${PILOT_JOB}" \
  jobs/rfd3_substrate_sweep_analyze.sbatch --seeds 101)
PRODUCTION_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${PILOT_ANALYSIS_JOB}" \
  --array=102-110 jobs/rfd3_substrate_sweep.sbatch)
ANALYSIS_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${PRODUCTION_JOB}:${PILOT_ANALYSIS_JOB}" \
  jobs/rfd3_substrate_sweep_analyze.sbatch)
printf 'preflight=%s\npilot=%s\npilot_analysis=%s\nproduction=%s\nfull_analysis=%s\n' \
  "${PREFLIGHT_JOB}" "${PILOT_JOB}" "${PILOT_ANALYSIS_JOB}" "${PRODUCTION_JOB}" "${ANALYSIS_JOB}" \
  | tee "logs/rfd3_substrate_sweep_submission_${PILOT_JOB}.txt"
