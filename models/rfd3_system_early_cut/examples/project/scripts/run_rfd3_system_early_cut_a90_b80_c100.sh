#!/usr/bin/env bash
set -euo pipefail

# Run the rfd3_system_early_cut shared-chain prototype from the CoreHPC project root.
# This script intentionally keeps outputs outside software/foundry/ and uses
# the Foundry Apptainer image through scripts/foundry_exec.sh.
PROJECT_DIR="${PROJECT_DIR:-/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design}"
RUN_NAME="${RUN_NAME:-a90_b80_c100_default_steps}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rfd3earlycut_a90_b80_c100}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-3}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"

HOST_OUT_DIR="${PROJECT_DIR}/outputs/foundry/rfd3_system_early_cut/${RUN_NAME}_${JOB_TAG}"
CONTAINER_OUT_DIR="/project/outputs/foundry/rfd3_system_early_cut/${RUN_NAME}_${JOB_TAG}"

cd "${PROJECT_DIR}"
mkdir -p "${HOST_OUT_DIR}" logs runtime/tmp runtime/home

echo "Writing rfd3_system_early_cut coupled outputs to ${HOST_OUT_DIR}"

# The base de novo contig creates the internal ABC source atom array.  The
# track-specific overrides then split it into A+B and A+C views, with A shared
# by the coupled sampler.  Do not set base select_* fields for no-input de novo
# specs; those selections require an input atom array.  Track select_* fields
# are valid because each track receives an atom_array_input after splitting.
scripts/foundry_exec.sh --gpu \
  env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/rfd3_system_early_cut/src:/project/software/foundry/src" \
  python -m rfd3_system_early_cut.cli design \
    inputs=null \
    "out_dir=${CONTAINER_OUT_DIR}" \
    "global_prefix=${OUTPUT_PREFIX}" \
    ckpt_path=/weights/rfd3_latest.ckpt \
    "+specification.contig='90,/0,80,/0,100'" \
    +specification.length=270 \
    "diffusion_batch_size=${DIFFUSION_BATCH_SIZE}" \
    n_batches=1 \
    seed=123 \
    dump_trajectories=False \
    coupling_mode=superdiff_shared_chain \
    inference_sampler.kind=superdiff_shared_chain \
    shared_chain_id=A \
    "complex_1_partners=[B]" \
    "complex_2_partners=[C]" \
    "+track_1_specification.contig='A1-90,/0,B1-80'" \
    +track_1_specification.length=170 \
    +track_1_specification.select_fixed_atoms=false \
    +track_1_specification.select_unfixed_sequence=true \
    "+track_2_specification.contig='A1-90,/0,C1-100'" \
    +track_2_specification.length=190 \
    +track_2_specification.select_fixed_atoms=false \
    +track_2_specification.select_unfixed_sequence=true \
    "$@"
