# De novo substrate comparison with shared randomness

The accepted comparison generates 20 independent ordinary RFD3 structures:
160 protein residues, five seeds (101–105), neutral 4MU-butyrate or 4MU-acetate,
and either all fixed ligand atoms or all movable ligand atoms. There is no
protein template, fixed sequence, catalytic motif, JS guidance, or sequence
tying. “Unconditional” in this experiment means de novo protein generation;
ligand chemistry remains conditioning in every run.

## Input provenance

The ligand-only CIFs were made from the corresponding
`inputs/serine_hydrolase_partial_diffusion/prepared/*.cif` by removing all
protein ATOM records. Ligand atom coordinates and chemical-component tables
are retained verbatim; the original authoritative SDFs are copied unchanged.
`manifest.json` records molecular graphs, atom names, and hashes. CPU validation
checks the parsed CIF chemistry against the SDF and rejects any protein input.

The common origin is retained from the preceding comparison. It places the
fixed ligand in the same frame relative to the protein generation origin, but
does not supply enzyme geometry. For full diffusion, native RFD3 sets all
movable coordinates to zero before adding the initial Gaussian noise. Thus
free ligands start from noise as well; their input pose is not imposed on the
diffused coordinates. Molecular reference features still describe their chemistry.

## Shared randomness and settings

- Each seed uses one saved noise bank for all four conditions.
- Every protein Atom14 slot receives identical initialization and applied
  churn noise across the four conditions; rigid rotations also match.
- The 16 common ligand atoms receive identical initial and churn noise across
  the two free-ligand conditions. Butyrate's two extra carbons have their own
  reproducible streams. Fixed ligands receive no initial or churn noise.
- Protein sequence predictions and coordinate updates remain independent.
- The native 200-point schedule is used in full (199 updates), with two
  recycling cycles, gamma0 0.6, gamma_min 1, noise_scale 1.003, and step_scale
  1.5. CFG, native realignment, translational jitter, and extra contact or
  burial conditioning are disabled. FP32 input geometry is retained with
  bfloat16 neural-operation autocast, as in the preceding pilot.

## Execution and outputs

From the CoreHPC project mirror:

```bash
sbatch jobs/rfd3_shared_noise_denovo_validate.sbatch
# After successful validation:
sbatch --array=101-105 jobs/rfd3_shared_noise_denovo.sbatch
# Substitute the returned GPU array ID:
sbatch --dependency=afterok:ARRAY_ID --kill-on-invalid-dep=yes jobs/rfd3_shared_noise_denovo_collect.sbatch
```

Each GPU task generates all four conditions for one seed in one allocation.
Seed 101 also performs two-step duplicate-prefix replays and individual
denoiser replays at the initial, first churn-free, and final updates. These
short checks do not add any full designs. GPU jobs request one compatible GPU,
two CPU cores, 16 GB RAM, and 15 minutes.

Outputs are under `outputs/rfd3_shared_noise/substrate_denovo_001/seed_SEED_JOB/`.
Each condition has a final compressed CIF, FASTA, metadata, and full trajectory.
Each seed also has the noise bank, atom mappings, source hashes, resolved
configuration, and applied-noise audits. Existing directories are rejected.
Fixed ligand drift and all coordinate finiteness are checked during sampling;
the final protein must have 160 canonical amino acids. No scientific geometry
filter removes structures, and no relaxation or downstream sequence design runs.

The sampler implementation is in
[`denovo.py`](../../extensions/rfd3_shared_noise/src/rfd3_shared_noise/denovo.py).
It reuses the tested native integration and noise bank while leaving the
partial-diffusion entry point unchanged.

## Launch record

CPU validation `2045915` passed all eight tests and all four input conditions.
The earlier validation `2045877` caught loss of a custom source-ID annotation
during native concatenation; the new runner now maps ligands using native
`gt_atom_name`. That failed diagnostic remains on CoreHPC and produced no designs.
GPU array `2045953` was submitted for seeds 101–105 after successful validation.
Job IDs and the configuration hash are recorded in the immutable `launch.json`.

All twenty structures were generated, but the five GPU tasks failed afterward
while invoking the unavailable `git` executable to record source provenance.
Collector `2046019` was therefore canceled. Successful recovery/analysis job
`2051817` independently validated all structures, trajectories, and noise
banks, and passed ten tests. No designs were rerun. The accepted index is
`outputs/rfd3_shared_noise/substrate_denovo_001/analysis_2051817/designs.csv`.
Original failure markers are retained; the recovery does not change SLURM
accounting or create misleading seed-level completion markers. Future generation
reads Git reference files before inference to avoid this bookkeeping error.
See [results and plots](../../docs/rfd3_denovo_substrates.md).
