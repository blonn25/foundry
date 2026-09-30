# Matched 4MU-Ac / 4MU-Bu release sweep

This experiment generates a 120-residue protein in each ligand context, with
21 fractions of coordinate coupling and ten matched seeds (101–110): 210 pairs,
420 structures. Inputs contain only aligned ligand poses, with fixed coordinates
and chemistry; no serine-hydrolase template or catalytic motif enters the model.
These are structural candidates, without tied sequence redesign or binding validation.

`config.json` is the editable template. The first preparation writes an immutable
`resolved_config.json` and 210-row manifest. Generation checks its input hashes and
sampling-source hashes. All model sampling uses the same checkpoint, 200 sigma
values / 199 updates, two recycles, equal mixing (kappa 0.5), and native churn.
No CFG, rigid realignment, origin jitter, or translation augmentation is used.
Fabric keeps coordinates in FP32; neural computation uses bfloat16 autocast.

Coupled fraction 0 means no coupled updates and 1 means all 199 updates. Other
fractions execute `floor(fraction * 199)` coupled updates. The first condition is
1, followed by .95 through 0. The exact release sigma is the pre-churn sigma at
the first independent update. Full coupling has no release sigma.

The common origin is the centroid of the 16 atom-name-matched substrate atoms.
Input CIFs are not edited; subtract the saved origin to reproduce the sampling
frame, or add it to generated coordinates to return to the original input frame.
The ligand has 16 heavy atoms in 4MU-Ac and 18 in 4MU-Bu. The bundled input manifest
traces their origin to the previous serine-hydrolase substrate preparation.

## CoreHPC execution

Develop and push on Wynton, then pull `origin/system_design_v0` on CoreHPC.
Restore the small files under `project/` into matching project directories;
compare existing files before replacing them. The project supplies
`scripts/foundry_exec.sh`, `scripts/esm_exec.sh`, its Foundry image/checkpoint,
and the ESM environment for Matplotlib. No new software or weights are needed.

From the CoreHPC project root, `scripts/submit_rfd3_substrate_sweep.sh` submits
the CPU preflight, pilot, pilot analysis, remaining nine seeds, and full analysis
with `afterok` dependencies and invalid-dependency cancellation. It writes job
IDs under `logs/`. Use `--pilot-job ID` only to attach the remaining stages to
an already submitted seed-101 pilot. Do not run the submitter twice for the
same active experiment.

Individual stage commands are also available:

```bash
sbatch jobs/rfd3_substrate_sweep_tests.sbatch
# After CPU checks pass:
sbatch --array=101 jobs/rfd3_substrate_sweep.sbatch
# After the 21-condition pilot passes all runtime audits:
sbatch --array=102-110 jobs/rfd3_substrate_sweep.sbatch
# After all ten seeds complete (or use afterok dependencies):
sbatch jobs/rfd3_substrate_sweep_analyze.sbatch
# Optional pilot-only analysis:
sbatch jobs/rfd3_substrate_sweep_analyze.sbatch --seeds 101
```

Every GPU array task loads the model once and runs all 21 release settings.
It resets Python, NumPy, and Torch random streams before each paired trajectory.
The sampler explicitly copies shared protein initialization and shares every
protein churn increment between tracks, even after release. Exact array hashes
verify initialization and all 199 increments also match across release settings.
There are no rotational draws because realignment is disabled.

A detached-copy sampler observer records 200 actual C-alpha states and verifies
fixed ligand drift, without modifying sampling or drawing random numbers.
CPU tests verify output/RNG parity even when an observer overwrites its copies.
Each condition additionally checks the exact coupled prefix, kappa, release flags,
noise equality, and consistency with native per-step RMSD diagnostics.

Failed attempts stay in their original directories. Retry with a new SLURM job ID;
analysis accepts exactly one complete attempt per seed and rejects duplicates.
Do not overwrite a manifest to change scientific settings: use a new experiment
root (and adjust the job's ROOT). No application controller or resubmission loop
is installed.

## Artifacts and metrics

Results are under `outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/`.
Each seed contains 21 condition directories with two native CIF/JSON outputs,
`ca_states.npz`, `noise_audit.json`, and release metadata. A seed-level completion
marker appears only after every pair and all matched-noise checks pass.

Analysis directories contain:

- `designs.csv`: all 420 structures and relative CIF paths, native sequences,
  ligand contacts/clashes, fixed-pose error, and chain-break flags.
- `pairs.csv`: all 210 pair-level final RMSDs, sequence identity, secondary
  structure assignments and fractions, and contact-map metrics.
- `trajectory.csv`: 42,000 paired state measurements. RMSD is measured before
  averaging over seeds, in both the common frame and after proper Kabsch alignment.
- `trajectory_summary.csv` and `final_summary.csv`: means, sample SDs, valid counts.
- `sequences.fasta`, `noise_audit.csv`, and machine-readable `summary.json`.
- Eight PNG/PDF figure pairs: dense/sparse trajectories against denoising fraction
  and descending log sigma, final RMSDs versus coupled fraction and release sigma,
  and topology comparisons. Sparse plots use 0, .2, .4, .6, .8, 1.

Topology uses Biotite's CA-geometry `annotate_sse` (helix/strand/coil). Report
same-label agreement across all valid positions and separately where at least
one track is structured, plus helix/strand position Jaccards. Long-range contact
Jaccard compares CA pairs below 8 Å separated by at least six residues. These are
structural similarity proxies, not a complete fold-topology classification.
Empty denominators are NA with valid counts, not perfect or zero similarity.

Protein–ligand heavy-atom contacts use 4 Å. Clashes use van der Waals overlap
above .4 Å; adjacent CA spacing outside 3.8 ± .75 Å is a chain-break flag.
All completed pairs remain in averages regardless of these geometry flags.
The never-released control is plotted separately from the release-sigma axis.
Native sequences are retained per track and are not enforced to match.

After successful jobs, use the project sync helper with
`--output-subdir foundry/rfd3_system_early_cut JOBID`; also copy the root-level
resolved manifest/configuration, whose names intentionally contain no job ID.
