# Matched substrate co-diffusion release sweep

Generate 120-residue proteins with 4MU-Ac in track 1 and 4MU-Bu in track 2.
Ligand chemistry is fixed, but all ligand coordinates diffuse. The inference
input explicitly pairs C1–C12 and O1–O4 across the substrates; the two extra
butyrate carbons remain independent. Protein and mapped ligand atoms share
initialization, coupled updates until release, and churn throughout sampling.
This code lives exclusively in `rfd3_system_early_cut`.

The same 21 fractions (1, .95, …, 0), seeds 101–110, 200 sigma values / 199
updates, two recycles, 50:50 mixing, native churn, FP32 coordinates and
bfloat16 neural evaluation are used as in the fixed-ligand substrate sweep.
There is no CFG, rigid realignment, origin jitter, sequence tying, or
post-generation refinement. Fraction f couples floor(f × 199) updates.
Fraction zero shares initialization/noise but no updates. Full coupling has
no release sigma. Both protein and mapped ligand coupling use this boundary.

This is de novo full diffusion from noise, with no hydrolase protein template.
The input conformers provide chemistry/reference features and atom identity;
they are not retained as fixed ligand poses. All ligand coordinates are zero
before the native full-noise initialization, which is checked during preflight.
Input provenance remains in `inputs/substrate_denovo_001/manifest.json`.

## Inputs and execution

`config.json` contains the full explicit atom map; edit that template before
preparing a new experiment. The general public option is documented in
[early-cut usage](../../docs/early_cut.md). Empty mapping preserves the previous
early-cut behavior. This sweep requests all 16 common atoms explicitly.
Element/charge compatibility and the mapped bond subgraph are validated while
allowing different external substituents.

The project entry point is:

```bash
# On CoreHPC, from the project mirror:
scripts/submit_rfd3_ligand_codiffusion_sweep.sh
# To attach analysis and production to an already submitted seed-101 pilot:
scripts/submit_rfd3_ligand_codiffusion_sweep.sh --pilot-job PILOT_ARRAY_ID
```

The submitter uses a fixed SLURM dependency graph: CPU preflight → seed-101
pilot → pilot analysis → seeds 102–110 → full analysis. A separate CPU job
validates synthetic analysis and plotting before pilot analysis can run.
Invalid dependencies cancel descendants. A submission record prevents a
duplicate launch. Failed attempts remain untouched; analyze exactly one
complete attempt per seed. Recovery should use new job IDs after inspecting
the recorded failure, not overwrite the immutable experiment configuration.

Each GPU task runs all 21 settings for one seed. The sampler retains the
original protein RNG draw order and uses a separate seeded generator for
mapped-ligand churn. The run checks protein initialization and every churn
hash against the corresponding fixed-ligand baseline, and checks protein
and ligand hashes across all release conditions. Geometry differences are
therefore not attributable to different injected protein noise. All mapped
states must remain exactly identical through the coupled prefix.

The baseline must exist at
`outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/`; analysis reads
its raw CA trajectories and audits, so it does not require a particular
baseline analysis job. The pilot compares only seed 101 on both sides.
Restore the new `project/` job/launcher files and the original substrate
sweep's input bundle and runtime helper. Foundry/ESM environments, weights,
and container are unchanged. Source edits and Git pushes occur on Wynton;
validation/inference occur through CoreHPC SLURM.

## Outputs and interpretation

Root: `outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/`.
The immutable manifest has 210 pairs / 420 structures. Each condition records
two native CIFs/JSONs, `ca_states.npz`, `ligand_states.npz`, the explicit ligand
atom order, noise audits, and release metadata. Ligand states contain both
all per-track ligand atoms and the ordered mapped subset. Native sequences
remain independent. No geometry-based selection is applied.

Analysis retains the original CA dynamics and topology measurements and adds:

- Mapped ligand RMSD in the common frame, after ligand-only fitting, and
  after protein CA fitting without a second ligand fit, at all 200 states.
- Whole-ligand reference-fitted RMSD, bond-length RMSE/max deviation from
  the input conformer, protein contacts/clashes, and ligand-to-protein
  centroid distance. Input bond lengths are a reference, not ideal chemistry;
  centroid distance is not a burial or affinity metric.
- Fixed-baseline CA and topology values, plus paired differences
  (co-diffused minus fixed for the same seed/fraction). Means and sample SDs
  are calculated after taking within-pair measurements/differences.
- 19 PNG/PDF plot pairs: eight original CA/topology figures, four ligand
  dynamics figures, four paired CA-difference trajectories, two final
  fixed-versus-co-diffused comparisons, and one ligand geometry figure.
  Dense views use all 21 conditions; sparse views use 0, .2, .4, .6, .8, 1.

`designs.csv`, `pairs.csv`, `trajectory.csv`, summary tables, sequence FASTA,
noise audit and `summary.json` index every completed output. CA topology uses
Biotite CA geometry and long-range contact-map Jaccard, not full fold
classification. Undefined denominators remain NA with valid counts.

The comparison changes ligand mobility and adds mapped-atom coupling together.
It does not isolate their individual effects, establish affinity, or prove that
the two native per-track sequences bind both ligands. Coupling can distort
ligands at the differing substituent boundary, so geometry diagnostics should
be considered alongside backbone divergence.

## Validation and launch record

CPU job `2110294` passed 22 tests (nine original, seven substrate, six new
ligand tests) and both real input feature pipelines. The new tests cover explicit
mapping, rejection of invalid chemistry/selectors, shared noise and release,
protein RNG parity, sigma equivalence, observer isolation and public Hydra input.
Pilot array `2110351` was submitted for seed 101; submission does not imply
completed results. Project-level documentation records subsequent job IDs.

Use the successful-job sync helper with
`--output-subdir foundry/rfd3_system_early_cut JOBID`; also copy root manifests
whose filenames do not contain a job ID. Synthetic analysis validation is
clearly separated under `ligand_analysis_validation_JOBID/`.
