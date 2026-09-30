# RFD3 System Early Cut

`software/foundry/models/rfd3_system_early_cut/` is an isolated duplicate of
the original `rfd3_system` with configurable release of shared-chain structural
coupling. The original model is unchanged. Corresponding formerly coupled
atoms continue receiving identical churn noise after release.

See the [model guide](../software/foundry/models/rfd3_system_early_cut/docs/early_cut.md)
for cutoff semantics, the fraction-to-sigma relationship, diagnostics, and
the separate-track output policy.

From a CoreHPC GPU SLURM allocation:

```bash
scripts/run_rfd3_system_early_cut_a90_b80_c100.sh \
  inference_sampler.coupling_cut_fraction=0.5
```

Alternatively use `inference_sampler.coupling_cut_sigma=60`; only one control
may be non-null. Both default to null, retaining full coupling. The default
schedule has 199 updates; fraction 0.5 releases before update 100.

Every paired sample records shared-chain C-alpha state RMSD at initialization
and after every update, in the common frame without alignment. Both whole-chain
and movable-residue metrics are retained, along with injected-noise agreement.
The accompanying plotter writes PNG and CSV diagnostics.

Validation entry points:

```bash
sbatch jobs/rfd3_system_early_cut_tests.sbatch
# After the CPU tests pass:
sbatch jobs/rfd3_system_early_cut_smoke.sbatch
```

The GPU job uses five full-schedule matched-seed cases and writes results under
`outputs/foundry/rfd3_system_early_cut/smoke_JOBID/`. This checks implementation
behavior; it does not establish design quality or an optimal release point.

Restore through the existing Foundry fork and container. The new model needs
no additional checkpoint or environment. Versioned copies of its project
launch helper and jobs are also included in the model's `examples/project/`.

## Completed validation (September 29, 2026)

CPU job `2071629` passed all nine tests, including the public Hydra controls
and checkpoint namespace mapping (13 seconds; 1.21 GiB peak host RSS).
The preceding eight-test run `2071624` also passed.

GPU job `2071625` completed on an H100 NVL in 5m38s with 13.89 GiB peak
host RSS. It generated ten track complexes from five paired runs, each using
199 updates with the same seed. All four early-cut cases retained 200 C-alpha
state measurements with full trajectory dumping disabled.

| Release setting | Coupled updates | Final shared-chain state CA RMSD (Å) |
| --- | ---: | ---: |
| Disabled | 199 | 0.000000 |
| Immediate (`fraction=0`) | 0 | 21.755810 |
| `fraction=0.5` | 99 | 8.860096 |
| `sigma=57.42110824584961` | 99 | 8.860096 |

These are unaligned, common-frame state RMSDs from one implementation smoke
seed, not design-quality estimates. Whole-chain and movable-CA selections
coincide in this de novo example. Fixed SEP/SER motif mapping is covered by
the CPU tests.

Both tracks' internal final coordinates matched exactly between the original
model and release-disabled copy, and between the fraction and equivalent
sigma cases (maximum absolute difference 0). Shared injected-noise differences
were exactly 0 at every update. Movable-CA RMSD stayed exactly 0 through the
coupled phase and became nonzero after release. The midpoint release used
pre-churn sigma 57.421108 Å and churned `t_hat` 91.873772 Å.

Results, JSON diagnostics, PNG plots, and CSV traces are under
`outputs/foundry/rfd3_system_early_cut/smoke_2071625/` on CoreHPC and Wynton.
The original `models/rfd3_system/` tree remains identical to the baseline
Foundry revision `5e51cf0`. Validation used the `0b225db` implementation;
`d5b9a7d` adds the public-configuration test and strict plotting failure status.

[Validation summary](../outputs/foundry/rfd3_system_early_cut/smoke_2071625/validation_summary.json)
and [halfway-release plot](../outputs/foundry/rfd3_system_early_cut/smoke_2071625/fraction/validation_0_coupling_early_cut_state.png).

## 4MU substrate release sweep

The implemented [experiment and restore bundle](../software/foundry/models/rfd3_system_early_cut/experiments/substrate_sweep/README.md)
uses 120-residue de novo proteins with fixed 4MU-Ac and 4MU-Bu poses, equal mixing,
21 coupled fractions, and seeds 101–110. This produces 210 pairs / 420 structures.
Fraction 0 is uncoupled; fraction 1 is fully coupled. The same protein initialization
and churn increments are verified within each pair and across all fractions for
each seed. It does not use a hydrolase protein template or tie final sequences.

CPU preflight `2073953` passed 14 sampler/metric tests and both complete input
pipelines in 51 seconds. Checkpoint preprocessing peaked at 7.80 GiB host RSS,
so the reusable CPU preflight now requests 12 GiB. Pilot array `2074201` runs seed
101 across all 21 fractions before the remaining seeds are allowed to run.

Use `jobs/rfd3_substrate_sweep_tests.sbatch`, `jobs/rfd3_substrate_sweep.sbatch`,
and `jobs/rfd3_substrate_sweep_analyze.sbatch`. The analysis saves actual per-step
CA RMSDs with and without alignment, final geometry flags, CA-based secondary
structure and contact-map comparisons, and eight dense/sparse PNG/PDF figure
pairs. No structures are selected out for poor geometry. Scientific results
belong under `outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/`.

Analysis validation job `2075927` passed all seven experiment tests, including
CIF/fixed-pose validation and missing-denominator statistics, and rendered all
eight PNG/PDF figure pairs from explicitly synthetic fixtures. Together with
the original nine tests, this covers 16 distinct unit tests. The synthetic
figures are test artifacts, not experimental results. Job `2075858` failed on
a test-fixture array type; that fixture was corrected, and its logs are retained.
Successful validation logs are synced to Wynton.

The following SLURM chain was submitted on September 30, 2026. At submission
the pilot was pending compatible GPU resources; no sweep structures or
scientific plots had been generated. The scheduler estimated 05:42 Pacific,
which may change.

| Stage | Job | Success dependency |
| --- | --- | --- |
| Seed 101, all 21 conditions | `2074201` | CPU preflight already passed |
| Pilot metrics and figures | `2076315` | Pilot generation |
| Seeds 102–110, 21 conditions each | `2076316` | Pilot generation **and analysis** |
| Full metrics and figures | `2076317` | Remaining seeds and pilot analysis |

Launch details are retained in `logs/rfd3_substrate_sweep_submission_2074201.txt`.
The reusable submitter uses native SLURM dependencies, with no polling controller.
Any failed upstream job invalidates its dependent stages; failed attempts remain
available for diagnosis. Final results will be in `analysis_2076317/` within the
sweep root. Pilot plots in `analysis_2076315/` have only one seed and must not be
interpreted as ten-replicate averages.

When jobs complete successfully, sync their outputs and logs:

```bash
scripts/sync_corehpc_job_outputs.sh --output-subdir foundry/rfd3_system_early_cut \
  2074201 2076315 2076316 2076317
rsync -a chpc-login:/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/manifest.json \
  outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/
rsync -a chpc-login:/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/manifest.csv \
  outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/
rsync -a chpc-login:/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/resolved_config.json \
  outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/
```
