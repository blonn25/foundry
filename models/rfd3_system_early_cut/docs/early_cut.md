# Early release of shared-chain coupling

## Run

Select the new package with `models/rfd3_system_early_cut/src` on `PYTHONPATH`,
then use `python -m rfd3_system_early_cut.cli design`. Enable both
`coupling_mode=superdiff_shared_chain` and
`inference_sampler.kind=superdiff_shared_chain`, with the same inputs and
track specifications as the original model.

From the CoreHPC project root, in a GPU SLURM allocation:

```bash
scripts/run_rfd3_system_early_cut_a90_b80_c100.sh \
  inference_sampler.coupling_cut_fraction=0.5
```

Alternatively, replace that option with
`inference_sampler.coupling_cut_sigma=60`. Neither option is set by default:
that preserves full coupling. Only one may be non-null. Separate track
complexes are the default (`merged_output_policy=none`). Optional merged
views identify the source of chain A and are labeled as composite views.
They do not represent an additional independently sampled complex.

## Cutoff semantics

For an executed schedule with N updates, a fraction f performs floor(f*N)
coupled updates. Release occurs before the following update. Fraction 0
still uses identical shared initialization; fraction 1 never releases.
The default 200 schedule values produce 199 updates, so f=0.5 keeps 99
updates coupled and releases before update 100 (zero-based index 99).

A sigma threshold releases before the first update whose scheduled **pre-churn**
sigma is at or below the threshold in Angstrom. A threshold never reached
at an update's start leaves the trajectory fully coupled. The terminal
schedule value is not the start of another update. Partial diffusion resolves
both controls against the shortened, executed schedule.

The existing schedule uses

    sigma(t) = sigma_data * [s_max^(1/p) + t*(s_min^(1/p)-s_max^(1/p))]^p

with linearly spaced t. Therefore a fraction of updates is not a fraction
of starting sigma. `coupling.cutoff` records the requested controls, executed
update count, coupled update count, first independent update's zero-based index,
its pre-churn sigma and its actual churned `t_hat`. Use the recorded sigma to
construct an exactly matching sigma-controlled run. Churn uses the original
schedule rule based on the next sigma; it does not decide the release boundary.

## State, noise, and constraints

Before release, the original proxy solver supplies one mixed update for all
mapped movable shared atoms. After release, neither shared-coordinate copying
nor proxy mixing runs. Each track evolves with its own denoiser/context.
No coordinate reset or extra noise draw occurs at the boundary.

All original random draws remain in the same order: two initial track draws,
then two full-track churn draws and a shared churn draw per update, even when
churn is zero. The shared tensor is assigned through the original atom map
before and after release. Other movable atoms retain independent noise;
fixed atoms receive no initialization or churn noise. This is not a promise
that entire complexes remain identical: their partner contexts differ.
The existing coupled restrictions on realignment, origin jitter and CFG remain.
Sequence predictions are uncoupled, and recycling retains the checkpoint default.

## Full-trajectory diagnostics

### Optional explicit ligand atom coupling

Only this early-cut package supports `coupled_ligand_atom_pairs`. Its default
is `[]`, which leaves the previous sampler behavior and RNG stream unchanged.
Supply source CIF identifiers for each correspondence; atom names may differ:

```yaml
coupling_mode: superdiff_shared_chain
inference_sampler:
  kind: superdiff_shared_chain
  coupling_cut_fraction: 0.5
coupled_ligand_atom_pairs:
  - track_1: {chain: L, residue: 1, atom: C1}
    track_2: {chain: L, residue: 1, atom: CX}
  - track_1: {chain: L, residue: 1, atom: O1}
    track_2: {chain: L, residue: 1, atom: OX}
```

In the real 4MU inputs, the paired names are identical (C1–C12, O1–O4).
The JSON template under `experiments/ligand_codiffusion_sweep/config.json`
lists all 16 pairs explicitly. The Python interface accepts the same list
in `RFD3InferenceConfig(coupled_ligand_atom_pairs=...)`. Hydra also accepts
a list override on the command line. In each track specification, ligand
coordinates must be movable (e.g. `select_fixed_atoms: false`); ligand
chemistry must remain fixed. This changes coordinates, not molecular identity.

Each pair must resolve uniquely to a ligand atom. Source-to-prepared mapping
uses native atom/source annotations, so compacted output chain names do not
change selectors. Duplicate, missing, ambiguous, fixed-coordinate or
designable-chemistry selections are rejected. Elements, available formal
charges, and bonds between mapped atoms must agree. Different substituents
outside the selected subgraph are allowed and their boundary bonds recorded.
There is no automatic atom mapping or silent ambiguity resolution.

Mapped ligand atoms share initialization and receive the protein's scalar
mixing coefficient until the same fraction/sigma release boundary. The proxy
coefficient is still computed from the protein; adding ligand atoms does not
change its selection or normalization. Following release, ligand updates
are independent and no coordinate copying remains. Shared churn continues
for both protein and mapped ligand atoms. Unmapped ligand atoms always use
their native independent updates/noise. Fraction zero still shares
initialization and noise, but performs no coupled updates.

Ligand churn uses a separate seeded Torch generator, leaving all original
protein RNG draws in place. Thus enabling ligand coupling does not consume
extra values from the protein random stream. This is not a promise of
identical protein outputs: the ligand coordinates affect the denoiser.
The original coupled restrictions on CFG/realignment/jitter still apply.
No new rigid-body rotations or bond constraints are introduced. Atomwise
mixing can distort ligand geometry, particularly at substituent boundaries;
the experiment measures that geometry and retains all outcomes.

Output metadata adds `coupling.ligand_coupling` (selectors, resolved indices,
boundary bonds, policies) and `coupling.shared_ligand_state` (common-frame
mapped-atom RMSD at initialization and every completed update, plus actual
shared-noise differences). The experiment observer also stores all ligand
coordinates and protein-fitted ligand RMSD is calculated in post-processing.

Validate this option with `jobs/rfd3_ligand_codiffusion_tests.sbatch` and
`jobs/rfd3_ligand_analysis_tests.sbatch`. See the
[co-diffusion experiment](../experiments/ligand_codiffusion_sweep/README.md).

### Protein state measurements

Every output JSON contains `coupling.shared_chain_state`, even when
`dump_trajectories=False`. Arrays use `[state, paired_sample]` ordering:
initialization after shared-coordinate copying, followed by every completed
update, for N+1 measurements including the final state.

- `all_ca_rmsd`: corresponding C-alpha atoms across the whole shared chain,
  including fixed motifs mapped by source component.
- `movable_ca_rmsd`: C-alpha atoms of the formerly coupled movable residues.
  This is exactly zero throughout the coupled phase. Fixed track-specific
  motifs may give the whole-chain metric a nonzero baseline.
- `completed_updates`, `sigma`, atom counts, coordinate-state definition and
  alignment convention identify each measurement and its selection.

RMSD is sqrt(mean(sum((CA1-CA2)^2))), in Angstrom, in the common sampling
frame without fitting. It is measured from unscaled sampler states after
updates, not from the scaled noisy trajectory or model-denoised predictions.
The latter can already differ before release. Metric selection is independent
of `kappa_atom_subset`. Diagnostic arithmetic casts views to FP32 without
changing sampler states or consuming randomness.

`coupling.diagnostics` adds one `coupling_active` flag per update,
`pre_churn_sigma`, and per-sample `shared_noise_max_abs_difference`. Noise
agreement compares the actual injected tensors, not rounded coordinate
differences. Proxy diagnostics are JSON null after release.

```bash
scripts/esm_exec.sh python \
  software/foundry/models/rfd3_system_early_cut/scripts/plot_coupling_diagnostics.py \
  outputs/foundry/rfd3_system_early_cut/RUN --recursive
```

The PNG includes both full-trajectory C-alpha curves, noise agreement, kappa,
and the release boundary. The companion CSV contains per-state measurements.
Outputs stay outside the source clone.

## Validation and restore

Submit `jobs/rfd3_system_early_cut_tests.sbatch` on CPU, then
`jobs/rfd3_system_early_cut_smoke.sbatch` on GPU after the tests pass.
CPU tests use unittest in the production container, including exact original
sampler parity, release boundaries, partial schedules, fixed SEP/SER mapping,
unequal track sizes, shared noise, actual independent updates and C-alpha metrics.

The GPU job runs five full 199-update matched-seed A90+B80/A90+C100 cases:
original, new model with release disabled, immediate release, fraction 0.5,
and the equivalent sigma threshold. It enables deterministic algorithms in
both packages for validation only, compares internal final coordinates exactly,
checks diagnostics without trajectory dumping, and generates PNG/CSV plots.
It records `validation_summary.json` and a completion marker. No production
campaign is launched or modified.

Restore the Foundry fork on `system_design_v0` and the existing Foundry image;
no additional dependencies or checkpoints are required. Project helper/job
copies are preserved under `examples/project/`. CPU validation additionally
places the original package on PYTHONPATH for the baseline comparison; normal
new-model inference does not import the original coupled implementation.

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
