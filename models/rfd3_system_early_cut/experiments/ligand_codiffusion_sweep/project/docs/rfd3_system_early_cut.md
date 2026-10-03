# RFD3 System Early Cut

`software/foundry/models/rfd3_system_early_cut/` is an isolated duplicate of
the original `rfd3_system` with configurable release of shared-chain structural
coupling. The original model is unchanged. Corresponding formerly coupled
atoms continue receiving identical churn noise after release.

See the [model guide](../software/foundry/models/rfd3_system_early_cut/docs/early_cut.md)
for cutoff semantics, the fraction-to-sigma relationship, diagnostics, and
the separate-track output policy.

The model also accepts explicit `coupled_ligand_atom_pairs` for co-diffusing
ligands. Paired atoms use the protein's release boundary and mixing coefficient,
then evolve independently while shared noise continues. See the ligand section
below and the model guide for the source-CIF selector schema.

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

## Completed substrate sweep (September 30, 2026)

All generation and analysis jobs above completed successfully. Final analysis
`2076317` contains all 210 pairs / 420 structures and eight dense/sparse PNG/PDF
figure sets. Each pair retained 200 CA states. The initialization and all churn
increments matched across tracks and release conditions within each seed, and
internal fixed-ligand coordinate drift was exactly zero. No geometry-based
selection was applied before averaging.

Representative final results (mean ± sample SD over ten paired seeds):

| Coupled fraction | Common-frame CA RMSD (Å) | Aligned CA RMSD (Å) | Mean long-range contact Jaccard |
| --- | ---: | ---: | ---: |
| 0.00 | 5.98 ± 3.59 | 5.44 ± 3.37 | 0.250 |
| 0.50 | 5.53 ± 3.27 | 4.93 ± 2.92 | 0.230 |
| 0.60 | 2.09 ± 0.73 | 1.97 ± 0.74 | 0.538 |
| 0.70 | 0.330 ± 0.127 | 0.289 ± 0.098 | 0.920 |
| 0.80 | 0.0319 ± 0.0143 | 0.0314 ± 0.0140 | 0.991 |
| 1.00 | 0 | 0 (numerical precision) | 1.000 |

Releasing within roughly the first half gave similar, substantial final
backbone divergence. The sharp reduction appeared at fractions 0.55–0.70
(release sigma approximately 33.84–5.17 Å); release at 0.80 left the two
backbones nearly identical. Raw and aligned RMSDs are similar, so the early
release differences mainly reflect backbone shape rather than rigid-body pose.
These are descriptive results from ten seeds, not an optimized release-point
estimate or a binding-quality comparison.

The structures are predominantly helical. Mean per-residue helix/strand/coil
agreement was 77.7% without coupling, 92.1% at fraction 0.60, 98.9% at 0.70,
and 99.9% at 0.80. Long-range contacts provide a stricter measure than secondary
labels alone: their mean Jaccard increased from 0.250 without coupling to 0.920
at 0.70. Strand Jaccard excludes empty unions and is based on only three to five
pairs per condition; its valid counts are retained in the CSV.

All 420 structures have ligand contacts and complete native amino-acid readouts.
Eleven have one CA-spacing flag each. At least one protein–ligand heavy-atom
clash (vdW overlap >0.4 Å) occurs in 353/420 structures; these were retained, as
planned. Sequences were not tied, and folding, affinity, and dual-substrate
binding have not been validated. No relaxation was applied to this sweep.

Figures and data:

- [Final RMSD, all 21 conditions](../outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/analysis_2076317/final_rmsd_dense.png)
- [Trajectory versus fraction, six conditions](../outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/analysis_2076317/trajectory_fraction_sparse.png)
- [Trajectory versus sigma, six conditions](../outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/analysis_2076317/trajectory_sigma_sparse.png)
- [Topology, all 21 conditions](../outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/analysis_2076317/topology_dense.png)
- [Final numerical summary](../outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/analysis_2076317/final_summary.csv)
- [Structure index and geometry flags](../outputs/foundry/rfd3_system_early_cut/substrate_sweep_001/analysis_2076317/designs.csv)

## Explicit ligand coupling and the co-diffusion sweep (October 2, 2026)

Implementation is confined to the early-cut model. The public inference input
`coupled_ligand_atom_pairs` is a list of explicit source-CIF selectors:

```yaml
coupled_ligand_atom_pairs:
  - track_1: {chain: L, residue: 1, atom: C1}
    track_2: {chain: L, residue: 1, atom: C1}
```

The default empty list retains the previous behavior. Selected ligand atoms
must have movable coordinates and fixed chemistry. Mapping validates identity,
elements, available charges, and bonds within the selected subgraph; external
substituent differences are permitted. Before release, protein and mapped
ligand updates use the same mixing coefficient; after release, coordinate
updates are independent. Initialization and every mapped churn increment
remain shared. Track 1's existing ligand draw is reused without extra RNG draws
or stream resets. Unmapped ligand atoms diffuse independently throughout.
The proxy solve itself still uses only the selected protein atoms.

The [experiment guide and explicit 16-atom input map](../software/foundry/models/rfd3_system_early_cut/experiments/ligand_codiffusion_sweep/README.md)
repeat the fixed-ligand sweep with co-diffused 4MU-Ac and 4MU-Bu: 120-residue
proteins, 21 fractions, ten seeds 101–110, 210 pairs / 420 structures. All 16
shared heavy atoms are mapped; Bu's two additional carbons remain independent.
The existing input conformers provide reference chemistry, rather than fixed
ligand poses; this is full diffusion, without a hydrolase protein template.
The same 199 updates, two recycles, equal mixing and native churn are used.
All completed pairs are retained, including geometry outliers.

Runtime audits require protein initialization and all 199 churn hashes to match
both across fractions and against the completed fixed-ligand baseline. They also
check the exact coupled protein/ligand prefix and ligand noise correspondence.
All runtime audits passed for every seed and fraction; see completed results below.

CPU preflight `2110294` passed the initial 22 tests and actual ligand feature
pipelines. Analysis validation `2110476` passed two additional geometry tests,
the seven existing substrate tests, a full synthetic 21-condition collection,
and 19 PNG/PDF plotting paths. Its test artifacts are synced to Wynton under
`outputs/foundry/rfd3_system_early_cut/ligand_analysis_validation_2110476/`;
they are synthetic validation, not new design results.

The final implementation (`dba1a69`) adds a repeated-batch noise regression test.
CPU job `2110505` passed all 23 tests and both real feature pipelines in
64 seconds. Together with the two analysis tests, 25 unique tests passed.
The submission record is `logs/rfd3_ligand_codiffusion_submission_2110506.txt`:

| Stage | SLURM ID | Role |
| --- | --- | --- |
| Final CPU preflight | 2110505 | 23 sampler/metric tests and real input features |
| Seed-101 pilot | 2110506 | 21 complete paired trajectories and noise audits |
| Pilot analysis | 2110507 | All metrics/plots; requires pilot and validated analysis |
| Seeds 102–110 | 2110508 | Nine independent GPU tasks; requires successful pilot analysis |
| Full analysis | 2110509 | All 210 pairs and paired fixed-baseline comparisons |

All listed jobs completed successfully, including all 420 structures and final analysis. The earlier
pilot `2110351` was canceled while queued before producing data to include the
final native-noise reuse refinement. Do not submit this sweep again. Recover
individual failed stages only after inspecting their logs and immutable config.

Results go to `outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/`.
Analysis records CA dynamics/topology, ligand common-frame and fitted RMSDs,
ligand RMSD after fitting only protein CA, conformer and bond-length distortion,
contacts/clashes, and paired differences from the original fixed-ligand sweep.
Dense and six-condition sparse plots retain mean/sample SD and valid counts.
The comparison changes both ligand mobility and mapped-atom coupling; it does
not isolate those effects or measure affinity.

After each successful stage, sync its outputs with:

```bash
scripts/sync_corehpc_job_outputs.sh --output-subdir foundry/rfd3_system_early_cut JOBID
```

Root-level `resolved_config.json` and `manifest.{json,csv}` have also been synced.
No new software, checkpoints or container images are required. Versioned
job/launcher copies live in the experiment's `project/` bundle; preserve both
this sweep and the fixed-ligand baseline when archiving scientific data.


## Completed co-diffusion results

All ten GPU seed tasks completed successfully (about 18–19 minutes per seed),
followed by final analysis `2110509` in 46 seconds. All 21 fractions and ten
seeds are present: **210 pairs, 420 native structures, 42,000 paired states**.
All structures and 19 PNG/PDF plot pairs are synced to Wynton, together with
the immutable root manifests and configuration. Every indexed CIF exists locally.

Protein initialization and all applied churn increments matched exactly between
tracks, across release conditions, and against the fixed-ligand experiment.
Mapped ligand initialization/churn also matched, and both protein and mapped
ligand coordinates remained identical throughout the coupled prefix. The fully
coupled endpoint has exactly zero common-frame CA and mapped-ligand RMSD.

Selected endpoint measurements follow. RMSDs are mean ± sample SD over ten
matched seeds; other fractions and full valid-count/SD columns are in the CSVs.
CA RMSD measures corresponding atoms between the two tracks after proper rigid
alignment. Ligand CA-fit RMSD uses that protein alignment without fitting the
ligand again. Fraction means the fraction of updates kept coupled, not the
fraction of initial sigma.

| Fraction coupled | Release sigma (Å) | Co-diffused aligned CA RMSD (Å) | Fixed aligned CA RMSD (Å) | CA-fit ligand RMSD (Å) | Secondary-label agreement | Contact Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 2560.000 | 8.983 ± 5.005 | 5.445 ± 3.372 | 11.581 ± 6.156 | 0.531 | 0.183 |
| 0.20 | 724.875 | 8.823 ± 4.856 | 5.521 ± 3.516 | 11.701 ± 5.077 | 0.610 | 0.156 |
| 0.40 | 148.614 | 7.764 ± 4.838 | 5.671 ± 3.540 | 10.915 ± 6.690 | 0.658 | 0.232 |
| 0.50 | 57.421 | 6.904 ± 5.278 | 4.929 ± 2.923 | 9.220 ± 5.111 | 0.651 | 0.352 |
| 0.55 | 33.840 | 2.838 ± 2.207 | 2.903 ± 2.495 | 6.670 ± 5.003 | 0.853 | 0.541 |
| 0.60 | 19.100 | 1.808 ± 1.202 | 1.971 ± 0.744 | 4.868 ± 3.039 | 0.923 | 0.661 |
| 0.65 | 10.245 | 0.936 ± 0.609 | 1.191 ± 0.826 | 3.761 ± 2.491 | 0.948 | 0.817 |
| 0.70 | 5.170 | 0.272 ± 0.167 | 0.289 ± 0.098 | 3.194 ± 1.867 | 0.983 | 0.949 |
| 0.75 | 2.423 | 0.100 ± 0.033 | 0.150 ± 0.056 | 1.588 ± 0.933 | 0.993 | 0.968 |
| 0.80 | 1.036 | 0.026 ± 0.009 | 0.031 ± 0.014 | 0.257 ± 0.183 | 1.000 | 0.991 |
| 1.00 | never released | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 | 1.000 |

With release at 0–50% of updates, the co-diffused experiment has larger mean
backbone divergence than the fixed-ligand baseline, with substantial seed-to-seed
variation. The main transition remains around 55–70% coupling (release sigma
about 34–5 Å). At 70%, mean aligned CA RMSD is 0.272 Å versus 0.289 Å with fixed
ligands. Secondary labels agree at 98.3% of positions and the mean long-range
CA contact Jaccard is 0.949. At 80%, CA RMSD is 0.026 Å and labels agree at 100%.
These CA geometry/contact metrics are proxies for shared topology.

Ligand poses remain more flexible after release than the backbones: at 70%,
mapped ligand RMSD after the protein fit averages 3.194 Å, while ligand-only
fitting reduces it to 0.499 Å. Thus much of the ligand difference is pose,
not just internal deformation. The final whole-ligand bond-length RMSE versus
the input conformer averages 0.042 Å (Ac) and 0.046 Å (Bu); the largest individual
bond-length deviation is 0.181 Å. These compare with the supplied conformer,
not an ideal geometry or a full chemical-quality validation.

Geometry flags were retained without filtering: 131/420 structures have at
least one protein–ligand heavy-atom clash (vdW overlap >0.4 Å), compared with
353/420 in the fixed-ligand baseline. Seven have a CA-spacing flag, versus eleven
previously. Every design has at least one protein residue within 4 Å of the
ligand (mean about 7.5/7.6 contacted residues for Ac/Bu). Fewer clashes alone do
not establish binding quality. This comparison changes ligand mobility and
mapped-atom coupling together, and native per-track sequences remain untied.

Plots and data (PNG companions also have exportable PDFs):

- [Fixed versus co-diffused endpoints and topology](../outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/analysis_2110509/fixed_vs_codiffused_dense.png)
- [CA trajectories versus denoising fraction, six conditions](../outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/analysis_2110509/trajectory_fraction_sparse.png)
- [CA trajectories versus sigma, six conditions](../outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/analysis_2110509/trajectory_sigma_sparse.png)
- [Mapped ligand dynamics, six conditions](../outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/analysis_2110509/ligand_trajectory_fraction_sparse.png)
- [Final summary, all 21 fractions](../outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/analysis_2110509/final_summary.csv)
- [All 420 structures and geometry flags](../outputs/foundry/rfd3_system_early_cut/ligand_codiffusion_sweep_001/analysis_2110509/designs.csv)

No cleanup is required. The approximately 302 MiB experiment and lightweight
validation artifacts are retained; no failed scientific attempts were produced.
