# Coupling diversity experiment

Primary system: motif-free co-generation of A90+B80 and A90+C100. This isolates
coupling behavior before the separate SEP/SER transfer experiment. Native
sequences are retained, then four tied Caliby candidates are designed on both
actual A conformations. Full local ESMFold2 predicts both states for each
candidate, with no FastRelax.

Success requires **the same candidate** to have whole-complex Cα RMSD <2 Å in
both A+B and A+C after a single joint complex alignment. Confidence, contacts,
clashes and secondary structure are diagnostics, not additional primary gates.
Missing/invalid results stop complete analysis instead of becoming numerical
failures or silently disappearing. Candidate and parent success rates are
reported separately.

The frozen screening matrix contains 37 conditions × 50 paired seeds = 1,850
parents, 7,400 shared-sequence candidates, and 14,800 ESMFold2 predictions.
It includes shared- and independent-noise uncoupled references, the exact
50/50 reference, five release points, ten residual schedules, four noise
schedules, six structural guidance settings, three coarse scales, four
sequence-guidance settings, and two sampler controls. See `conditions.py` in
the Foundry variation bundle for the authoritative enumeration.

Sampling uses 200 sigma values / 199 updates, step scale 3, gamma 0.2,
gamma_min 1, noise_scale 1.003, p=7, two recycling passes, non-loopy conditioning,
and no CFG, rigid augmentation, or origin jitter. The two sampler controls
change only step_scale to 1.5 or disable non-loopy conditioning.

## Run on CoreHPC

All commands start in the project mirror, after `module load CBI`. Source edits
and Git commits happen on Wynton; pull the reviewed revision on CoreHPC before
preparing immutable manifests. Do not change that checkout while a stage is
active. The stages reject a source revision differing from their manifest.

```bash
sbatch jobs/rfd3_coupling_variants_cpu.sbatch
sbatch jobs/rfd3_coupling_variants_gpu.sbatch

python pipeline/coupling_diversity_001/prepare.py \
  outputs/pipeline/coupling_diversity_001/smoke_001 --stage smoke
python pipeline/coupling_diversity_001/submit.py \
  outputs/pipeline/coupling_diversity_001/smoke_001

python pipeline/coupling_diversity_001/prepare.py \
  outputs/pipeline/coupling_diversity_001/calibration_001 --stage calibration
python pipeline/coupling_diversity_001/submit.py \
  outputs/pipeline/coupling_diversity_001/calibration_001

# Only after checkpoint validation, the full smoke workflow, and calibration:
python pipeline/coupling_diversity_001/prepare.py \
  outputs/pipeline/coupling_diversity_001/screen_001 --stage screen \
  --calibration outputs/pipeline/coupling_diversity_001/calibration_001/calibration.json
python pipeline/coupling_diversity_001/submit.py \
  outputs/pipeline/coupling_diversity_001/screen_001 \
  --validation outputs/pipeline/coupling_diversity_001/validation_2126955 \
  --smoke outputs/pipeline/coupling_diversity_001/smoke_001
```

Submission writes a journal immediately after each accepted SLURM job and
refuses duplicate submissions. Matching arrays connect RFD generation → CPU
Caliby → GPU ESMFold2 with `aftercorr`; failed dependencies invalidate only
their corresponding descendants. Analysis requires all folds to complete.
Use explicit `--indices` for inspected sparse recoveries. Completed stages
and individual completed folds are retained; incomplete generation outputs
require an explicitly preserved/reviewed retry directory before rerunning.

## Analysis and decisions

`collect.py` writes candidate/parent/condition CSV and JSON files plus per-residue
displacements. `diversity.py` runs exhaustive Foldseek TM-align per condition,
requires every directed comparison, and retains coverage. Clustering uses
complete linkage and TM ≥0.5 with ≥80% alignment coverage, plus 0.6/0.7
sensitivity. Outputs distinguish A+B and A+C, all parents and dual passers,
within-pair flexibility and between-parent diversity. Rarefaction, effective
cluster counts, and population farthest-point selection at 5/10/20 successful
parents address sample-count effects. Population selection is compared with
random selection from exactly the same successful pool and budget.

`plot.py` writes PNG/PDF compatibility–diversity plots, state success intervals,
runtime comparisons, trajectory-versus-progress/sigma plots, representative
guidance/entropy diagnostics and `analysis/report.md`.

After inspecting the complete screen, run `select.py` to propose at most three
distinct families with observed dual success at least the baseline and greater
successful-pool diversity. If baseline diversity is undefined, it says so and
does not claim improvement over an unmeasurable baseline. Confirm shortlisted
families and the baseline on 200 fresh paired seeds using `--stage confirm`
and `--selected .../analysis/selected.json`.

Combination model names are deliberately reserved until the individual results
support them. At most two combinations and a baseline/best-individual control
receive 200 fresh pairs. The later 50-specification Production-005 2B05 SEP/SER
transfer needs motif-bearing inputs and an independently reported strict motif
subset; `prepare.py` currently refuses `--stage transfer` so the primary
motif-free inputs cannot be mistaken for that experiment. These downstream
steps are conditional work after reviewing actual screening results.

## Recovery and provenance

The Foundry bundle is `software/foundry/models/rfd3_coupling_variants/`. Its
`project/` templates recover this pipeline and the SLURM jobs. No new weights
or environments are required. Existing pinned Foundry, Caliby, ESM and Foldseek
installations are reused. Preserve immutable manifests and useful results
separately from rebuilding source; a Git checkout does not archive structures.

Successful validation 2126955 is synced to Wynton. Initial smoke jobs are
2126967/2126968/2126969/2126970; calibration jobs are 2126976/2126977. Completion
markers and SLURM accounting, not the presence of a submission file, establish
which work has finished.
