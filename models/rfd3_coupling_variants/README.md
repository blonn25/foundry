# Coupled-sampling variations

This bundle derives from `rfd3_system` without modifying it, `rfd3`, or
`rfd3_system_early_cut`. Thin named packages share one engine adapter and one
experimental sampler. No checkpoints are copied or trained.

## Original mechanism

The original engine builds A+B and A+C from one specification, resolves the
shared A residue/atom mapping, and excludes fixed motif atoms from the update
map. It runs the same checkpoint separately on each track. The representation
is Cartesian atomic coordinates, including generated protein Atom14 slots.
Both tracks share a coordinate frame, identical mapped A initialization, and
identical mapped A churn noise. B and C have independent noise.

In `SampleDiffusionWithSuperDiffSharedChainProxy`, the actual mixed quantity is
the EDM direction `delta = (X_noisy - X0) / sigma_hat`. With
`proxy_norm_weight=0.5`, the native solver returns exactly 0.5. Both A states
receive `X_noisy_A + step_scale*(sigma_next-sigma_hat)*(delta_AB+delta_AC)/2`.
Because the current A states and sigma are identical, this is algebraically
equivalent to averaging their predicted clean coordinates. It does not average
sequence logits, frames, torsions, or internal embeddings. It performs no
per-step Kabsch alignment. Partners are never averaged.

The native sequence head predicts token logits from the current noisy state;
it is a sibling of the coordinate output, not a function of the returned X0.
The sampler has no independent sequence diffusion clock or persistent shared
sequence latent. Native per-track sequences remain separate. Final shared
sequences in this benchmark come from tied multistate Caliby.

## Variations

| Package | Structural change | Main parameters |
|---|---|---|
| `rfd3_mean_5050` | Exact native equal-weight update | none |
| `rfd3_uncoupled` | Independent model directions | `shared_initialization`, `noise_correlation` |
| `rfd3_residual_consensus` | Retain disagreement in the native directions | `alpha` |
| `rfd3_late_uncoupling` | Native coupling for the first floor(f×N) updates | `release_fraction` |
| `rfd3_correlated_noise` | Independent directions, correlated native churn | `noise_correlation` |
| `rfd3_soft_guidance` | Correct X0 using invariant A geometry energy | `strength`, `guidance_schedule` |
| `rfd3_coarse_coupling` | Same correction on residue-block centroids | `block`, `strength`, `guidance_schedule` |
| `rfd3_sequence_coupling` | Native sequence-JS coordinate guidance | `strength`, `temperature` |
| `rfd3_population_diversity` | Native generation plus staged diverse selection | selection sizes 5/10/20 |

Schedules accept a scalar or `{start, end, kind: linear|cosine, until: 1}`.
Denoising progress runs from 0 at highest noise to 1 at the final update.
Noise-correlation schedules span the actual churn-active window. No new noise
term is invented. Every variant consumes the same native random draws; mixing
changes only mapped A noise and preserves its marginal variance.

Residual consensus mixes **directions**. For already-diverged noisy states,
mixing directions and mixing X0 are different algorithms; this experiment
deliberately retains the quantity mixed by the original architecture. Alpha=0
throughout reproduces the baseline; alpha=1 leaves each direction unchanged.

## Guidance and differentiation

Structural guidance uses the mean squared difference of intra-A Cα distances,
divided by `2*(10 Å)^2`. It is invariant under independent rigid motion of each
state. Coarse guidance uses centroids of contiguous 3/5/9-residue blocks and
does not pool across excluded motif gaps. The coordinate derivative is computed
in residue-translation coordinates and lifted to all movable atoms of that
residue. `X0_guided = X0 - eta*grad(E)` is passed to the unchanged EDM update.
This does not require a denoiser backward pass.

Sequence guidance computes JS over the 19 canonical Caliby-allowed amino acids
(Cys excluded) at designable shared A tokens. It differentiates through the
native sequence decoder and **every continuous recycling pass** with frozen
weights. Exact sequential recomputed VJPs avoid retaining both track graphs.
The resulting noisy-coordinate derivative is applied as
`X0_guided = X0 - lambda*sigma_hat^2*grad_Xnoisy(JS)` to all movable atoms of
each complex. This follows `score=(X0-Xnoisy)/sigma^2` and preserves the correct
descent sign under the negative EDM integration step. Fixed coordinates remain
untouched. Integer neighbor choices and distance bins retain their native
piecewise-constant behavior; there is no straight-through approximation.
In the present checkpoint, recycled X reaches the next cycle only through
bucketized distances and integer neighbor selection. Its exact cross-cycle
derivative is therefore zero. All cycles are recomputed with gradient tracking;
the final sequence prediction retains its continuous derivative with respect
to the current noisy input through the encoder and decoder. The GPU audit
checks this architectural fact explicitly and compares the overall derivative
against FP32 finite differences.

Guidance coefficients are frozen after three held-out calibration seeds.
Targets 0.03/0.1/0.3 refer to the median maximum state correction/native-update
norm ratio over progress 0.1–0.8. A single pairwise clipping factor limits the
correction in either state to 0.5 of its native update. Standalone guidance
profiles default to strength zero until a coefficient is explicitly supplied;
the prepared screen supplies the calibrated nonzero values.

Low JS does not guarantee one good sequence or physical binding. Native
mean-logit sequence probabilities are auxiliary diagnostics. All methods are
scored using the same four tied Caliby candidates and full local ESMFold2.
The native coupled sampler rejects CFG. No extra unconditional sequence branch
or product-of-experts experiment is introduced here.

## Execution

Use the project `pipeline/coupling_diversity_001` workflow and SLURM jobs.
The package directory `src` and existing `rfd3_system/src` must be on
`PYTHONPATH`. Named modules follow the original Hydra inference arguments:

```bash
python -m rfd3_residual_consensus.cli design \
  inputs=/project/inputs/example.json out_dir=/project/outputs/example \
  +experiment_file=/project/inputs/residual_parameters.json
```

The example parameter file can contain `{"alpha": 0.25}`. The existing system
specification must still identify the shared chain and the two partner sets.
For the full motif-free benchmark, the project manifest builder supplies those
arguments and records the exact source revision and all seeds.

Deterministic PyTorch reductions and deterministic cuBLAS are enabled by these
entry points to eliminate CUDA reduction-order divergence. Native modules and
global upstream source files are not patched. Original parity tests run with
the same deterministic setting for both implementations.

## Validation and outputs

CPU job 2126966 passed 14 sampler/operator tests and six evaluation contracts.
GPU job 2126955 passed all short-trajectory controls and exact final-coordinate
and RNG parity against the original for A90+B80 / A90+C100 with 199 updates.
The initial nondeterministic test attempt is retained as diagnostic provenance.

Every trajectory records mapped initialization/churn hashes, aligned and
common-frame A Cα RMSD, alpha/rho/strength, sequence entropy/JS, correction
ratios, and runtime. Debug runs additionally save full coordinate trajectories.
Only the first three replicates per condition enable these larger dumps.

The source-controlled `project/` directory contains recovery templates for the
top-level pipeline, jobs and documentation. Scientific outputs belong outside
the Foundry checkout under `outputs/pipeline/coupling_diversity_001/`.
