# Shared-Chain SuperDiff Prototype Notes

This directory is a research copy of `models/rfd3` for investigating coupled
RFdiffusion3 inference with a shared chain A across two contexts:

- Track 1: A+B
- Track 2: A+C

The original `models/rfd3` directory must remain untouched.

## Current Status

Approximate shared-chain coupling is implemented behind:

```bash
coupling_mode=superdiff_shared_chain inference_sampler.kind=superdiff_shared_chain
```

Exact SuperDiff AND coupling is not implemented. The exactness gate found that
the current RFD3 inference stack does not expose the same objects assumed by
the SuperDiff AND density-control derivation.

SuperDiff assumes access to model scores, `grad_x log q_t^i(x)`, and samples
with an SDE whose density change can be tracked with the Ito density estimator.
The public SuperDiff protein implementation uses explicit `trans_score` and
`rot_score` tensors from score-based protein diffusion models when computing
the AND weights.

RFD3 instead exposes an EDM-style denoiser and sampler:

- The diffusion module predicts denoised coordinates, `X_denoised_L`.
- The sampler computes `delta_L = (X_noisy_L - X_denoised_L) / t_hat`.
- The default update follows the AF3/EDM Karras-style update
  `X_L = X_noisy_L + step_scale * d_t * delta_L`.
- The sampler can include churn-like stochastic noising through `gamma`,
  `noise_scale`, fixed-coordinate masks, and optional realignment.

Under an ideal VE/EDM denoising interpretation, RFD3's denoised coordinate
prediction suggests a score-like relation:

```text
score(x, sigma) ~= (D(x, sigma) - x) / sigma^2
delta_L          ~= -sigma * score(x, sigma)
```

That relation is useful, but it is not enough to claim exact SuperDiff density
control for the existing sampler. Exact density control would still require a
derivation that maps RFD3's discrete EDM update, schedule, stochastic noising,
masks, and any coordinate transforms onto the SuperDiff SDE and its density
increment equations.

## Exact Implementation Gate

Do not describe the current implementation as exact SuperDiff unless this
derivation is completed and documented:

1. Define the forward noising SDE for the diffused atom coordinates used by RFD3.
2. Derive the exact score from RFD3's denoiser output, including sign, scale,
   time variable, and mask handling.
3. Derive the reverse SDE step used for shared-chain sampling.
4. Show how the SuperDiff AND linear system is evaluated for chain A using the
   RFD3-derived score and vector field.
5. State which RFD3 sampler features are compatible with the derivation.
   Features such as classifier-free guidance, symmetry sampling, realignment,
   and EDM churn should remain disabled unless explicitly handled.

If any item cannot be satisfied, exact shared-chain coupling remains blocked.
The implemented mode below is only an approximation.

## Approximate Coupling Implementation

The implemented approximation uses RFD3's denoiser-derived `delta_L` as a
score-like update proxy for the shared chain A:

1. Split one ABC input into A+B and A+C track views.
2. Maintain a single shared coordinate state for chain A.
3. At every denoising step, copy A into both tracks, draw identical A noise for
   both tracks, and draw independent B/C noise.
4. Run the RFD3 denoiser once per track.
5. Extract A updates, `delta_A_ab` and `delta_A_ac`.
6. Estimate adaptive weights from the two proxy vectors with a stabilized
   SuperDiff-inspired two-track solve.
7. Clamp weights to a configured range and fall back to equal weighting when
   the proxy system is degenerate.
8. Update A with the weighted proxy update, while updating B and C with their
   normal per-track updates.
9. Relabel track 2's non-shared partner chains back to the user-facing global
   chain IDs before output formatting.
10. Write A+B, A+C, merged A+B+C, trajectory outputs, and metadata containing
    all weights, residuals, norms, and approximation warnings. Kappa and
    proxy-residual PNG plots can be generated after inference from the JSON
    diagnostics.

The proxy solve lives in `src/rfd3_system/system/proxy.py`. Given two shared-A
update proxies `delta_1` and `delta_2`, it defines:

```text
delta_mix = kappa * delta_1 + (1-kappa) * delta_2
proxy_i(delta_mix) = <delta_mix, delta_i> - norm_weight * ||delta_i||^2
```

It chooses `kappa` so that:

```text
proxy_1(delta_mix) ~= proxy_2(delta_mix)
```

Then it clamps `kappa` to `[proxy_kappa_min, proxy_kappa_max]` and falls back
to `0.5` when the system is degenerate. This is only a stabilized local proxy
for equalizing condition-specific update pressure; it is not the SuperDiff
Itô-density estimator.

The sampler implementation lives in
`src/rfd3_system/model/inference_sampler.py` as
`SampleDiffusionWithSuperDiffSharedChainProxy`. It rejects classifier-free
guidance, symmetry, motif realignment, and `s_jitter_origin` because those
features would need separate coupled derivations.

The engine implementation lives in `src/rfd3_system/engine.py`. It:

- uses `shared_chain_id`, `complex_1_partners`, and `complex_2_partners` to
  split one source atom array into two track-specific atom arrays;
- accepts `track_1_specification` and `track_2_specification` overrides so each
  track can define a normal RFD3 contig/selection against its chain-split view;
- accepts explicit track-specific A+B and A+C input structures for joint motif
  scaffolding;
- validates that non-fixed shared-chain atom ordering and initial shared
  coordinates match between tracks;
- excludes fixed shared-chain motif residues from the kappa solve and shared
  coordinate update;
- writes track 1, track 2, and optional merged outputs;
- leaves sequence logits uncoupled, with merged A+B+C controlled by
  `merged_output_policy`.

Output metadata records `superdiff_exact: false`, the coupling configuration,
the sequence policy, proxy weights, residuals, norms, and degenerate-step flags.
The physical RFD3 noise scale `t_hat` remains available in JSON diagnostics.
After inference, `scripts/plot_coupling_diagnostics.py` can be run with a
Matplotlib-capable environment to create PNG kappa and proxy-residual plots
from those JSON diagnostics. The plotter can use track or merged JSON files and
writes one batch-level plot set by default; pass `--model-index N` for a
one-off plot set for a single generated model. The kappa PNG x-axis uses
normalized `t` values, ordered from 0 on the noisy left side to 1 at the final
denoised end on the right.

For the derivation of `kappa_raw`, the proxy residual, and the relationship
between RFD3 denoiser deltas and EDM-style scores, see
[`shared_chain_coupling_math.md`](./shared_chain_coupling_math.md).

## Implementation Map

The implementation is intentionally localized to this `rfd3_system` research
copy:

- `src/rfd3_system/system/proxy.py` contains the approximate two-track proxy
  solve and its diagnostics dataclass.
- `src/rfd3_system/system/chains.py` contains chain splitting, shared-chain
  validation, fixed-motif-aware shared atom mapping, track-2 partner relabeling,
  and merged-output helpers.
- `src/rfd3_system/model/inference_sampler.py` registers
  `SampleDiffusionWithSuperDiffSharedChainProxy` behind
  `inference_sampler.kind=superdiff_shared_chain`.
- `src/rfd3_system/model/RFD3.py` adds `forward_coupled`, an inference-only
  wrapper that initializes two track views and delegates rollout to the coupled
  sampler.
- `src/rfd3_system/engine.py` adds `coupling_mode=superdiff_shared_chain`,
  builds A+B and A+C track specifications from either a single ABC input or
  explicit track-specific motif inputs, validates shared-chain consistency, and
  formats track and merged outputs.
- `configs/inference_engine/rfdiffusion3.yaml` exposes the coupling mode,
  shared-chain ID, partner-chain IDs, and track-specific specification
  overrides.
- `docs/examples/superdiff_shared_chain_proxy.yaml` shows the intended CLI
  override shape.
- `docs/joint_motif_scaffolding.md` documents the track-specific motif
  scaffolding workflow and selection patterns.
- `tests/test_superdiff_proxy.py` covers the standalone proxy-weight solver.

## Key Design Decisions

The original `models/rfd3` package is not modified.  This keeps upstream RFD3
available as a reference implementation and prevents accidental changes to
ordinary Foundry behavior while the coupled sampler is still experimental.

The mode is opt-in through both `coupling_mode=superdiff_shared_chain` and
`inference_sampler.kind=superdiff_shared_chain`.  Requiring both switches makes
it hard to accidentally run the proxy sampler as an ordinary single-track
sampler or to request coupled engine behavior with the default sampler.

A single source ABC input is split into two condition-specific views instead of
asking users to provide two unrelated inputs.  This makes it possible to verify
that chain A has identical atom identity, atom ordering, and initial
coordinates in both tracks before denoising begins.

The source ABC atom array may come from either an input structure or a de novo
multi-chain base specification.  For de novo tests, a base contig such as
`90,/0,80,/0,100` is built once into an internal A+B+C source atom array and
then split into A+B and A+C views.  Track-specific `select_fixed_atoms: false`
and `select_unfixed_sequence: true` overrides should be used after splitting so
the generated chains are not treated as fixed input motifs.

Track 1 owns the shared A state.  At initialization and before every denoiser
call, track 2's A coordinates are overwritten with track 1's shared A
coordinates.  After the two denoiser calls, both tracks receive the same mixed
A update.  B and C retain independent partner states and are updated with their
own ordinary RFD3 denoiser deltas.

Shared-chain stochastic noise is identical across both tracks at each step,
while partner-chain noise remains independent.  This keeps the two denoiser
queries evaluating the same instantaneous A state rather than coupling two
different noisy realizations of A.

The proxy weight is solved per diffusion sample, not once globally across the
batch.  This preserves per-sample adaptive behavior when multiple diffusion
samples are generated in one RFD3 batch.

The implementation leaves sequence logits uncoupled.  RFD3 exposes
track-specific sequence-head outputs, but this prototype only couples
coordinate updates for the shared chain.  The merged A+B+C output therefore
uses chain A from track 1 and records this policy in metadata.

RFD3's normal per-complex formatting can compact a split two-chain view to
A+B, even when the source global chains were A+C.  The coupled engine therefore
relabels track 2's non-shared output chains to `complex_2_partners` immediately
before building `RFD3Output` objects.  This is an output-annotation step only:
it does not change the sampled coordinate tensors or denoising behavior.  It is
needed so the final files preserve the intended A+B, A+C, and merged A+B+C
chain organization.

The mode rejects classifier-free guidance, symmetry, motif realignment, and
`s_jitter_origin`.  These features alter the effective denoiser query, coordinate
frame, or update semantics and would need their own coupled derivation before
being mixed into the proxy rule.

The engine currently requires one process/GPU.  Multi-process execution would
need explicit coordination of the paired A+B/A+C track views and their shared-A
diagnostics before it is safe to support.

Trajectory dumping is supported but optional.  When enabled, the engine writes
track-specific trajectories and a merged A+B+C trajectory by appending track 2's
non-shared partner coordinates to track 1's coordinate order.

## Known Validation State

Local lightweight validation has covered syntax parsing for the copied
`rfd3_system` Python tree and direct assertions for the proxy-weight solver.

CoreHPC GPU job `703094` completed successfully for the prepared A90/B80/C100
de novo coupled run using the default RFD3 denoising step configuration. The
outputs are under:

```text
outputs/foundry/rfd3_system/a90_b80_c100_default_steps_703094/
```

Validation of the final CIF outputs found three generated models with:

- track 1 residue counts: A=90, B=80;
- track 2 residue counts: A=90, C=100;
- merged residue counts: A=90, B=80, C=100;
- shared-chain A coordinates exactly matching between track 1 and track 2 for
  all three final models.

The output metadata records `superdiff_exact: false`,
`coupling_mode: superdiff_shared_chain`, the A/B/C partner configuration, and
per-step proxy diagnostics including weights, residuals, norms, and degenerate
flags.

The prepared A90/B80/C100 de novo test uses the default RFD3 denoising step
count and can be submitted from the project root with:

```bash
sbatch jobs/rfd3_system_a90_b80_c100_smoke.sbatch
```

## References

- SuperDiff paper: https://arxiv.org/abs/2412.17762
- SuperDiff code: https://github.com/necludov/super-diffusion
- RFdiffusion3 paper: https://www.biorxiv.org/content/early/2025/11/19/2025.09.18.676967
