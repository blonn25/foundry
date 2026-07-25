# RFdiffusion3 System v2

`rfd3_system_v2` is an inference-only research copy of Foundry RFdiffusion3.
It implements a post-churn deterministic SuperDiff AND density-rate estimate
for one shared chain in two condition-specific trajectories:

- track 1: shared A with partner B;
- track 2: shared A with partner C;
- optional reference: isolated A from either track, without its partner.

This package is separate from both `models/rfd3` and `models/rfd3_system`.
The original package remains Foundry's implementation, while `rfd3_system`
retains the earlier denoiser-proxy experiment.

## Method

At every step, RFD3 first applies its normal stochastic churn using identical
noise for the movable atoms of A in both tracks. From that common post-churn
state, v2:

1. evaluates the A+B and A+C denoiser fields;
2. estimates `-div(v_i)` on a selected subset of A using shared Hutchinson
   probes and forward-mode JVPs;
3. solves the deterministic two-density SuperDiff equation for `kappa`;
4. applies the resulting mixed field to every movable atom of A;
5. applies ordinary condition-specific RFD3 updates to B and C.

The density accounting covers the deterministic post-churn step. It does not
include the preceding stochastic churn transition, so it is a principled local
density-rate estimate rather than exact end-to-end stochastic density tracking.
See [docs/deterministic_superdiff_math.md](docs/deterministic_superdiff_math.md)
for the derivation, assumptions, diagnostics, and parameter definitions.

## Interface

The new path is opt-in:

```bash
PYTHONPATH=/project/software/foundry/models/rfd3_system_v2/src:/project/software/foundry/src \
python -m rfd3_system_v2.cli design \
  coupling_mode=superdiff_shared_chain_density \
  inference_sampler.kind=superdiff_shared_chain_density \
  shared_chain_id=A \
  "complex_1_partners=[B]" \
  "complex_2_partners=[C]" \
  unconditional_reference_track=track2 \
  inference_sampler.kappa_atom_subset=ALL \
  inference_sampler.superdiff_guidance_scale=1.0 \
  inference_sampler.superdiff_lift=0.0 \
  inference_sampler.kappa_min=-1.0 \
  inference_sampler.kappa_max=2.0 \
  inference_sampler.density_hutchinson_probes=1 \
  inference_sampler.density_validation_probes=0 \
  ...
```

`unconditional_reference_track` is always required so the scientific choice is
recorded. At the default guidance scale `g=1`, the isolated-A field cancels
algebraically and is not evaluated. At `g!=1`, v2 constructs isolated A from
the selected track. Use `track2` when track 2 contains the desired
unphosphorylated shared-chain state.

An expanded override example is in
[docs/examples/superdiff_shared_chain_density.yaml](docs/examples/superdiff_shared_chain_density.yaml).

## Current Boundaries

- Forward-mode JVP is required. There is deliberately no silent VJP or proxy
  fallback.
- Native RFD3 classifier-free guidance, symmetry, realignment, and origin
  jitter are unsupported in this coupled path.
- `superdiff_guidance_scale` must be greater than zero.
- One process and one GPU are supported.
- Fixed shared motif residues are condition-specific context and are excluded
  from the shared update and density solve.
- Sequence-head predictions remain track-specific.

## Diagnostics

Output JSON metadata records per-step `kappa`, estimated divergences, density
rates, post-clamp residuals, field norms, cosine similarities, and optional
independent-probe validation residuals. Generate PNGs after inference with:

```bash
envs/esm/bin/python \
  software/foundry/models/rfd3_system_v2/scripts/plot_density_diagnostics.py \
  outputs/foundry/rfd3_system_v2/<run_directory>
```

Keep generated outputs outside this source directory.

## Validation

CoreHPC GPU smoke validation completed on NVIDIA L40S nodes:

- job `1069924`: `g=1`, one solve probe, no isolated-A model evaluation;
- job `1069932`: `g=1.5`, track-2 isolated-A reference, one solve probe,
  and one independent validation probe.

Both jobs completed with `0:0`, wrote A+B, A+C, and both merged-output policies,
and produced all expected diagnostic plots. Shared-chain CA coordinates matched
exactly between the two track outputs in the default smoke run.

During validation, two PyTorch forward-AD gaps were identified and handled
without changing differentiation methods:

- sparse top-k attention neighborhoods are treated as locally constant by
  detaching only their `cdist` inputs;
- atom-to-token scatter mean uses an exact `scatter_add / count`
  decomposition instead of unsupported `index_reduce`.
