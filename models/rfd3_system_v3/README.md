# rfd3_system_v3

`rfd3_system_v3` is an inference-only RFdiffusion3 research prototype with two
strict reverse-SDE sampling modes:

- `reverse_sde` runs an ordinary single RFD3 trajectory with Euler-Maruyama.
- `superdiff_shared_chain_sde` couples A+B and A+C trajectories through one
  shared A state using the stochastic SuperDiff AND density-change equation.

This package is independent of `rfd3_system` (proxy coupling) and
`rfd3_system_v2` (post-churn deterministic JVP coupling). Those packages are
not imported or modified.

## Important Interpretation

RFD3 is trained to predict clean coordinates from coordinates corrupted as:

```text
noisy = clean + sigma * standard_gaussian_noise
```

V3 interprets that corruption as the marginal of a variance-exploding SDE and
uses the corresponding reverse SDE. It does not use RFD3's inference-time
churn, `step_scale`, or `noise_scale` heuristics.

The full derivation, assumptions, source ledger, and diagnostic definitions are
in [docs/stochastic_reverse_sde_superdiff_math.md](docs/stochastic_reverse_sde_superdiff_math.md).

## Single-Track Sampling

This accepts the standard RFD3 design specification. For example:

```bash
python -m rfd3_system_v3.cli design \
  inputs=null \
  out_dir=/project/outputs/foundry/rfd3_system_v3/single_track \
  ckpt_path=/weights/rfd3_latest.ckpt \
  "+specification.contig='90,/0,80,/0,100'" \
  +specification.length=270 \
  diffusion_batch_size=1 \
  n_batches=1 \
  seed=123 \
  inference_sampler.kind=reverse_sde
```

## Coupled A+B / A+C Sampling

```bash
python -m rfd3_system_v3.cli design \
  inputs=null \
  out_dir=/project/outputs/foundry/rfd3_system_v3/coupled \
  ckpt_path=/weights/rfd3_latest.ckpt \
  diffusion_batch_size=3 \
  n_batches=1 \
  seed=123 \
  coupling_mode=superdiff_shared_chain_sde \
  inference_sampler.kind=superdiff_shared_chain_sde \
  shared_chain_id=A \
  "complex_1_partners=[B]" \
  "complex_2_partners=[C]" \
  merged_output_policy=both \
  "+track_1_specification.contig='90,/0,80'" \
  +track_1_specification.length=170 \
  "+track_2_specification.contig='90,/0,100'" \
  +track_2_specification.length=190
```

The two track outputs contain identical coordinates for every mapped, movable A
atom. B and C are updated independently. Fixed shared-chain motif residues are
track-specific context and are excluded from the shared update and kappa solve.

## Main Controls

| Setting | Default | Meaning |
|---|---:|---|
| `kappa_atom_subset` | `ALL` | Atoms used to solve kappa: `ALL`, `BKBN`, or `CA`. |
| `superdiff_guidance_scale` | `1.0` | Conditional guidance `g`; `g=1` avoids the isolated-A call. |
| `superdiff_lift` | `0.0` | Per-trajectory density-change offset distributed across steps. |
| `kappa_min` | `-1.0` | Lower kappa clamp; `null` disables it. |
| `kappa_max` | `2.0` | Upper kappa clamp; `null` disables it. |
| `kappa_eps` | `1e-8` | Relative field-difference threshold for the `kappa=0.5` fallback. |
| `unconditional_reference_track` | `null` | Required only when guidance differs from 1. |

`ALL` is the primary density target. `BKBN` and `CA` are approximate ablations:
they solve one scalar kappa on fewer coordinates, then apply it to all movable A
atoms.

Strict SDE modes require:

```text
gamma_0=0
gamma_min=0
noise_scale=1
step_scale=1
use_classifier_free_guidance=false
allow_realignment=false
s_jitter_origin=0
```

Symmetry sampling is not supported in v3.

## Sequence Outputs

The coordinate-derived kappa also mixes sequence logits for mapped A residues
whose sequence is unfixed in both tracks. This makes the final movable-A
sequence prediction identical in the two output views. Fixed or chemically
different motif residues keep their track-specific identities.

This logit mixing is an output-coordination heuristic. It is not a sequence
diffusion or a SuperDiff density derivation.

## Diagnostics

Each output JSON stores per-step SDE and coupling diagnostics. Generate plots
after inference with the project ESM environment:

```bash
scripts/esm_exec.sh \
  python software/foundry/models/rfd3_system_v3/scripts/plot_sde_diagnostics.py \
  outputs/foundry/rfd3_system_v3/coupled
```

The plotter writes one batch-level set by default. Use `--model-index N` for a
single diffusion-batch sample.

## Validation

The production Foundry image does not bundle pytest. Run the dependency-free
integration checks through SLURM:

```bash
sbatch jobs/rfd3_system_v3_tests.sbatch
```

This imports the actual Foundry stack and checks the reverse-SDE update,
closed-form stochastic kappa, raw Itô residual, sequence-logit mixing, strict
configuration guards, and a complete lightweight two-track rollout.
