# Proxy-Kappa Regularization Grid 001

## Purpose

This experiment tests whether scale-aware regularization stabilizes the
approximate two-track kappa solve without replacing it with a hard
denominator threshold. The tested pre-clamp solve is:

```text
S = 0.5 * (||delta_1||^2 + ||delta_2||^2)
D = ||delta_1 - delta_2||^2
R = D / (D + rho*S)

kappa_regularized = 0.5 + R * (kappa_raw - 0.5)
kappa_applied = clamp(kappa_regularized, -1, 2)
```

Rho is dimensionless. It controls when the difference between the two update
vectors is large enough, relative to their magnitudes, to trust the raw solve.
Rho zero exactly reproduces the previous implementation.

The complete derivation is in `shared_chain_coupling_math.md`.

## Experiment

- Date: July 29, 2026
- Foundry commit: `7369e6a`
- Smoke inference: CoreHPC job `1135274`
- Smoke collector: CoreHPC job `1135275`
- Full 12-row array: CoreHPC job `1135276`
- Full collector: CoreHPC job `1135277`
- Hardware: one NVIDIA L40S (`ggpu1-11`) for every full-grid row
- Designs per condition: three A90/B80/C100 systems
- Timesteps: 200 schedule values, producing 199 update steps
- Seed: 123
- Kappa atom subset: `ALL`
- Rho values: `0`, `1e-5`, `1e-4`, `1e-3`

Sampling modes:

| Mode | `step_scale` | `gamma_0` |
|---|---:|---:|
| Default | 1.5 | 0.6 |
| ODE | 1.5 | 0.0 |
| Binder | 3.0 | 0.2 |

All other schedule, proxy, clamp, and output controls were held fixed. Final
CIF/JSON outputs were saved without coordinate trajectories.

## Results

The clamp fraction is the fraction of all 199 x 3 applied weights where
regularization still left kappa outside `[-1, 2]`. Step variation is the mean
absolute change in applied kappa between consecutive denoising steps. The
normalized residual is `|proxy_1 - proxy_2| / S` after regularization and any
required clamping.

| Mode | Rho | Clamp fraction | Median reliability | Applied step variation | Median normalized residual |
|---|---:|---:|---:|---:|---:|
| Default | 0 | 31.16% | 1.000 | 0.6524 | 8.72e-8 |
| Default | 1e-5 | 15.58% | 0.979 | 0.5801 | 2.91e-6 |
| Default | 1e-4 | 0.34% | 0.818 | 0.3055 | 1.66e-5 |
| Default | 1e-3 | 0% | 0.329 | 0.0983 | 6.74e-5 |
| ODE | 0 | 17.25% | 1.000 | 0.0426 | 4.08e-8 |
| ODE | 1e-5 | 10.55% | 0.999 | 0.0432 | 1.46e-6 |
| ODE | 1e-4 | 0% | 0.997 | 0.0254 | 9.74e-6 |
| ODE | 1e-3 | 0% | 0.971 | 0.0178 | 4.21e-5 |
| Binder | 0 | 22.78% | 1.000 | 0.5575 | 8.27e-8 |
| Binder | 1e-5 | 14.91% | 0.987 | 0.5163 | 2.16e-6 |
| Binder | 1e-4 | 1.01% | 0.884 | 0.2843 | 1.23e-5 |
| Binder | 1e-3 | 0% | 0.444 | 0.0928 | 4.87e-5 |

## Interpretation

`rho=1e-4` is the best tested opt-in starting point:

- it nearly eliminated clamp saturation in all three sampling modes;
- it reduced applied-kappa step variation by about 40-53% relative to rho zero;
- median reliability remained `0.818`, `0.997`, and `0.884` for default, ODE,
  and binder sampling, respectively;
- normalized proxy residuals increased, as required by the regularization
  tradeoff, but remained much lower than at `rho=1e-3`.

`rho=1e-5` was too weak to prevent frequent clamping. `rho=1e-3` was effective
at suppressing extreme weights, but its median reliability shows that it
substantially damped the raw solve over much of the default and binder
trajectories.

The code default remains `rho=0` for backward compatibility. Users who want
the tested stabilization should set:

```bash
inference_sampler.proxy_kappa_regularization_rho=1e-4
```

## Limitations

- This is a three-sample numerical stability experiment, not a structure
  quality or design-success benchmark.
- Regularization changes the trajectory after the first step. Raw-kappa
  diagnostics at different rho values therefore do not describe an identical
  coordinate trajectory at later steps.
- The result applies directly to the tested A90/B80/C100 de novo setup with
  `kappa_atom_subset=ALL`. Motif-conditioned campaigns and `BKBN`/`CA` solves
  should be checked separately.
- Kappa stability does not establish that the proxy is an exact SuperDiff
  density-change estimator.

## Artifacts

On CoreHPC and the Wynton mirror:

```text
outputs/foundry/rfd3_system/proxy_kappa_regularization_grid_001/
```

The `summary/` directory contains the complete CSV and cross-grid plots. Each
condition directory contains final structures, output JSON, row metadata, and
batch-level kappa, reliability, denominator, residual, and cosine plots.

All full-grid GPU tasks completed in 81-82 seconds with approximately 11.3 GB
peak host RSS. The CPU collector completed in 22 seconds with approximately
275 MB peak RSS.
