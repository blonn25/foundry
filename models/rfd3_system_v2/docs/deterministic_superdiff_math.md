# Deterministic SuperDiff AND for Shared-Chain RFD3

## Scope

`rfd3_system_v2` couples two RFD3 trajectories that contain one shared
coordinate state:

```text
track 1: A + B
track 2: A + C
```

Only movable atoms of A are coupled. B and C retain independent coordinates,
churn noise, and ordinary RFD3 updates. Fixed shared motif residues may differ
chemically between tracks, but they are excluded from the shared state and the
density calculation.

## Post-Churn State

Let `sigma_prev` be the current scheduled noise level. RFD3 applies churn:

```text
gamma = gamma_0 if sigma_next > gamma_min else 0
sigma = sigma_prev * (1 + gamma)
epsilon_scale = noise_scale * sqrt(sigma^2 - sigma_prev^2)
x_hat = x + epsilon_scale * Normal(0, I)
```

Movable A atoms receive the same sampled noise in both tracks, making their
post-churn coordinates identical. B and C receive independent noise. The v2
density solve starts at `x_hat`; it does not estimate the density change caused
by this churn transition.

## Denoiser Fields

For RFD3 denoiser prediction `D_i(x_hat, sigma)`, define:

```text
v_i = (x_hat - D_i(x_hat, sigma)) / sigma
```

On shared A:

```text
v1 = field under A+B
v2 = field under A+C
v0 = field for isolated A from unconditional_reference_track
```

`v0` is the analogue of the unconditional field used for classifier-free
guidance. It is evaluated only when `g != 1`.

The guided base and condition difference are:

```text
Delta  = v1 - v2
v_base = v0 + g * (v2 - v0)
       = v2                         when g = 1
```

The pre-step-scale mixed field is:

```text
v_mix(kappa) = v_base + g * kappa * Delta
```

The actual shared-A path field includes RFD3's user-selected step scale:

```text
u(kappa) = step_scale * v_mix(kappa)
```

The coordinate update is:

```text
d_sigma = sigma_next - sigma
A_next  = A_hat + d_sigma * u(kappa)
```

Because the schedule decreases, `d_sigma` is normally negative.

## Density-Change Estimate

For each conditional field, the deterministic SuperDiff density rate used by
the implementation is:

```text
rho_i(u) = -div_A(v_i) + <v_i, v_i - u> / sigma
```

All divergences and inner products are restricted to
`inference_sampler.kappa_atom_subset`:

- `ALL`: every movable shared-A atom;
- `BKBN`: movable N, CA, C, and O atoms;
- `CA`: movable CA atoms.

The resulting scalar `kappa` is still applied to the mixed update for all
movable shared-A atoms.

### Hutchinson JVP

The divergence is estimated with Rademacher probes `z`:

```text
div_A(v_i) = trace(J_i)
           ~= z^T J_i z

-div_A(v_i) ~= -z^T JVP(v_i, z)
```

The same `z` is used for both tracks at a given probe, reducing variance in the
difference. `density_hutchinson_probes` estimates are averaged. Forward-mode
JVP is mandatory; a failure raises an error instead of changing methods.

`density_validation_probes` optionally draws fresh probes after solving. These
do not affect the trajectory and measure how well the selected `kappa`
generalizes beyond the random trace estimate used to solve it.

## Closed-Form Kappa

For zero lift, v2 solves:

```text
rho_1(u(kappa)) = rho_2(u(kappa))
```

Define:

```text
dlog1 = -div_A(v1)
dlog2 = -div_A(v2)
s     = step_scale
```

Then:

```text
kappa_raw =
    [ sigma * (dlog1 - dlog2)
      + ||v1||^2 - ||v2||^2
      - s * <Delta, v_base> ]
    /
    [ s * g * ||Delta||^2 ]
```

This is the deterministic two-condition SuperDiff AND solve adapted to the
actual scaled path used by RFD3. If the denominator is numerically degenerate,
v2 uses `kappa=0.5` and records `degenerate=true`.

The applied value is:

```text
kappa = clamp(kappa_raw, kappa_min, kappa_max)
```

The defaults are `[-1, 2]`. Either bound may be set to `null`. Density rates
and residuals in output metadata are recomputed after clamping, so a nonzero
residual correctly exposes the constraint introduced by the clamp.

## Lift

`superdiff_lift` defaults to zero. Following the notebook convention, nonzero
lift adds:

```text
sigma * lift / (num_steps * d_sigma)
```

to the kappa numerator. This targets:

```text
rho_1 - rho_2 = -lift / (num_steps * d_sigma)
```

and therefore a per-step log-density increment difference:

```text
Delta log(q1) - Delta log(q2) = -lift / num_steps
```

Lift is retained as an experimental control rather than a recommended default.

## Diagnostics

For each diffusion sample and step, metadata includes:

- `raw_kappa`, `kappa`, `clamped`, and `degenerate`;
- `dlog_1`, `dlog_2`;
- `density_rate_1`, `density_rate_2`;
- `density_rate_residual`, computed after clamping;
- `delta_log_q_1`, `delta_log_q_2`;
- field norms and `||v1-v2||^2`;
- cosines between each conditional field and the applied path;
- `validation_density_rate_residual` when validation probes are enabled;
- physical `t_hat`/`sigma` and normalized schedule `t`.

## Difference From Exact Stochastic SuperDiff

This implementation is more faithful than the v1 denoiser-alignment proxy
because it estimates the divergence term and solves the deterministic
SuperDiff density-rate equation. It is still not exact accounting for RFD3's
complete stochastic transition:

1. churn is performed before the solve and omitted from accumulated density
   accounting;
2. divergences use finite-sample Hutchinson estimates;
3. clamping can intentionally violate the equality;
4. RFD3's learned denoiser field is treated as the probability-flow field for
   this local deterministic step.

When `gamma=0`, no churn noise is added and this distinction narrows to trace
estimation, model-field interpretation, and any active clamp.

## User Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `superdiff_guidance_scale` | `1.0` | CFG-like scale `g`; must be positive. |
| `unconditional_reference_track` | required | Track whose isolated A defines `v0`; use `track2` for unphosphorylated A. |
| `superdiff_lift` | `0.0` | Experimental cumulative density preference. |
| `kappa_min`, `kappa_max` | `-1.0`, `2.0` | Applied-kappa bounds; `null` disables a bound. |
| `kappa_eps` | `1e-8` | Degenerate denominator threshold. |
| `kappa_atom_subset` | `ALL` | Atoms used for divergence and kappa only. |
| `density_hutchinson_probes` | `1` | Shared probes used in the solve. |
| `density_validation_probes` | `0` | Fresh diagnostic probes; zero disables them. |

