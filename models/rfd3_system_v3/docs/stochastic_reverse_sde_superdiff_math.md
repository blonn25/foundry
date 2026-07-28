# Stochastic Reverse SDE and SuperDiff AND

This note defines the mathematics implemented by `rfd3_system_v3`, distinguishes
source facts from derived assumptions, and documents every user-facing coupling
parameter.

## 1. What RFD3 Was Trained To Predict

RFD3 training does not simulate a complete trajectory. For each example it
samples a noise level `sigma`, adds Gaussian coordinate noise, and trains the
network to predict the clean coordinates:

```text
epsilon ~ standard normal
noisy_structure = clean_structure + sigma * epsilon
denoised_structure = D(noisy_structure, sigma)
```

The public configuration uses `sigma_data = 16`. Training noise levels follow
the log-normal sampling implemented by AtomWorks `SampleEDMNoise`.

For additive isotropic Gaussian corruption, a conditional-mean denoiser gives
the score estimate:

```text
score(x, sigma) = (D(x, sigma) - x) / sigma^2
```

RFD3's native sampler works with the equivalent velocity:

```text
velocity(x, sigma) = (x - D(x, sigma)) / sigma
velocity = -sigma * score
```

Coordinates are measured in angstroms, `sigma` has angstrom units, the score
has inverse-angstrom units, and the velocity is dimensionless when sigma is
treated as the integration variable.

## 2. Forward and Reverse SDE

The Gaussian corruption above is the marginal distribution of this
variance-exploding forward SDE:

```text
dX = sqrt(2 * sigma) * dW
```

The accumulated variance from noise level zero to `sigma` is:

```text
integral from 0 to sigma of 2 * s ds = sigma^2
```

This exactly recovers `clean + sigma * epsilon` for the primary isotropic
corruption.

The corresponding reverse-time SDE has drift determined by the score. Written
using RFD3's velocity and stepping from `sigma` to a smaller `sigma_next`:

```text
delta_sigma = sigma_next - sigma       # negative
h = -delta_sigma                       # positive

brownian_noise =
    sqrt(2 * sigma * h) * standard_gaussian_noise

coordinate_increment =
    2 * delta_sigma * velocity
    + brownian_noise

x_next = x + coordinate_increment
```

This is Euler-Maruyama. The factor of two distinguishes the reverse SDE from
the probability-flow ODE, whose drift would use only
`delta_sigma * velocity`.

V3 retains RFD3's Karras/AF3 sigma schedule and its default 200 schedule
points. The final level remains:

```text
sigma_data * s_min = 16 * 0.0004 = 0.0064
```

V3 does not silently append a final denoiser call at sigma zero.

## 3. Two Conditional Views of Shared A

At every step, both model evaluations receive exactly the same movable A
coordinates:

```text
track 1: A + B  -> velocity_1
track 2: A + C  -> velocity_2
```

Only the A components of these fields participate in the SuperDiff solve. B
and C are evolving contexts. They are not included in the density equality and
are not coupled to one another.

Mapped A atoms with fixed coordinates are context rather than shared state.
They receive no Brownian noise or drift and are excluded from kappa.

## 4. Optional Isolated-A Guidance

When `superdiff_guidance_scale` is not one, v3 also evaluates isolated A:

```text
velocity_0 = field predicted for isolated A

base_velocity =
    velocity_0
    + guidance * (velocity_2 - velocity_0)
```

The isolated input is constructed from the explicitly selected
`unconditional_reference_track`. For the SEP/SER use case, track 2 is the
unphosphorylated reference.

At the default `guidance = 1`:

```text
base_velocity = velocity_2
```

The isolated-A field cancels, so the extra denoiser call is skipped. Guidance
must remain strictly greater than zero because it appears in the kappa
denominator.

## 5. Stochastic SuperDiff Kappa

Define the conditional field difference:

```text
field_difference = velocity_1 - velocity_2
```

Draw the Brownian noise for A:

```text
A_noise = sqrt(2 * sigma * h) * gaussian_A
```

This exact tensor is used both in the kappa equation and in the final A update.
Using a different draw would invalidate the finite-step Itô equality.

First construct the update that would be taken at kappa zero:

```text
base_A_increment =
    2 * delta_sigma * base_velocity
    + A_noise
```

The implemented two-track solve is:

```text
numerator =
    h * dot(velocity_2 - velocity_1, velocity_2 + velocity_1)
    - dot(base_A_increment, velocity_1 - velocity_2)
    + sigma * lift / number_of_steps

denominator =
    2
    * delta_sigma
    * guidance
    * squared_norm(velocity_1 - velocity_2)

kappa_raw = numerator / denominator
```

All dot products and norms are reduced separately for each diffusion-batch
sample.

The applied value is:

```text
kappa = clamp(kappa_raw, kappa_min, kappa_max)
```

The defaults are `-1` and `2`. These bounds allow limited extrapolation while
preventing numerically extreme updates. Clamping intentionally sacrifices exact
equality, which is why both raw and applied residuals are recorded.

If the relative squared field difference is at most `kappa_eps`, the two
conditions are treated as indistinguishable and `kappa = 0.5` is used.

The mixed field and actual A update are:

```text
mixed_A_velocity =
    base_velocity
    + guidance * kappa * (velocity_1 - velocity_2)

actual_A_increment =
    2 * delta_sigma * mixed_A_velocity
    + A_noise

A_next = A_current + actual_A_increment
```

The same `A_next` is copied into both track views.

At unit guidance and zero lift, the unclamped solution simplifies to:

```text
kappa_raw =
    0.5
    + dot(A_noise, velocity_1 - velocity_2)
      / (
          2
          * h
          * squared_norm(velocity_1 - velocity_2)
        )
```

This makes the stochastic dependence explicit: the same Brownian realization
that moves A also determines the weight needed to equalize its estimated
conditional density changes.

## 6. Itô Density-Change Estimate

For each condition, v3 records:

```text
delta_log_q_i =
    -(h / sigma) * squared_norm(velocity_i)
    - dot(actual_A_increment, velocity_i) / sigma
```

The density difference and solver residual are:

```text
density_difference = delta_log_q_1 - delta_log_q_2

target_difference = -lift / number_of_steps

density_residual =
    density_difference - target_difference
```

With zero lift and no clamping, the algebraic raw residual should be near
floating-point zero. A small residual means the finite-step estimator was
equalized; it does not prove that the learned RFD3 conditional densities are
exact.

## 7. Updates Outside the Kappa Subset

`kappa_atom_subset` controls only the dot products used to solve the scalar:

```text
ALL  = every mapped, movable shared-A atom
BKBN = N, CA, C, and O atoms
CA   = alpha-carbon atoms only
```

The resulting scalar always mixes and updates all mapped, movable A atoms.
`ALL` is the primary density target. `BKBN` and `CA` are approximate ablations
because they equalize only a projected coordinate state.

B and C use their own conditional fields:

```text
B_next =
    B_current
    + 2 * delta_sigma * velocity_1_B
    + independent_B_noise

C_next =
    C_current
    + 2 * delta_sigma * velocity_2_C
    + independent_C_noise
```

The Brownian increments for B and C are independent of each other and of A.

## 8. Sequence-Logit Policy

RFD3 sequence logits are auxiliary predictions rather than a diffused state.
After the last step, v3 coordinates mapped, sequence-designable A residues
using the applied coordinate kappa:

```text
mixed_logits =
    track_2_logits
    + kappa * (track_1_logits - track_2_logits)
```

With optional guidance:

```text
base_logits =
    reference_logits
    + guidance * (track_2_logits - reference_logits)

mixed_logits =
    base_logits
    + guidance * kappa * (track_1_logits - track_2_logits)
```

The mixed logits and their argmax identities are placed in both track views.
Residues with fixed sequence in either track and fixed/variant motifs are left
unchanged. This policy is a coherent output heuristic, not a sequence-density
claim.

## 9. Unsupported State Changes

Strict SDE paths reject native churn, non-unit `step_scale` or `noise_scale`,
native CFG, per-step realignment, origin jitter, and symmetry. These operations
would change coordinates outside the documented SDE.

Input centering, ORI conditioning, fixed motifs, guideposts, ligands, and
partial diffusion remain supported because they define the initial state or
conditioning rather than inserting undocumented trajectory steps.

## 10. Assumptions and Limitations

1. `D(x, sigma)` is treated as an estimate of the conditional clean-coordinate
   mean, allowing the standard Gaussian denoiser-to-score identity.
2. The reverse SDE matches RFD3's primary isotropic coordinate corruption, not
   every training augmentation.
3. RFD3 training additionally adds a small common COM translation to movable
   atoms. Modeling it exactly would require a mask-dependent, rank-3 covariance
   SDE and a generalized score. Native RFD3 sampling does not separately reverse
   this augmentation either, so v3 defers it and documents the omission.
4. Euler-Maruyama and the finite-step Itô estimator introduce discretization
   error.
5. A+B and A+C are conditional views with evolving partner contexts, not two
   independently normalized explicit density functions.
6. Fixed atoms and atoms absent from either shared map are outside the density
   state.
7. Kappa clamping and reduced atom subsets intentionally make the applied
   update approximate.

## 11. Source Ledger

The local implementation was derived against Foundry commit:

```text
afbb5e55ef7b3454681aff544eba40d8347aed8f
```

Local sources:

- `models/rfd3/src/rfd3/transforms/pipelines.py`
  - Instantiates `SampleEDMNoise` and the RFD3 design transforms.
- `models/rfd3/src/rfd3/model/RFD3_diffusion_module.py`
  - Defines EDM preconditioning and clean-coordinate output `D(x, sigma)`.
- `models/rfd3/src/rfd3/model/inference_sampler.py`
  - Defines the native sigma schedule, churn, and ODE-style update.
- `models/rfd3/src/rfd3/transforms/design_transforms.py`
  - Defines centering, rigid augmentation, and the auxiliary COM perturbation.
- `models/rfd3/src/rfd3/metrics/losses.py`
  - Defines the noise-weighted coordinate regression objective.

External primary sources:

- RFdiffusion3 preprint:
  <https://www.biorxiv.org/content/10.1101/2025.09.18.676967v1>
- AtomWorks v2.1.1 EDM transform:
  <https://github.com/RosettaCommons/atomworks/blob/v2.1.1/src/atomworks/ml/transforms/diffusion/edm.py>
- Elucidating the Design Space of Diffusion-Based Generative Models:
  <https://arxiv.org/abs/2206.00364>
- Score-Based Generative Modeling through Stochastic Differential Equations:
  <https://arxiv.org/abs/2011.13456>
- The Superposition of Diffusion Models Using the Itô Density Estimator:
  <https://arxiv.org/abs/2412.17762>
- Official stochastic AND notebook:
  <https://github.com/necludov/super-diffusion/blob/main/notebooks/superposition_AND.ipynb>

The RFD3 corruption and denoiser parameterization are sourced facts. The
specific VE forward SDE, reverse-SDE sampler, and its application to a
track-conditioned shared protein chain are the documented v3 derivation.
