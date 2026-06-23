# Shared-Chain Coupling Math Notes

These notes summarize the approximate math used by the `rfd3_system`
shared-chain A+B/A+C prototype. The implementation is intentionally described
as an approximation, not exact SuperDiff density control.

## Objects in the Coupled Run

The current prototype runs two RFdiffusion3 denoising tracks:

- track 1: shared chain A with partner context B;
- track 2: shared chain A with partner context C.

Chain A is represented by one shared coordinate state. At each denoising step,
the same noisy A coordinates are presented to both tracks. The partner chains
B and C remain separate and receive their own ordinary RFD3 updates.

For the shared chain A:

```text
delta_A_1 = update proxy from the A+B track
delta_A_2 = update proxy from the A+C track
```

The prototype chooses a scalar mixing weight `kappa` per diffusion sample and
per denoising step, then constructs:

```text
delta_A_mix = kappa * delta_A_1 + (1 - kappa) * delta_A_2
```

This mixed update is applied to chain A in both tracks.

## Denoiser Prediction and Score Interpretation

In generic EDM notation:

```text
x            = current noisy state
sigma        = current noise level
D(x, sigma)  = denoiser prediction of the clean state
```

`D(x, sigma)` can be interpreted as the denoising network's estimate of the
clean structure given the current noisy structure:

```text
D(x, sigma) ~= E[x0 | x at noise level sigma]
```

For Gaussian denoising models, a Tweedie-style relation connects the denoiser
prediction to the score:

```text
score(x, sigma) ~= (D(x, sigma) - x) / sigma^2
```

RFD3 exposes an EDM-style denoiser prediction in the sampler. In the code, the
analogous objects are:

```text
x              -> X_noisy_L
D(x, sigma)    -> X_denoised_L
sigma          -> t_hat
```

RFD3 computes the denoiser-derived update proxy:

```text
delta = (X_noisy_L - X_denoised_L) / t_hat
```

If `t_hat` is identified with the EDM noise level `sigma`, then:

```text
delta = (x - D(x, sigma)) / sigma
      ~= -sigma * score(x, sigma)
```

So `delta` is not the exact score. It is a score-like denoising update vector
with a sign flip and a noise-level scaling. In this prototype, `delta_A_1` and
`delta_A_2` are compared at the same denoising step and therefore the same
scheduled `t_hat`, which makes them locally comparable as track-specific update
pressures on shared chain A.

## Relation Between `sigma` and `t_hat`

`sigma` is the generic EDM/math notation for the noise level. `t_hat` is the
RFD3 implementation's scheduled noise value. RFD3 constructs `t_hat` from an
EDM/Karras-style noise schedule:

```text
t_hat = sigma_data * (s_max^(1/p) + t * (s_min^(1/p) - s_max^(1/p)))^p
```

For interpreting this implementation:

```text
sigma ~= t_hat
```

This does not make the prototype exact SuperDiff. It only explains why RFD3's
denoiser-derived `delta` is directly related to an EDM score-like direction.

## Full RFD3 Denoising Step

An RFD3 sampler step is not simply:

```text
x_next = x + delta
```

It has a stochastic noise-injection stage followed by a denoiser prediction and
an Euler-like update. Let:

```text
X_L        = current coordinate state
c_prev     = current noise schedule value
c_next     = next lower noise schedule value
gamma      = gamma_0 if c_next > gamma_min else 0
t_hat      = c_prev * (1 + gamma)
step_scale = RFD3 sampler step scale
```

RFD3 first optionally increases the noise level from `c_prev` to `t_hat`:

```text
t_hat = c_prev * (1 + gamma)
```

It then samples Gaussian noise at the matching variance increment:

```text
epsilon =
    noise_scale * sqrt(t_hat^2 - c_prev^2) * N(0, I)
```

and forms the denoiser query:

```text
X_noisy = X_L + epsilon
```

Fixed motif atoms have their noise set to zero. The denoiser predicts:

```text
X_denoised = D(X_noisy, t_hat)
```

RFD3 then computes:

```text
delta = (X_noisy - X_denoised) / t_hat
d_t   = c_next - t_hat
```

and updates coordinates with:

```text
X_next = X_noisy + step_scale * d_t * delta
```

Since `c_next < t_hat` during normal denoising, `d_t` is negative. This can
also be read as moving from the noisy query point back toward the denoiser
prediction:

```text
X_next =
    X_noisy
  + step_scale * (t_hat - c_next) / t_hat * (X_denoised - X_noisy)
```

Thus each step is:

```text
current coordinates -> stochastic noisy query -> denoiser prediction -> update toward denoised prediction
```

## Coupled Shared-Chain Noise Handling

In `rfd3_system`, the shared chain A is presented to both tracks with the same
instantaneous noisy coordinates. The implementation first makes track 2's
shared A coordinates match track 1:

```text
X2_L[A] = X1_L[A]
```

It then samples independent partner noise for the full track tensors, but
replaces the shared-chain noise in both tracks with the same Gaussian sample:

```text
epsilon_shared =
    noise_scale * sqrt(t_hat^2 - c_prev^2) * N(0, I)

epsilon_1[A] = epsilon_shared
epsilon_2[A] = epsilon_shared
```

Fixed shared atoms have their shared noise set to zero, and fixed atoms in each
track also have their noise set to zero. After this assignment:

```text
X1_noisy[A] = X1_L[A] + epsilon_shared
X2_noisy[A] = X2_L[A] + epsilon_shared
```

Because `X2_L[A]` was overwritten with `X1_L[A]` first, the shared-chain noisy
states are identical before denoising:

```text
X1_noisy[A] = X2_noisy[A]
```

The partner-chain noise remains condition-specific:

```text
epsilon_1[B] is independent
epsilon_2[C] is independent
```

This is intentional. It means the two denoiser calls evaluate the same noisy
state of A under two different contexts, rather than two unrelated noisy
realizations of A.

## Would Dividing by `t_hat` Make `delta` More Score-Like?

A score-like proxy could be formed as:

```text
score_proxy = -delta / t_hat
            = (X_denoised_L - X_noisy_L) / t_hat^2
```

This is closer to the generic EDM score expression. However, for the current
two-track same-timestep kappa solve, replacing both track updates by the same
shared scalar multiple mostly leaves `kappa_raw` unchanged.

If:

```text
delta_A_1' = c * delta_A_1
delta_A_2' = c * delta_A_2
```

then with `proxy_norm_weight = 1`, the numerator and denominator in the kappa
solve both scale by `c^2`, so the unconstrained `kappa_raw` is unchanged. The
sign flip also cancels because both tracks are transformed identically.

Using a score-scaled proxy may still change:

- diagnostic magnitudes such as norms and residuals;
- behavior if future thresholds use absolute delta/residual sizes;
- behavior if `proxy_norm_weight` changes;
- behavior if tracks ever use different effective noise scales or masks.

For this reason, the current implementation keeps the RFD3 sampler-native
`delta` as the coupling proxy and documents it as approximate.

## Proxy Equalization Quantity

The prototype does not compute exact SuperDiff density changes. Instead, it
defines a scalar proxy value for each track:

```text
proxy_i(delta_mix) = <delta_mix, delta_i> - w * ||delta_i||^2
```

where:

- `delta_i` is the shared-chain update proxy from track `i`;
- `delta_mix` is the proposed mixed shared-chain update;
- `<., .>` is the sum of coordinate-wise products over all shared-chain atoms;
- `||.||^2` is the corresponding squared norm;
- `w = proxy_norm_weight`, currently `1.0`.

The vector `delta_i` is the score-like denoising update proxy. The scalar
`proxy_i(delta_mix)` is the quantity that is equalized between tracks.

## Kappa Solve

Let:

```text
delta_1 = track 1 shared-chain update proxy
delta_2 = track 2 shared-chain update proxy
Delta   = delta_1 - delta_2
delta_mix = kappa * delta_1 + (1 - kappa) * delta_2
w = proxy_norm_weight
```

First rewrite the mixed update:

```text
delta_mix = delta_2 + kappa * (delta_1 - delta_2)
delta_mix = delta_2 + kappa * Delta
```

The proxy equalization condition is:

```text
proxy_1(delta_mix) = proxy_2(delta_mix)
```

Substitute the proxy definitions:

```text
<delta_mix, delta_1> - w ||delta_1||^2
    =
<delta_mix, delta_2> - w ||delta_2||^2
```

Move terms:

```text
<delta_mix, delta_1> - <delta_mix, delta_2>
    =
w ||delta_1||^2 - w ||delta_2||^2
```

Factor the inner product:

```text
<delta_mix, delta_1 - delta_2>
    =
w (||delta_1||^2 - ||delta_2||^2)
```

Use `Delta = delta_1 - delta_2`:

```text
<delta_mix, Delta>
    =
w (||delta_1||^2 - ||delta_2||^2)
```

Substitute `delta_mix = delta_2 + kappa * Delta`:

```text
<delta_2 + kappa * Delta, Delta>
    =
w (||delta_1||^2 - ||delta_2||^2)
```

Distribute:

```text
<delta_2, Delta> + kappa * <Delta, Delta>
    =
w (||delta_1||^2 - ||delta_2||^2)
```

Since `<Delta, Delta> = ||Delta||^2`:

```text
<delta_2, Delta> + kappa * ||Delta||^2
    =
w (||delta_1||^2 - ||delta_2||^2)
```

Solve for `kappa`:

```text
kappa * ||Delta||^2
    =
w (||delta_1||^2 - ||delta_2||^2) - <delta_2, Delta>
```

Therefore:

```text
kappa_raw =
    [w (||delta_1||^2 - ||delta_2||^2) - <delta_2, delta_1 - delta_2>]
    /
    ||delta_1 - delta_2||^2
```

The implementation then handles degenerate cases and clamps:

```text
if ||delta_1 - delta_2||^2 <= eps:
    kappa_raw = 0.5

kappa = clamp(kappa_raw, kappa_min, kappa_max)
```

Current defaults:

```text
eps = 1e-8
kappa_min = -1.0
kappa_max = 2.0
proxy_norm_weight = 1.0
```

The clamped range allows extrapolation beyond a convex average. For example:

- `kappa = 1` uses the track 1 update only;
- `kappa = 0` uses the track 2 update only;
- `kappa = 0.5` is an equal mix;
- `kappa > 1` extrapolates past track 1 away from track 2;
- `kappa < 0` extrapolates past track 2 away from track 1.

## Coordinate Update

After solving for `kappa`, the shared-chain update is:

```text
delta_A_mix = kappa * delta_A_1 + (1 - kappa) * delta_A_2
```

RFD3's sampler then applies:

```text
A_next = A_noisy + step_scale * d_t * delta_A_mix
```

The non-shared partner chains are updated normally by their own tracks:

```text
X1_next = X1_noisy + step_scale * d_t * delta_1
X2_next = X2_noisy + step_scale * d_t * delta_2
```

Then the shared chain positions in both tracks are overwritten with the same
`A_next`:

```text
X1_next[A] = A_next
X2_next[A] = A_next
```

## Proxy Residual

The proxy residual is computed after the final, clamped `kappa` has been chosen:

```text
proxy_residual = proxy_1(delta_mix) - proxy_2(delta_mix)
```

Expanded:

```text
proxy_residual =
    [<delta_mix, delta_1> - w ||delta_1||^2]
  - [<delta_mix, delta_2> - w ||delta_2||^2]
```

A residual close to zero means the implemented proxy equalization equation was
well satisfied. A large residual means the proxy condition was not well
satisfied, commonly because:

- `kappa_raw` was outside the allowed range and had to be clamped;
- the denominator `||delta_1 - delta_2||^2` was small;
- the two update proxies created an ill-conditioned local solve.

If `degenerate = true`, the solve was considered numerically unreliable and
the implementation falls back to `kappa_raw = 0.5` before clamping.

## Kappa Plot Interpretation

Future coupled merged outputs include a batch-level SVG plot:

```text
*_merged_kappa.svg
```

The x-axis tick labels are normalized `t` values. They increase from 0 on the
left to 1 on the right, matching the direction of the denoising process from
the noisiest state to the final denoised end. The y-axis is `kappa` in:

```text
delta_mix = kappa * delta(track 1) + (1 - kappa) * delta(track 2)
```

Interpretation:

- higher `kappa` means the shared-chain update leans more toward track 1;
- lower `kappa` means it leans more toward track 2;
- `kappa = 0.5` is an equal mix;
- values outside `[0, 1]` are extrapolating rather than interpolating.

The physical RFD3 noise scale `t_hat` is still saved in JSON diagnostics for
analysis, but it is not used as the kappa plot x-axis.

## Exactness Caveat

This implementation does not use exact model scores or the SuperDiff Ito
density estimator. It mixes RFD3 denoiser-derived update proxies at inference
time. The diagnostics should therefore be interpreted as local, approximate
coupling diagnostics, not exact density-control quantities.
