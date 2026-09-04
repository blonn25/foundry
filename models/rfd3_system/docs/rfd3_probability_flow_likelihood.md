# RFD3 Probability-Flow Likelihood

`scripts/compute_rfd3_flow_likelihood.py` estimates the coordinate log
likelihood of one supplied structure under the probability-flow ODE implied by
the RFD3 denoiser. The negative log likelihood can be used as a learned energy
for controlled comparisons. It is not a calibrated thermodynamic free energy.

The method follows *Can We Extract Physics-like Energies from Generative
Protein Diffusion Models?* ([paper][diffenergy-paper], [code][diffenergy-code]).
DiffEnergy evaluates an exact low-dimensional divergence for rigid-body
docking. RFD3 has thousands of coordinate dimensions, so this implementation
uses a stochastic Hutchinson trace estimate instead.

## Quantity Being Estimated

Let `x` contain the active RFD3 coordinate slots, `c` contain all fixed
conditioning, and `D(x, sigma, c)` be RFD3's denoised-coordinate prediction.
The EDM denoising interpretation gives the approximate score

```text
score(x, sigma, c) = [D(x, sigma, c) - x] / sigma^2.
```

The corresponding probability-flow ODE, written in the physical RFD3 noise
scale `sigma`, is

```text
dx / d sigma = -sigma * score
             = [x - D(x, sigma, c)] / sigma.
```

For a deterministic flow `f`, probability conservation gives

```text
d log p / d sigma = -divergence(f).
```

The script starts from the supplied structure at `sigma_min`, follows the ODE
to `sigma_max`, and evaluates

```text
estimated log p(x_initial | c)
    = terminal Gaussian log p(x_terminal)
      + integral from sigma_min to sigma_max of divergence(f) d sigma.
```

The sign follows directly from integrating the continuity equation backward:
`log p_terminal - log p_initial` is the negative integrated divergence.

The terminal term is the isotropic Gaussian used by RFD3 initialization:

```text
log p_terminal = -0.5 * [
    ||x_terminal||^2 / sigma_max^2
    + dimension * log(2 * pi * sigma_max^2)
].
```

Fixed atoms are held constant and excluded from the divergence, dimensionality,
and terminal prior. The result is therefore a conditional coordinate density
for the active structure given the fixed context.

## Divergence Estimate

Computing the exact trace of RFD3's coordinate Jacobian would require one
derivative per scalar coordinate. Instead, the script draws fixed Rademacher
vectors whose entries are independently `-1` or `+1` and computes

```text
divergence(f) ~= average over probes of
                 epsilon dot [Jacobian(f) transpose times epsilon].
```

Each term uses one reverse-mode vector-Jacobian product. The model is evaluated
once per ODE right-hand-side call, followed by one backward sweep per probe.
The same probes are reused across the complete trajectory. This makes the
estimated vector field deterministic for RK4 and makes paired conformer scores
substantially easier to compare.

## Input Configuration

The script accepts PDB, CIF, and compressed CIF input. Configuration is a small
YAML file with three sections:

```yaml
checkpoint_path: /weights/rfd3_latest.ckpt

specification:
  ligand: L1
  select_fixed_atoms:
    L1: ALL
  select_buried:
    L1: ALL

likelihood:
  sequence_conditioning: masked
  integration_intervals: 50
  hutchinson_probes: 5
  probe_seed: 123
  precision: float32
  gauge: auto
```

`specification` uses RFD3's normal source-chain atom selectors. For example,
`{L1: ALL, A20: BKBN}` fixes every ligand atom and the backbone of protein
residue A20. Use `select_fixed_atoms: false` to score every coordinate.

`sequence_conditioning` is mandatory:

- `masked` hides protein identities and matches backbone generation.
- `observed` conditions the denoiser on the protein sequence in the file.

The scorer manages `input`, `partial_t`, centering, and exact-structure loading.
Generation fields such as `contig`, `unindex`, `length`, and `symmetry` are
rejected because they can rebuild the system instead of scoring the supplied
coordinates.

With `gauge: auto`, the fixed-context centroid is moved to the origin. If no
atoms are fixed, the active physical-atom centroid is used. This deterministic
translation gauge is necessary because a density that is uniform over all
global translations is not normalizable. `gauge: as_supplied` is available for
diagnostics but makes the terminal Gaussian term dependent on file placement.
No rotational alignment is applied; the isotropic terminal prior is rotationally
invariant and RFD3 is trained with random rigid augmentation.

## Running on CoreHPC

Use the Foundry container in a GPU allocation:

```bash
CONFIG=/project/software/foundry/models/rfd3_system/docs/examples/\
rfd3_flow_likelihood_ligand.yaml

scripts/foundry_exec.sh --gpu \
  env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src" \
  python /project/software/foundry/models/rfd3_system/scripts/compute_rfd3_flow_likelihood.py \
    /project/inputs/example.cif \
    --config "$CONFIG" \
    --output-dir /project/outputs/foundry/rfd3_system/flow_likelihood/example
```

Submit this command through SLURM using `gpu:1` and exclude
`ggpu1-[13-19]`, as required by RFD3. The defaults require 200 denoiser
evaluations and 1,000 reverse sweeps: 50 RK4 intervals times four stages with
five Hutchinson probes. This is intentionally more expensive than generation.

Every run writes a summary JSON and an integration-trace CSV. The CSV records
the sigma interval, RK4-weighted divergence, cumulative density correction,
and active-coordinate norm. Reuse the same checkpoint, conditioning, schedule,
precision, and `probe_seed` for paired comparisons.

## RFD3 State Reconstruction

RFD3's input pipeline is reused for tokenization, conditioning, and atom14
preparation. With masked protein sequence, missing atom14 slots are represented
by virtual atoms initialized from the token representative atom, matching
RFD3's normal canonical padding behavior. These virtual coordinates are active
and included in the density dimension.

Cleaned RFD3 CIFs omit virtual slots. Their exact generated virtual coordinates
cannot be recovered; the script reconstructs them canonically and records this
in the JSON. For future highest-fidelity studies, retain raw outputs with
`cleanup_virtual_atoms=false`. Paired comparisons remain more defensible when
both structures use the same reconstruction policy.

## Interpretation and Limitations

Only compare absolute NLL values when all of the following match:

- active topology and coordinate dimension;
- fixed-atom and sequence conditioning;
- checkpoint, sigma schedule, precision, and probe seed;
- atom14 reconstruction and gauge policy.

For different protein lengths, per-coordinate NLL is available but remains an
exploratory normalization rather than a rigorous free-energy comparison.

The estimate differs from an exact model likelihood for several reasons:

1. RFD3 provides a denoiser, so its score is inferred through the EDM identity.
2. Hutchinson probes approximate rather than exactly compute divergence.
3. RFD3's training noise includes a correlated center-of-mass perturbation that
   is not represented by the scalar isotropic probability-flow ODE.
4. Sparse attention neighbors are selected under `no_grad`, making the learned
   field piecewise differentiable with the selected graph held fixed locally.
5. Integration uses finite `sigma_min` and `sigma_max`, and cleaned structures
   require canonical reconstruction of virtual slots.
6. Learned score fields can be inaccurate or nonconservative away from the
   training distribution, as also discussed by DiffEnergy.

The most meaningful ligand-conformer test is therefore a paired delta:

```text
Delta NLL = NLL(protein | conformer 2) - NLL(protein | conformer 1),
```

computed with identical numerical settings and probes. This can test whether a
single-conformer design is preferentially scored in its intended context and
whether adaptive/equal mixtures are more balanced, without interpreting the
numbers as physical binding free energies.

## References

- [Can We Extract Physics-like Energies from Generative Protein Diffusion Models?][diffenergy-paper]
- [Graylab/DiffEnergy implementation][diffenergy-code]
- [Score-Based Generative Modeling through Stochastic Differential Equations][score-sde]
- [Maximum Likelihood Training of Score-Based Diffusion Models][score-ml]
- Local RFD3 equations: `model/RFD3_diffusion_module.py` and
  `model/inference_sampler.py`.
- Local input/noise assumptions: `transforms/virtual_atoms.py` and
  `transforms/design_transforms.py`.

[diffenergy-paper]: https://www.biorxiv.org/content/10.1101/2025.11.28.690021v4.full
[diffenergy-code]: https://github.com/Graylab/DiffEnergy
[score-sde]: https://arxiv.org/abs/2011.13456
[score-ml]: https://arxiv.org/abs/2101.09258
