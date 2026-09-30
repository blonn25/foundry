# Proxy Designability Grid 001

This experiment compares `proxy_norm_weight` and
`proxy_kappa_regularization_rho` by measuring whether tied ProteinMPNN
sequences recover both structures in a generated shared-A system.

## Design

- Track 1: A90+B80
- Track 2: A90+C100
- Binder sampler: `step_scale=3`, `gamma_0=0.2`
- Both tracks: `is_non_loopy=true`
- Kappa solve atoms: `ALL`
- Kappa clamp: `[0, 1]`
- Backbones per setting: 25
- ProteinMPNN sequences per backbone: 8
- ProteinMPNN temperature: 0.1
- ProteinMPNN backbone noise: 0.0 Angstrom
- Structure predictor: ESMFold2-Fast

The shared A sequence is tied between the A+B and D+C ProteinMPNN contexts.
No RFD3-predicted residue identity is retained.  The two complexes are placed
100 Angstroms apart during sequence design so B and C do not condition each
other.

## Grid

```text
proxy_norm_weight = 0.5, 0.625, 0.75, 0.875, 1.0
rho = 0, 1e-5, 1e-4, 1e-3
```

At `proxy_norm_weight=0.5`, kappa is always 0.5, so rho is irrelevant and only
the `rho=0` averaging control is run. The current solver returns this identity
directly so bfloat16 cancellation cannot perturb the equal-mixing control. The
resulting grid contains 17 settings and 425 coupled backbones.

## Primary Metric

For each ProteinMPNN sequence index `m`, ESMFold2-Fast predicts A+B and D+C.
Each prediction is optimally superimposed on its corresponding RFD3 track
using every matched C-alpha atom:

```text
AB_RMSD[m] = full-complex C-alpha RMSD to RFD3 track 1
AC_RMSD[m] = full-complex C-alpha RMSD to RFD3 track 2, mapping D to A

joint_score[m] = max(AB_RMSD[m], AC_RMSD[m])
joint_minimax_RMSD = min over the 8 joint_score values

joint_designable = joint_minimax_RMSD < 2.0 Angstrom
```

The same tied sequence must pass for both contexts.  Separate AB and AC pass
indicators are also reported so independent pair recovery remains visible.

## Interpretation

The experiment ranks only the tested parameter settings.  The primary ranking
uses joint designability rate, followed by median joint minimax RMSD and
independent pair pass rate.  Wilson intervals and paired bootstrap differences
from the unregularized `w=1, rho=0` baseline quantify uncertainty.

Nonzero rho and lower norm weights intentionally relax the original proxy
equalization behavior.  Better designability would be empirical evidence for a
more useful trajectory, not evidence that the proxy is an exact SuperDiff
density estimator.

Results and CoreHPC job identifiers will be added after the campaign finishes.
