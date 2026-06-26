# RFdiffusion3 System Prototype

`rfd3_system` is an isolated research copy of Foundry's `models/rfd3`
implementation. It exists to prototype coupled RFdiffusion3 inference while
leaving the upstream `models/rfd3` directory untouched.

The current prototype implements an approximate shared-chain coupling mode for
two coupled inference tracks:

- track 1: chain A + partner chain B;
- track 2: chain A + partner chain C;
- chain A is maintained as one shared coordinate trajectory;
- B and C remain condition-specific partner contexts.

The implementation is opt-in and should not affect ordinary RFD3 behavior.

## Status

The mode is an approximation, not exact SuperDiff density control. RFD3 exposes
an EDM denoiser-derived coordinate update, not the exact score/SDE interface
required by the SuperDiff AND density-control derivation. The coupled sampler
therefore uses RFD3 denoiser deltas as score-like update proxies and records
`superdiff_exact: false` in output metadata.

Detailed implementation notes, assumptions, limitations, and validation results
are in [docs/superdiff_shared_chain.md](docs/superdiff_shared_chain.md).
The update-proxy, kappa, and residual equations are summarized in
[docs/shared_chain_coupling_math.md](docs/shared_chain_coupling_math.md).
The track-specific motif scaffolding plan and command patterns are documented
in [docs/joint_motif_scaffolding.md](docs/joint_motif_scaffolding.md).
The tied ProteinMPNN post-design workflow for separated A+B and A+C sequence
design is documented in
[docs/tied_proteinmpnn.md](docs/tied_proteinmpnn.md).

## Active Layout

```text
src/rfd3_system/                         Python package for the research copy
configs/                                 Hydra configs used by inference/checkpoint loading
tests/                                   Unit and regression tests for the copied package
docs/superdiff_shared_chain.md           Prototype design and validation notes
docs/shared_chain_coupling_math.md       Coupling equations and diagnostics
docs/joint_motif_scaffolding.md          Track-specific motif scaffolding notes
docs/tied_proteinmpnn.md                 Tied ProteinMPNN sequence-design workflow
docs/examples/superdiff_shared_chain_proxy.yaml
                                          Minimal example override shape
scripts/plot_coupling_diagnostics.py     Post-run Matplotlib diagnostics plotter
scripts/build_tied_mpnn_input.py         Build separated A+B / D+C tied-MPNN inputs
archived/                                Upstream RFD3 docs/assets kept for reference only
```

Files under `archived/` are intentionally not part of the active
`rfd3_system` workflow and should not be used by the prototype.

## Running the Prototype

Run this package explicitly through `PYTHONPATH` so it does not shadow or
modify the normal Foundry `rfd3` package:

```bash
PYTHONPATH=/path/to/foundry/models/rfd3_system/src:/path/to/foundry/src \
python -m rfd3_system.cli design \
  coupling_mode=superdiff_shared_chain \
  inference_sampler.kind=superdiff_shared_chain \
  shared_chain_id=A \
  complex_1_partners='[B]' \
  complex_2_partners='[C]' \
  ...
```

On this project, CoreHPC runs should use the project wrapper and SLURM job
templates rather than invoking the container directly:

```bash
sbatch jobs/rfd3_system_a90_b80_c100_smoke.sbatch
```

The prepared A90/B80/C100 test writes outputs under:

```text
outputs/foundry/rfd3_system/a90_b80_c100_default_steps_n5_<jobid>/
```

The helper script sets `global_prefix=rfd3sys_a90_b80_c100`, so new output
files start with names such as:

```text
rfd3sys_a90_b80_c100_0_track1_model_0.cif.gz
rfd3sys_a90_b80_c100_0_merged_denoised_model_0.cif.gz
```

## Output Behavior

For A+B/A+C coupled runs, the engine writes:

- track 1 outputs containing A+B;
- track 2 outputs containing A+C;
- merged outputs containing A+B+C, controlled by
  `merged_output_policy=track1|track2|both|none`;
- optional track-specific and merged trajectory files;
- metadata with coupling settings and per-step proxy diagnostics.

The physical RFD3 noise scale `t_hat` is recorded in the JSON diagnostics for
runs that need noise-level interpretation.

To create PNG plots after a run, use the Matplotlib post-processing script with
an environment that has Matplotlib installed. On CoreHPC, `envs/esm` is already
validated for this:

```bash
envs/esm/bin/python \
  software/foundry/models/rfd3_system/scripts/plot_coupling_diagnostics.py \
  outputs/foundry/rfd3_system/<run_dir>
```

The script reads any coupled track or merged JSON metadata file for each
diffusion batch and writes PNG plots next to the chosen JSON:

```text
*_kappa.png
*_proxy_residual.png
```

Track, merged, and repeated `model_N` JSON files from the same diffusion batch
contain the same coupling diagnostics. The default output is one batch-level
kappa PNG and one batch-level proxy-residual PNG, even when
`merged_output_policy=none`. Use `--model-index N` to create a one-off plot set
for a single generated model.

During coupled inference, the sampler logs periodic progress lines with the
step count, normalized `t`, `t_hat`, kappa mean/min/max, and mean absolute
proxy residual. These messages are intended for SLURM log monitoring during
longer default-step runs.

RFD3's normal per-complex formatting can compact a split two-chain view to A+B,
even when the original global source chains were A+C. `rfd3_system` relabels
track 2's non-shared partner chains back to `complex_2_partners` during output
formatting so final files preserve the intended A+B, A+C, and merged A+B+C
chain organization. This is an output-annotation step only; it does not change
the sampled coordinate tensors or denoising behavior.

## Validation

CoreHPC GPU job `703094` completed the prepared A90/B80/C100 default-step
coupled test. Final CIF validation found:

- track 1 residue counts: A=90, B=80;
- track 2 residue counts: A=90, C=100;
- merged residue counts: A=90, B=80, C=100;
- shared-chain A coordinates exactly matching between track 1 and track 2 for
  all three generated models.

The corresponding outputs are under:

```text
outputs/foundry/rfd3_system/a90_b80_c100_default_steps_703094/
```

That validation run predated the filename cleanup, so its files still use the
older `_0_...` prefix.

## Development Rules

- Keep all edits localized to `models/rfd3_system`.
- Do not modify the original `models/rfd3` implementation for this prototype.
- Keep ordinary upstream RFD3 reference material in `archived/` unless it has
  been rewritten to describe `rfd3_system` specifically.
- Keep generated structures, logs, model weights, and runtime caches outside
  the Foundry source tree.
- Develop and commit on Wynton, push to GitHub, then pull on CoreHPC before
  running SLURM validation jobs.
