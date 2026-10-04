# RFD3 coupling diversity benchmark

The isolated [variation bundle](../software/foundry/models/rfd3_coupling_variants/README.md)
adds named alternatives to the existing equal-weight `rfd3_system` sampler.
Original model sources are unchanged. The [experiment workflow](../pipeline/coupling_diversity_001/README.md)
documents parameters, SLURM commands, scientific success criteria, output
locations and later evidence-dependent decisions.

The principal comparison is useful structural diversity **among paired designs
that pass in both states with one shared A sequence**. The primary motif-free
screen uses 37 conditions, 50 matched seeds per condition, four tied Caliby
sequences per paired backbone, and full local ESMFold2 for both states.

Validation so far: CPU 2126966 passed 20 tests; GPU 2126955 passed exact native
sampler state/RNG parity and all short-trajectory controls, including finite
sequence gradients. Smoke jobs 2126967–2126970 completed one paired backbone,
four tied Caliby candidates, eight ESMFold2 predictions and analysis/plots.
That test parent passed A+B but not A+C. Calibration jobs 2126976/2126977
completed all 15 held-out trajectories. No scientific winner has been
established by these implementation tests.

Source and project templates are versioned in the Foundry fork on
`system_design_v0`. Existing project restore controls include the top-level
pipeline and jobs; the bundle also provides a missing-files-only installer.
Generated outputs remain under `outputs/pipeline/coupling_diversity_001/`.
