# rfd3_system_early_cut

An isolated copy of `models/rfd3_system` at Foundry commit
`5e51cf02c4ccb03a19398e908f5048f4da09f1c2`, with configurable release of
shared-chain coordinate coupling. The original model is unchanged.

The denoiser, checkpoint, recycling, motif handling and pre-release proxy
coupling are inherited unchanged. After release, each track uses its own
update while corresponding formerly shared atoms continue receiving identical
churn noise. This remains the original approximate denoiser-delta method.

See [usage and diagnostics](docs/early_cut.md). The copied documentation and
scripts retain the baseline model's other capabilities; the early-cut guide
is authoritative for release behavior and defaults.

CoreHPC launch helpers and CPU/GPU validation jobs are preserved in
[examples/project](examples/project/) as well as the project's `scripts/`
and `jobs/` directories. No extra weights or environments are needed.
