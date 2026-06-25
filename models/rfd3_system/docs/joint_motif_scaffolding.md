# Joint Motif Scaffolding Plan

This note documents the `rfd3_system` extension for motif scaffolding in the
coupled A+B / A+C shared-chain prototype.

The feature is intended for cases where any of the designed proteins A, B, or C
contains fixed structural motifs, including the harder case where shared chain A
has the same fixed backbone placement in both tracks but different chemistry or
sequence in each context.

Example target:

- track 1 input: A+B, with an A-chain helix motif sequence `RRAS`;
- track 2 input: A+C, with the same A-chain helix backbone but sequence `RRAE`
  or `RRA-SEP`;
- the non-motif part of A is shared and coupled;
- the fixed motif residues are context-specific constraints and are excluded
  from the shared-chain kappa solve.

## Scope

This is an inference-time motif scaffolding workflow. It does not modify
training and does not change the original `models/rfd3` package.

The implemented behavior is:

- accept explicit track-specific input structures for A+B and A+C;
- allow fixed shared-chain A motif residues to differ chemically between
  tracks;
- exclude fixed shared-chain A motif residues from the SuperDiff-like proxy
  kappa solve and from the shared coordinate update;
- require all non-fixed shared-chain A residues to have identical residue and
  atom identity between tracks;
- allow independent fixed motifs on B and C using ordinary RFD3 selection
  semantics;
- provide `merged_output_policy=track1|track2|both|none` so merged A+B+C output
  can use either track's fixed shared-chain A chemistry.

This workflow treats PTM residues such as phosphoserine as fixed input motif
chemistry. It does not add a de novo PTM sequence-generation mode.

## Track-Specific Input Mode

Ordinary de novo coupled runs still build one ABC source and split it into A+B
and A+C internally. Motif scaffolding uses a separate track-specific input mode:

```text
track_1_specification.input = /project/inputs/track1_AB_motifs.cif
track_2_specification.input = /project/inputs/track2_AC_motifs.cif
```

Both track inputs must be supplied together. Supplying only one track input is
an error.

Each input is processed through the normal RFD3 input parser, so `contig`,
`select_fixed_atoms`, `select_unfixed_sequence`, guidepost settings, and other
per-track specification fields are still ordinary RFD3 controls. If neither
track specifies `ori_token` or `infer_ori_strategy`, the engine computes one
common origin from both input structures and applies it to both tracks so the
two A-chain coordinate frames remain comparable.

## Fixed Shared A Motifs

Shared-chain motif residues are identified from the per-track
`is_motif_atom_with_fixed_coord` masks produced by the RFD3 pipeline. A
shared-chain residue is considered a fixed motif residue when any atom in that
residue has fixed coordinates.

For shared chain A, the mapper applies these rules residue by residue:

- fixed in both tracks: the residue is treated as fixed motif context, may have
  different residue or atom identity in track 1 and track 2, and is excluded
  from kappa and the shared update;
- fixed in only one track: error, because one track would be trying to move a
  residue that the other treats as a fixed motif;
- fixed coordinates but unfixed sequence in either track: error, because
  chemically different motif variants are only well-defined when the motif
  residue identity is fixed in each track;
- not fixed in either track: residue and atom identity must match exactly
  between tracks, and all atoms are paired for the shared update.

This means a fixed A-chain `SER` motif residue in track 1 and a fixed A-chain
`SEP` motif residue in track 2 are allowed, but neither residue participates in
the coupled A update. They influence denoising as context through their
respective track-specific denoiser calls.

For ordinary generated shared-chain residues, tracks are paired by the generated
`res_id`. For fixed shared-chain motifs and unindexed guideposts, tracks are
paired by the RFD3 `src_component` annotation when available, for example
`A237`. This avoids confusing RFD3's temporary guidepost numbering with source
motif identity. A generated residue with `res_id=237` and a fixed guidepost with
`src_component=A237` are kept in separate typed key spaces, so they do not
collide.

## Kappa Solve With Motifs

The approximate coupling still computes:

```text
delta_mix = kappa * delta(track 1) + (1-kappa) * delta(track 2)
```

but `delta(track 1)` and `delta(track 2)` are now restricted to the paired
non-fixed shared-chain atoms. Fixed A motifs are not part of the proxy vectors,
so they do not directly determine kappa. They still affect kappa indirectly by
conditioning each track's denoiser prediction for the movable portion of A.

This differs from exact SuperDiff in the same way as the base prototype:
RFD3's denoiser deltas are used as score-like update proxies, not exact
Itô-density vector fields. See `shared_chain_coupling_math.md` for the proxy
equations.

## B and C Motifs

B and C are not shared between tracks. Their fixed motifs are handled by normal
RFD3 masks inside each track:

- B motifs are specified only in `track_1_specification`;
- C motifs are specified only in `track_2_specification`;
- B and C updates remain independent except for their indirect effect on the
  shared A denoiser calls.

## Selection Guidance

String selections such as `A10-13` fix all atoms in those residues in the
native RFD3 selection parser. Dictionary selections can specify atom subsets:

```yaml
select_fixed_atoms:
  A10-13: ALL
  A45-49: BKBN
  B20-25: ALL
```

Use residue/all-atom fixed selections for chemically variant A motifs such as
`SER` versus `SEP`. Backbone-only fixed selections are useful for ordinary
canonical backbone scaffolding, but a PTM side chain should generally be fixed
as motif context if the PTM chemistry is part of the design condition.

When using `unindex` for guidepost-style motif scaffolding across multiple
source chains, include `/0` between source-chain groups. For example, use
`A237-240,/0,B56,B129-133` rather than `A237-240,B56,B129-133`. Without the
chain break, RFD3 treats the later guideposts as part of the same unindexed
guidepost chain, which can make B-source guideposts appear in the shared A
chain during `rfd3_system` validation.

Do not use `select_unfixed_sequence=true` when fixed motif sequences must be
preserved. Instead, explicitly select only the non-motif designable ranges.
For example, if A motifs are A12-15 and A48-52, B has motif B20-25, and C has
motif C30-35:

```bash
+track_1_specification.select_fixed_atoms='A12-15,A48-52,B20-25' \
+track_1_specification.select_unfixed_sequence='A1-11,A16-47,A53-90,B1-19,B26-80' \
+track_2_specification.select_fixed_atoms='A12-15,A48-52,C30-35' \
+track_2_specification.select_unfixed_sequence='A1-11,A16-47,A53-90,C1-29,C36-100'
```

That pattern keeps the motif coordinates and motif identities fixed while
allowing the remaining scaffold residues to be designed.

## Example Command Shape

This example scaffolds two A motifs, one B motif, and one C motif. Track 1 and
track 2 may contain chemically different fixed A motifs at the same residue
numbers.

```bash
scripts/foundry_exec.sh --gpu \
  env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src" \
  python -m rfd3_system.cli design \
    inputs=null \
    "out_dir=/project/outputs/foundry/rfd3_system/joint_motif_test" \
    ckpt_path=/weights/rfd3_latest.ckpt \
    coupling_mode=superdiff_shared_chain \
    inference_sampler.kind=superdiff_shared_chain \
    shared_chain_id=A \
    "complex_1_partners=[B]" \
    "complex_2_partners=[C]" \
    merged_output_policy=both \
    "+track_1_specification.input=/project/inputs/track1_AB_motifs.cif" \
    "+track_1_specification.contig='A1-90,/0,B1-80'" \
    "+track_1_specification.length=170" \
    "+track_1_specification.select_fixed_atoms='A12-15,A48-52,B20-25'" \
    "+track_1_specification.select_unfixed_sequence='A1-11,A16-47,A53-90,B1-19,B26-80'" \
    "+track_2_specification.input=/project/inputs/track2_AC_motifs.cif" \
    "+track_2_specification.contig='A1-90,/0,C1-100'" \
    "+track_2_specification.length=190" \
    "+track_2_specification.select_fixed_atoms='A12-15,A48-52,C30-35'" \
    "+track_2_specification.select_unfixed_sequence='A1-11,A16-47,A53-90,C1-29,C36-100'"
```

The leading `+` on a Hydra override adds a key that is not present in the base
config. Quote values that contain commas, brackets, or shell-sensitive
characters so Hydra receives the intended string or list.

## Merged Output Policy

The fixed A motif chemistry can differ between tracks, so a single merged A+B+C
file must choose which track supplies chain A:

- `merged_output_policy=track1`: write one merged output using A from track 1;
- `merged_output_policy=track2`: write one merged output using A from track 2;
- `merged_output_policy=both`: write both merged variants;
- `merged_output_policy=none`: write only track-specific A+B and A+C outputs.

The default is `track1` for backward compatibility with the original prototype.

Track-specific outputs are always written. The merged files are convenience
views for inspection and downstream analysis; they do not feed back into the
denoising process.

## Validation Expectations

Before using this mode for production runs, validate with small motif examples:

- chemically identical A motifs fixed in both tracks;
- chemically different A motifs fixed in both tracks;
- one or more B/C fixed motifs;
- `merged_output_policy=track1`, `track2`, and `both`;
- expected failure when a shared A motif is fixed in only one track;
- expected failure when a fixed shared A motif is included in
  `select_unfixed_sequence`;
- expected failure when a non-fixed shared A residue differs chemically between
  tracks.

The output JSON records `shared_atom_mapping`, including the number of paired
shared update atoms and a list of excluded fixed shared residues.
