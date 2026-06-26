# Tied ProteinMPNN Post-Design Workflow

This workflow prepares ProteinMPNN inputs from paired `rfd3_system` A+B and A+C
track outputs.  It is intended for sequence redesign after coupled
`rfd3_system` backbone generation, while keeping the two partner contexts
separated during ProteinMPNN graph construction.

## Combined Input Geometry

For each rfd3_system diffusion-batch model, the helper script builds one
combined CIF with four chains:

- `A`: shared chain A from track 2, using the SER motif variant;
- `B`: partner chain B from track 1;
- `D`: a translated copy of track 2 chain A, renamed from A to D;
- `C`: track 2 partner chain C, translated with D.

The D+C complex is translated 50 Angstroms along the x-axis by default.  This
keeps B and C outside each other's ProteinMPNN neighborhood while allowing one
ProteinMPNN decode to apply tied sequence groups across the two A copies.

The helper validates that track 1 and track 2 shared-chain A backbone atoms are
already colocated before mixing track 2 A with track 1 B.  If that check fails,
the script exits instead of silently writing a geometrically inconsistent A+B
complex.

## Fixed And Tied Residues

Fixed residues are read from each rfd3_system output JSON
`diffused_index_map`, because indexed motifs and unindexed guideposts can be
renumbered in the final generated output.

Default fixed source residues are:

- shared A motif: `A237`, `A238`, `A240`, fixed in both A and D;
- B guideposts: `B56`, `B129`, `B130`, `B133`, fixed in B;
- C: no fixed residues.

All non-fixed A/D residue positions are tied with explicit ProteinMPNN
`symmetry_residues` groups such as `["A1", "D1"]`.  Explicit residue groups are
used instead of `homo_oligomer_chains` so fixed A motif positions can be left
out of the tied decoding constraints.

## Helper Script

Run the helper inside the Foundry container so it uses the same AtomWorks and
Biotite versions as the rest of Foundry:

```bash
scripts/foundry_exec.sh \
  env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src" \
  python software/foundry/models/rfd3_system/scripts/build_tied_mpnn_input.py \
    /project/outputs/foundry/rfd3_system/<rfd3_system_run_dir> \
    --out-dir /project/outputs/foundry/mpnn/<mpnn_run_dir> \
    --name-prefix tied_ab_ac \
    --batch-size 8 \
    --number-of-batches 1 \
    --temperature 0.1
```

The helper writes:

- `combined_inputs/*_tied_mpnn_model_<i>.cif.gz`;
- `proteinmpnn_config.json`;
- `tied_mpnn_manifest.json`.

The generated config uses Foundry's bundled original ProteinMPNN checkpoint:

```text
checkpoint_path=/weights/proteinmpnn_v_48_020.pt
is_legacy_weights=true
```

`is_legacy_weights=true` is required for this checkpoint because it is in the
original ProteinMPNN weight format, not the newer Foundry-native checkpoint
format.

## ProteinMPNN Command

After the helper has written `proteinmpnn_config.json`, run:

```bash
scripts/foundry_exec.sh \
  env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/project/software/foundry/models/mpnn/src:/project/software/foundry/src" \
  python -m mpnn.inference \
    --config_json /project/outputs/foundry/mpnn/<mpnn_run_dir>/proteinmpnn_config.json
```

ProteinMPNN outputs should remain under `outputs/foundry/mpnn/`.  Do not write
sequence-design outputs into the Foundry source tree.
