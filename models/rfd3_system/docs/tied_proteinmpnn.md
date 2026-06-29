# Tied Sequence-Design Post-Design Workflow

This workflow prepares sequence-design inputs from paired `rfd3_system` A+B and
A+C track outputs.  It is intended for sequence redesign after coupled
`rfd3_system` backbone generation, while keeping the two partner contexts
separated during downstream graph construction.

## Combined Input Geometry

For each rfd3_system diffusion-batch model, the helper script builds one
combined structure with four chains:

- `A`: shared chain A from track 2, using the SER motif variant;
- `B`: partner chain B from track 1;
- `D`: a translated copy of track 2 chain A, renamed from A to D;
- `C`: track 2 partner chain C, translated with D.

The D+C complex is translated 100 Angstroms along the x-axis by default.  This
keeps B and C outside each other's sequence-design neighborhood while allowing
one decode to apply tied sequence groups across the two A copies.

The helper validates that track 1 and track 2 shared-chain A backbone atoms are
already colocated before mixing track 2 A with track 1 B.  If that check fails,
the script exits instead of silently writing a geometrically inconsistent A+B
complex.

Before writing the combined structure, the helper drops atoms whose coordinates are
not finite.  This is necessary because AtomWorks may reconstruct missing
template atoms, especially hydrogens or terminal atoms, with `NaN` coordinates
when it parses sparse rfd3_system outputs.  Those atoms are not real generated
coordinates, and writing them explicitly can break PyMOL and downstream
readers.  The number of dropped atoms is recorded in the generated manifest
under `combined_atom_filter`.

## Fixed And Tied Residues

Fixed residues are read from each rfd3_system output JSON
`diffused_index_map`, because indexed motifs and unindexed guideposts can be
renumbered in the final generated output.

Default fixed source residues are:

- shared A motif: `A237`, `A238`, `A240`, fixed in both A and D;
- B guideposts: `B56`, `B129`, `B130`, `B133`, fixed in B;
- C: no fixed residues.

All non-fixed A/D residue positions are tied.  In ProteinMPNN mode, ties are
written as explicit `symmetry_residues` groups such as `["A1", "D1"]`.  In
Caliby mode, ties are written as `symmetry_pos` groups such as `A1,D1|A2,D2`.
Explicit residue groups are used so fixed A motif positions can be left out of
the tied decoding constraints.

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
    --prepare-for mpnn \
    --name-prefix tied_ab_ac \
    --batch-size 8 \
    --number-of-batches 1 \
    --temperature 0.1
```

`--prepare-for mpnn` is the default and writes:

- `combined_inputs/*_tied_mpnn_model_<i>.cif.gz`;
- `proteinmpnn_config.json`;
- `tied_mpnn_manifest.json`.

The generated ProteinMPNN config writes both FASTA sequence outputs and
redesigned CIF structures by default.  Add `--no-write-structures` if you only
want FASTA sequence outputs.

The generated config uses Foundry's bundled original ProteinMPNN checkpoint:

```text
checkpoint_path=/weights/proteinmpnn_v_48_020.pt
is_legacy_weights=true
```

`is_legacy_weights=true` is required for this checkpoint because it is in the
original ProteinMPNN weight format, not the newer Foundry-native checkpoint
format.

`--prepare-for caliby` writes Caliby-native inputs:

- `combined_inputs/*_tied_caliby_model_<i>.pdb`;
- `caliby_constraints.csv`;
- `tied_caliby_manifest.json`.

The Caliby PDBs are written directly from the combined AtomWorks `AtomArray`
with `to_pdb_string()`.  The CSV uses Caliby's native positional-constraint
columns: `pdb_key`, `fixed_pos_seq`, and `symmetry_pos`.  Sidechains are not
fixed by default.

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

## Caliby Command

After the helper has written `caliby_constraints.csv` and the PDB inputs, run
Caliby through its native Hydra CLI:

```bash
scripts/caliby_exec.sh \
  python software/caliby/caliby/eval/sampling/seq_des.py \
    ckpt_name_or_path=soluble_caliby_v1 \
    input_cfg.pdb_dir=/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/caliby/<caliby_run_dir>/combined_inputs \
    pos_constraint_csv=/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/caliby/<caliby_run_dir>/caliby_constraints.csv \
    seq_des_cfg.atom_mpnn.sampling_cfg=/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/software/caliby/caliby/configs/seq_des/inference.yaml \
    sampling_cfg_overrides.num_seqs_per_pdb=3 \
    sampling_cfg_overrides.batch_size=3 \
    hydra.run.dir=/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/caliby/<caliby_run_dir>/hydra_outputs \
    out_dir=/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/caliby/<caliby_run_dir>
```

`soluble_caliby_v1` is used for this workflow because it was trained on
monomers and interfaces.  Do not override `input_cfg.pdb_name_list`; Caliby will
use all generated PDBs in `combined_inputs/`.

## ESMFold2 Command

After Caliby writes `seq_des_outputs.csv`, fold the designed sequences with the
local/offline ESMFold2 installation:

```bash
scripts/esm_exec.sh \
  python software/foundry/models/rfd3_system/scripts/fold_caliby_esmfold2.py \
    /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/caliby/<caliby_run_dir> \
    --out-dir /mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design/outputs/esm/<esmfold2_run_dir> \
    --model esmfold2 \
    --top-n-per-model all \
    --sep-source-residue A240
```

The helper reads Caliby's `A:B:C:D` sequence order and creates two ESMFold2
folds for each selected Caliby row:

- `A+B`, where chain A carries an ESMFold2 `SEP` modification at the
  A240-derived final residue position;
- `D+C`, where D is the tied copy of A and remains the unphosphorylated SER
  variant.

Use `--top-n-per-model <N>` to fold only the best N Caliby rows per rfd3_system
model by ascending Caliby `U`.  Omit ESMFold2 inference overrides to use the
model defaults, or pass options such as `--num-loops`, `--num-sampling-steps`,
and `--num-diffusion-samples` for faster validation runs.
