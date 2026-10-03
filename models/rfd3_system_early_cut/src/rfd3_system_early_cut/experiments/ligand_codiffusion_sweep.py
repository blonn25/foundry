"""Co-diffuse explicit ligand atom pairs and compare with the fixed-ligand sweep."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .substrate_common import (
    StateRecorder, digest, input_information, read_json, require, source_hashes as base_hashes, sha256, write_csv, write_json,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "experiments/ligand_codiffusion_sweep/config.json"


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    extra = [Path(__file__), package/'model/RFD3.py', package/'system/ligand_coupling.py']
    return base_hashes() | {str(p.relative_to(package)): sha256(p) for p in extra}


class LigandStateRecorder(StateRecorder):
    def __init__(self):
        super().__init__()
        self.mapped = []
        self.ligands = {1: [], 2: []}
        self.ligand_initial = None
        self.ligand_noise = []

    def __call__(self, event):
        super().__call__(event)
        mapped = np.stack([event[f'mapped_ligand_{t}'].numpy()[0] for t in (1, 2)])
        require(np.isfinite(mapped).all(), 'Nonfinite mapped ligand state')
        self.mapped.append(mapped.copy())
        for track in (1, 2):
            coords = event[f'ligand_{track}'].numpy()[0]
            require(np.isfinite(coords).all(), 'Nonfinite ligand state')
            self.ligands[track].append(coords.copy())
        if event['completed_updates'] == 0:
            require(np.array_equal(mapped[0], mapped[1]), 'Ligand initialization mismatch')
            self.ligand_initial = [digest(event[f'ligand_{t}'].numpy()) for t in (1, 2)]
        else:
            require(np.array_equal(event['mapped_ligand_noise_1'].numpy(), event['mapped_ligand_noise_2'].numpy()),
                    'Mapped ligand churn differs between tracks')
            self.ligand_noise.append([digest(event[f'ligand_noise_{t}'].numpy()) for t in (1, 2)])

    def audit(self):
        result = super().audit()
        result.pop('maximum_fixed_coordinate_drift')  # No fixed atoms in this experiment.
        return result | dict(ligand_initial_sha256=self.ligand_initial, ligand_churn_sha256=self.ligand_noise,
                             protein_noise_matches_fixed_baseline=True)

    def save(self, directory):
        super().save(directory)
        np.savez_compressed(directory/'ligand_states.npz', mapped=np.stack(self.mapped),
                            track_1=np.stack(self.ligands[1]), track_2=np.stack(self.ligands[2]))


def prepare(project, root, config_path):
    config = read_json(config_path)
    info = input_information(project, config)
    resolved = dict(config, **info, source_hashes=source_hashes(),
                    foundry_revision=os.environ["FOUNDRY_REVISION"])
    rows = [dict(condition=f"coupled_{percent:03d}", coupled_fraction=percent/100,
                 coupled_updates=int(percent*199//100), seed=seed,
                 track_1="4MU-Ac", track_2="4MU-Bu")
            for seed in config["seeds"] for percent in config["coupled_percentages"]]
    require(len(rows) == 210 and len(set((r["condition"], r["seed"]) for r in rows)) == 210,
            "The manifest must contain 210 unique paired trajectories")
    if root.exists():
        require(read_json(root / "resolved_config.json") == resolved,
                "Existing immutable configuration differs; use a fresh experiment directory")
        require(read_json(root / "manifest.json") == rows, "Manifest changed")
    else:
        root.mkdir(parents=True)
        write_json(root / "resolved_config.json", resolved)
        write_json(root / "manifest.json", rows)
        write_csv(root / "manifest.csv", rows)
    return resolved


def track_spec(project, config, name):
    return dict(input=str(project / config["input_dir"] / f"{name}.cif"),
                length=config["protein_length"], ligand=config["ligands"][name]["resname"],
                select_fixed_atoms=False, ori_token=config["origin"])


def make_engine(project, config, pipeline_only=False):
    from rfd3_system_early_cut.engine import RFD3InferenceConfig, RFD3InferenceEngine

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    options = asdict(RFD3InferenceConfig(
        ckpt_path=config["checkpoint"], diffusion_batch_size=1, seed=0,
        inference_sampler=config["sampler"], dump_trajectories=False, skip_existing=False,
        prevalidate_inputs=False, coupling_mode="superdiff_shared_chain",
        complex_1_partners=["L"], complex_2_partners=["L"], merged_output_policy="none",
        coupled_ligand_atom_pairs=config["coupled_ligand_atom_pairs"],
        track_1_specification=track_spec(project, config, "acetate"),
        track_2_specification=track_spec(project, config, "butyrate")))
    engine = RFD3InferenceEngine(**options)
    # Fabric bf16-mixed would quantize ligand input coordinates. Autocast only
    # neural operations below, retaining exactly the prepared FP32 geometry.
    engine._assign_override("trainer.precision", "32-true")
    engine.transform_overrides["residue_cache_dir"] = None
    if pipeline_only:
        checkpoint = torch.load(engine.ckpt_path, map_location="cpu", weights_only=False)
        cfg = engine._override_checkpoint_config(checkpoint["train_cfg"])
        engine._construct_pipeline(cfg)
    else:
        engine.initialize()
    return engine


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_inputs(engine, config):
    from rfd3_system_early_cut.system.chains import build_shared_update_atom_map, build_shared_ca_atom_map

    reset_seed(101)
    specs = engine._build_coupled_track_input_specs({})
    examples = [engine.pipeline(spec.to_pipeline_input(example_id=f"validate_{i}"))
                for i, spec in enumerate(specs)]
    masks = []
    seq_masks = []
    for example, name in zip(examples, config["track_ligands"]):
        atoms, feats = example["atom_array"], example["feats"]
        protein = atoms.is_protein.astype(bool)
        fixed = feats["is_motif_atom_with_fixed_coord"].numpy().astype(bool)
        fixed_seq = feats["is_motif_atom_with_fixed_seq"].numpy().astype(bool)
        require(len(set(atoms.res_id[protein])) == 120, "Expected 120 generated protein residues")
        require(not fixed[protein].any() and not fixed_seq[protein].any(), "Protein must be fully designable")
        require(not fixed.any() and fixed_seq[~protein].all(), "Ligand must have free coordinates and fixed chemistry")
        require(set(atoms.gt_atom_name[~protein]) == set(config["ligands"][name]["atom_names"]),
                "Native concatenation changed ligand atoms")
        coords = example["coord_atom_lvl_to_be_noised"].numpy()[0]
        require(coords.dtype == np.float32, "Coordinates must be FP32 before device conversion")
        expected = np.array([config["ligands"][name]["coordinates"][str(a)]
                             for a in atoms.gt_atom_name[~protein]]) - config["origin"]
        require(np.count_nonzero(coords) == 0, "All co-diffused coordinates must start at origin before noise")
        for field in ("ref_mask", "ref_charge", "ref_pos"):
            require(not torch.any(feats[field][protein] != 0), "Generated protein exposes reference chemistry")
        require(not atoms.is_motif_atom_unindexed.any(), "Unexpected catalytic motif")
        masks.append(fixed)
        seq_masks.append(fixed_seq)
    mapping = build_shared_update_atom_map(examples[0]["atom_array"], examples[1]["atom_array"], "A",
                                          masks[0], masks[1], seq_masks[0], seq_masks[1])
    ca = build_shared_ca_atom_map(examples[0]["atom_array"], examples[1]["atom_array"], "A", *masks)
    require(len(ca["all_1"]) == len(ca["movable_1"]) == 120, "Incorrect shared CA map")
    engine._validate_shared_initial_coordinates(examples[0], examples[1], mapping.update_indices_1, mapping.update_indices_2)
    from rfd3_system_early_cut.system.ligand_coupling import resolve_ligand_pairs
    ligand_indices, ligand_metadata = resolve_ligand_pairs(config['coupled_ligand_atom_pairs'],
        specs[0].atom_array_input, specs[1].atom_array_input, *examples)
    require(len(ligand_indices['mapped_1']) == 16, 'Pilot must use the explicit 16-atom map')
    return dict(protein_length=120, ligand_heavy_atoms=[16, 18], shared_atom_count=len(mapping.update_indices_1),
                common_atom_count=16, origin=config["origin"], validated=True, ligand_mapping=ligand_metadata,
                ligand_atom_names={str(i): list(ex['atom_array'].gt_atom_name[ex['atom_array'].is_ligand.astype(bool)])
                                   for i, ex in enumerate(examples, 1)})


def validate_pair(meta, recorder, percent):
    k = percent*199//100
    states = np.stack(recorder.ca)
    require(states.shape == (200, 2, 120, 3), "Incomplete CA trajectory")
    require(meta["cutoff"]["coupled_update_count"] == k, "Wrong release boundary")
    require(np.array_equal(states[:k+1, 0], states[:k+1, 1]), "Shared CA states differ before release")
    expected = [True]*k + [False]*(199-k)
    require(meta["diagnostics"]["coupling_active"] == expected, "Wrong coupling flags")
    kappas = meta["diagnostics"]["kappa"]
    require(all(row == [0.5] for row in kappas[:k]), "Equal mixing was not exact")
    require(all(row == [None] for row in kappas[k:]), "Proxy solver remained active after release")
    require(np.max(meta["diagnostics"]["shared_noise_max_abs_difference"]) == 0,
            "Noise differs between tracks")
    measured = np.sqrt(np.mean(np.sum((states[:, 0]-states[:, 1])**2, axis=-1), axis=-1))
    require(np.allclose(measured, np.asarray(meta["shared_chain_state"]["all_ca_rmsd"])[:, 0],
                        atol=2e-5, rtol=2e-6), "Observer does not match native state diagnostic")
    ligand = np.stack(recorder.mapped)
    require(ligand.shape == (200, 2, 16, 3), 'Incomplete ligand trajectory')
    require(np.array_equal(ligand[:k+1, 0], ligand[:k+1, 1]), 'Ligand diverged before release')
    rmsd = np.sqrt(np.mean(np.sum((ligand[:, 0]-ligand[:, 1])**2, axis=-1), axis=-1))
    require(np.allclose(rmsd, np.asarray(meta['shared_ligand_state']['mapped_atom_rmsd'])[:, 0],
                       atol=2e-5, rtol=2e-6), 'Ligand recorder differs from native diagnostic')
    require(np.max(meta['shared_ligand_state']['shared_noise_max_abs_difference']) == 0, 'Ligand noise mismatch')
    if percent == 100:
        require(meta["cutoff"]["release_pre_churn_sigma"] is None, "Full coupling has no release sigma")


def run_seed(project, root, seed, attempt):
    config = read_json(root / "resolved_config.json")
    require(source_hashes() == config["source_hashes"], "Generation sources differ from immutable manifest")
    require(seed in config["seeds"], "Unrequested seed")
    require(input_information(project, config)["ligands"] == config["ligands"], "Inputs changed")
    directory = root / f"seed_{seed}_{attempt}"
    directory.mkdir(exist_ok=False)
    t0 = time.monotonic()
    try:
        engine = make_engine(project, config)
        input_report = validate_inputs(engine, config)
        write_json(directory / "input_validation.json", input_report)
        baseline_dirs = [p for p in (project/config['fixed_baseline']).glob(f'seed_{seed}_*') if (p/'_COMPLETE').exists()]
        require(len(baseline_dirs) == 1, 'Missing or ambiguous fixed-ligand baseline')
        fixed_audit = read_json(baseline_dirs[0]/'coupled_100/noise_audit.json')
        model = engine._get_forward_coupled_model(engine.trainer.state["model"])
        sampler = model.inference_sampler.sampler
        baseline = None
        rows = []
        for percent in config["coupled_percentages"]:
            condition = f"coupled_{percent:03d}"
            pair_dir = directory / condition
            pair_dir.mkdir()
            sampler.coupling_cut_fraction = percent/100
            sampler.coupling_cut_sigma = None
            recorder = LigandStateRecorder()
            sampler.state_observer = recorder
            reset_seed(seed)
            print(f"START seed={seed} condition={condition}", flush=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = engine._model_forward_coupled("substrates", {})
            sampler.state_observer = None
            require(len(outputs) == 2, "Expected separate Ac/Bu track complexes only")
            meta = outputs[0].metadata["coupling"]
            validate_pair(meta, recorder, percent)
            audit = recorder.audit()
            matched = (audit["initial_shared_sha256"], audit["churn_sha256"],
                       audit['ligand_initial_sha256'], audit['ligand_churn_sha256'])
            require(audit['initial_shared_sha256'] == fixed_audit['initial_shared_sha256'] and
                    audit['churn_sha256'] == fixed_audit['churn_sha256'],
                    'Protein noise no longer matches the fixed-ligand baseline')
            if baseline is None:
                baseline = matched
            require(matched == baseline, "Protein noise changed across release conditions")
            for output in outputs:
                require(np.isfinite(output.atom_array.coord).all(), "Nonfinite final structure")
                output.metadata["seed"] = seed
                output.metadata["inference_sampler"] = dict(config["sampler"], coupling_cut_fraction=percent/100)
                output.metadata['coupled_ligand_atom_pairs'] = config['coupled_ligand_atom_pairs']
                output.dump(pair_dir)
            recorder.save(pair_dir)
            write_json(pair_dir/'ligand_atom_names.json', input_report['ligand_atom_names'])
            write_json(pair_dir / "pair.json", dict(seed=seed, coupled_fraction=percent/100,
                       condition=condition, cutoff=meta["cutoff"], gpu=torch.cuda.get_device_name(),
                       foundry_revision=config["foundry_revision"], native_sequences_tied=False))
            (pair_dir / "_COMPLETE").touch()
            rows.append(dict(seed=seed, condition=condition, coupled_fraction=percent/100,
                             path=str(pair_dir.relative_to(root))))
            print(f"DONE seed={seed} condition={condition} final_CA_RMSD={meta['shared_chain_state']['all_ca_rmsd'][-1][0]:.6f}", flush=True)
            del outputs, recorder
        require(len(rows) == 21, "Incomplete seed")
        write_csv(directory / "pairs.csv", rows)
        write_json(directory / "seed_summary.json", dict(seed=seed, pair_count=21, elapsed_seconds=time.monotonic()-t0,
                   gpu=torch.cuda.get_device_name(), gpu_memory_peak_bytes=torch.cuda.max_memory_allocated(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda, source_hashes=config["source_hashes"],
                   identical_noise_across_all_conditions=True, protein_noise_matches_fixed_baseline=True))
        (directory / "_COMPLETE").touch()
    except Exception:
        (directory / "_FAILED.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate", "run"])
    parser.add_argument("--project", type=Path, default=Path("/project"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", default=os.environ.get("SLURM_JOB_ID", "manual"))
    args = parser.parse_args()
    if args.action in ("prepare", "validate"):
        config = prepare(args.project, args.root, args.config)
        if args.action == "validate":
            report = validate_inputs(make_engine(args.project, config, pipeline_only=True), config)
            write_json(args.root / "input_validation.json", report)
            print(json.dumps(report, indent=2))
    else:
        run_seed(args.project, args.root, args.seed, args.attempt)


if __name__ == "__main__":
    main()
