#!/usr/bin/env python3
"""Five matched full-schedule GPU runs; requires a SLURM GPU allocation.

Determinism is enabled for validation in both the original and copied model.
This changes neither model's defaults and makes exact CUDA parity meaningful.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import torch


def run_case(case, root):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    package = "rfd3_system" if case == "original" else "rfd3_system_early_cut"
    module = importlib.import_module(f"{package}.model.inference_sampler")
    cls = module.SampleDiffusionWithSuperDiffSharedChainProxy
    sample = cls.sample_coupled_superdiff_proxy
    out = root / case
    out.mkdir()

    def record(self, **kwargs):
        result = sample(self, **kwargs)
        # Internal final states avoid precision loss from CIF serialization.
        torch.save({key: result[key]["X_L"].detach().cpu()
                    for key in ("track_1", "track_2")}, out / "sampler_states.pt")
        return result

    cls.sample_coupled_superdiff_proxy = record
    overrides = [
        "inputs=null", f"out_dir={out}", "global_prefix=validation_",
        "ckpt_path=/weights/rfd3_latest.ckpt", "+specification.contig='90,/0,80,/0,100'",
        "+specification.length=270", "diffusion_batch_size=1", "n_batches=1", "seed=123",
        "dump_trajectories=False", "merged_output_policy=none",
        "coupling_mode=superdiff_shared_chain", "inference_sampler.kind=superdiff_shared_chain",
        "shared_chain_id=A", "complex_1_partners=[B]", "complex_2_partners=[C]",
        "+track_1_specification.contig='A1-90,/0,B1-80'", "+track_1_specification.length=170",
        "+track_1_specification.select_fixed_atoms=false", "+track_1_specification.select_unfixed_sequence=true",
        "+track_2_specification.contig='A1-90,/0,C1-100'", "+track_2_specification.length=190",
        "+track_2_specification.select_fixed_atoms=false", "+track_2_specification.select_unfixed_sequence=true",
    ]
    if case == "immediate":
        overrides.append("inference_sampler.coupling_cut_fraction=0")
    elif case == "fraction":
        overrides.append("inference_sampler.coupling_cut_fraction=0.5")
    elif case == "sigma":
        meta = metadata(root / "fraction")
        overrides.append(f'inference_sampler.coupling_cut_sigma={meta["cutoff"]["release_pre_churn_sigma"]}')
    sys.argv = [f"{package}.cli", "design", *overrides]
    runpy.run_module(f"{package}.cli", run_name="__main__")


def metadata(directory):
    candidates = []
    for path in directory.glob("*.json"):
        data = json.loads(path.read_text())
        if data.get("coupling", {}).get("output") == "track_1_A_plus_partners":
            candidates.append(data["coupling"])
    if len(candidates) != 1:
        raise AssertionError(f"Expected one track-1 metadata file in {directory}: {len(candidates)}")
    return candidates[0]


def collect(root):
    states = {case: torch.load(root / case / "sampler_states.pt", weights_only=True)
              for case in ("original", "disabled", "immediate", "fraction", "sigma")}
    comparisons = {}
    for a, b in (("original", "disabled"), ("fraction", "sigma")):
        differences = {key: float((states[a][key].float() - states[b][key].float()).abs().max())
                       for key in ("track_1", "track_2")}
        comparisons[f"{a}_vs_{b}"] = differences
        assert all(value == 0 for value in differences.values()), comparisons
    results = {}
    for case in ("disabled", "immediate", "fraction", "sigma"):
        meta = metadata(root / case)
        state = meta["shared_chain_state"]
        diag = meta["diagnostics"]
        n = meta["cutoff"]["executed_update_count"]
        k = meta["cutoff"]["coupled_update_count"]
        assert n == 199 and len(state["all_ca_rmsd"]) == n + 1
        assert len(list((root / case).glob("*.cif*"))) == 2
        rmsd = torch.tensor(state["movable_ca_rmsd"])
        assert torch.isfinite(rmsd).all() and torch.all(rmsd[:k+1] == 0)
        if k < n:
            assert rmsd[k+1:].max() > 0
            assert all(row == [None] for row in diag["kappa"][k:])
        noise_max = torch.tensor(diag["shared_noise_max_abs_difference"]).max().item()
        assert noise_max == 0
        results[case] = dict(cutoff=meta["cutoff"], final_ca_rmsd=float(rmsd[-1, 0]),
                             max_ca_rmsd=float(rmsd.max()), noise_max_abs_difference=noise_max,
                             state_measurements=len(rmsd))
    report = dict(status="passed", gpu=torch.cuda.get_device_name(), comparisons=comparisons, cases=results)
    (root / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "_COMPLETE").touch()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case", choices=["original", "disabled", "immediate", "fraction", "sigma"])
    args = parser.parse_args()
    root = args.out_dir.resolve()
    if args.case:
        return run_case(args.case, root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing a nonempty validation directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for case in ("original", "disabled", "immediate", "fraction", "sigma"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out-dir", str(root),
                        "--case", case], env=environment, check=True)
    collect(root)


if __name__ == "__main__":
    main()
