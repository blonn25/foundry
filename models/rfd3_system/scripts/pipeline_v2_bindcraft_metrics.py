#!/usr/bin/env python3
"""BindCraft-paper metrics for rfd3_system pipeline v2.

This helper intentionally computes only the small metric set used for pipeline
v2 filtering, not the full BindCraft CSV schema and not project-specific SEP
phosphate contact metrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adapted_bindcraft_functions.pyrosetta_utils import pr_relax, score_interface


PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_DIR",
        "/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design",
    )
)
PAIR_CHAINS = {
    "AB_SEP": ("A", "B"),
    "DC_SER": ("D", "C"),
    "AC_SEP": ("A", "C"),
    "DB_SER": ("D", "B"),
    "AB": ("A", "B"),
    "DC": ("D", "C"),
}
MONOMER_KINDS = {"A_SEP", "D_SER", "B", "C"}
PREFILTER_THRESHOLDS = {
    "ShapeComplementarity": (0.50, True),
    "n_InterfaceHbonds": (2.0, True),
    "n_InterfaceUnsatHbonds": (6.0, False),
    "Surface_Hydrophobicity": (0.37, False),
    # "InterfaceAAs_K": (3.0, False),
    # "InterfaceAAs_M": (3.0, False),
}
FINAL_THRESHOLDS = {
    "pLDDT": (0.8, True),
    "i_pTM": (0.5, True),
    "i_pAE_raw": (12.5, False),
    "ShapeComplementarity": (0.50, True),
    "n_InterfaceHbonds": (2.0, True),
    "n_InterfaceUnsatHbonds": (6.0, False),
    "Surface_Hydrophobicity": (0.37, False),
    "Binder_RMSD": (3.5, False),
    # "InterfaceAAs_K": (3.0, False),
    # "InterfaceAAs_M": (3.0, False),
}
MPNN_CIF_RE = re.compile(
    r"^(?P<mpnn_name>.+)_b(?P<batch>\d+)_d(?P<design>\d+)\.cif$"
)


@dataclass(frozen=True)
class StructureRecord:
    task_id: str
    complex_kind: str
    design_key: str
    model_index: int
    batch_index: int
    design_index: int
    raw_cif: str
    json_path: str
    plddt_raw: float | None
    iptm_raw: float | None
    mean_ipae_raw: float | None
    mean_pae_raw: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prefilter = subparsers.add_parser(
        "prefilter",
        help="Relax and score post-MPNN on-target AB/DC structures.",
    )
    prefilter.add_argument("mpnn_output_dir", type=Path)
    prefilter.add_argument("--out-dir", type=Path, required=True)
    prefilter.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional CSV path. Defaults to <out-dir>/prefilter_summary.csv.",
    )

    folded = subparsers.add_parser(
        "score-folded",
        help="Relax and score ESMFold2 folded structures.",
    )
    folded.add_argument("esmfold2_output_dir", type=Path)
    folded.add_argument("--out-dir", type=Path, required=True)
    folded.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional CSV path. Defaults to <out-dir>/folded_bindcraft_metrics_long.csv.",
    )
    return parser.parse_args()


def normalize_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if str(path) == "/project":
        return PROJECT_ROOT
    path_text = str(path)
    if path_text.startswith("/project/"):
        return PROJECT_ROOT / path_text.removeprefix("/project/")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def extract_chains_to_pdb(input_path: Path, chain_ids: set[str], output_path: Path) -> Path:
    """Write a PDB containing only the requested chain IDs."""

    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

    class ChainSelect(Select):
        def accept_chain(self, chain):
            return str(chain.id) in chain_ids

    input_path = normalize_project_path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Structure not found: {input_path}")
    parser = (
        MMCIFParser(QUIET=True)
        if input_path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else PDBParser(QUIET=True)
    )
    with open_text(input_path) as handle:
        structure = parser.get_structure("structure", handle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), ChainSelect())
    return output_path


def interface_metrics(pdb_path: Path, target_chain: str, binder_chain: str) -> dict[str, Any]:
    scores, interface_aa, residues = score_interface(
        pdb_path,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    return {
        "ShapeComplementarity": scores["interface_sc"],
        "n_InterfaceHbonds": scores["interface_interface_hbonds"],
        "n_InterfaceUnsatHbonds": scores["interface_delta_unsat_hbonds"],
        "Surface_Hydrophobicity": scores["surface_hydrophobicity"],
        "InterfaceAAs_K": interface_aa.get("K", 0),
        "InterfaceAAs_M": interface_aa.get("M", 0),
        "interface_residues_pdb_ids": residues,
    }


def threshold_failures(metrics: dict[str, Any], thresholds: dict[str, tuple[float, bool]]) -> list[str]:
    failures: list[str] = []
    for key, (threshold, higher_is_better) in thresholds.items():
        value = metrics.get(key)
        if value in (None, ""):
            failures.append(key)
            continue
        value_f = float(value)
        if higher_is_better and value_f <= threshold:
            failures.append(key)
        if not higher_is_better and value_f >= threshold:
            failures.append(key)
    return failures


def flatten_prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def parse_mpnn_designs(mpnn_output_dir: Path) -> list[dict[str, Any]]:
    designs: list[dict[str, Any]] = []
    for path in sorted(mpnn_output_dir.glob("*.cif")):
        match = MPNN_CIF_RE.fullmatch(path.name)
        if not match:
            continue
        if "_model_" not in match.group("mpnn_name"):
            raise ValueError(
                f"Cannot infer model_index from ProteinMPNN output name: {path.name}"
            )
        design_key = path.name[: -len(".cif")]
        designs.append(
            {
                "design_key": design_key,
                "mpnn_name": match.group("mpnn_name"),
                "model_index": int(match.group("mpnn_name").rsplit("_model_", 1)[-1]),
                "batch_index": int(match.group("batch")),
                "design_index": int(match.group("design")),
                "mpnn_cif": str(path),
            }
        )
    if not designs:
        raise ValueError(f"No ProteinMPNN design CIFs found in {mpnn_output_dir}")
    return designs


def run_prefilter(args: argparse.Namespace) -> None:
    mpnn_output_dir = normalize_project_path(args.mpnn_output_dir)
    out_dir = normalize_project_path(args.out_dir)
    summary_csv = normalize_project_path(args.summary_csv or out_dir / "prefilter_summary.csv")
    work_dir = out_dir / "work"
    relaxed_dir = out_dir / "relaxed"
    out_dir.mkdir(parents=True, exist_ok=True)
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for design in parse_mpnn_designs(mpnn_output_dir):
        pair_rows: dict[str, dict[str, Any]] = {}
        for pair_kind, chains in {"AB": {"A", "B"}, "DC": {"D", "C"}}.items():
            target_chain, binder_chain = PAIR_CHAINS[pair_kind]
            unrelaxed = work_dir / f"{design['design_key']}_{pair_kind}.pdb"
            relaxed = relaxed_dir / f"{design['design_key']}_{pair_kind}_relaxed.pdb"
            extract_chains_to_pdb(Path(design["mpnn_cif"]), chains, unrelaxed)
            pr_relax(unrelaxed, relaxed)
            metrics = interface_metrics(relaxed, target_chain, binder_chain)
            failures = threshold_failures(metrics, PREFILTER_THRESHOLDS)
            pair_rows[pair_kind] = {
                "unrelaxed_pdb": str(unrelaxed),
                "relaxed_pdb": str(relaxed),
                "pass": not failures,
                "failed_filters": ";".join(failures),
                **metrics,
            }

        failed_filters = [
            f"{pair}_{name}"
            for pair, metrics in pair_rows.items()
            for name in metrics["failed_filters"].split(";")
            if name
        ]
        rows.append(
            {
                **design,
                "prefilter_pass": not failed_filters,
                "failed_filters": ";".join(failed_filters),
                **flatten_prefixed("AB", pair_rows["AB"]),
                **flatten_prefixed("DC", pair_rows["DC"]),
            }
        )

    write_csv(summary_csv, rows)
    write_json(
        out_dir / "prefilter_manifest.json",
        {
            "mpnn_output_dir": str(mpnn_output_dir),
            "summary_csv": str(summary_csv),
            "design_count": len(rows),
            "pass_count": sum(1 for row in rows if row["prefilter_pass"]),
            "thresholds": PREFILTER_THRESHOLDS,
        },
    )
    print(f"Wrote {len(rows)} prefilter row(s) to {summary_csv}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def plddt_to_bindcraft(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0 if value > 1.0 else value


def pae_to_bindcraft(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 31.0 if value > 1.0 else value


def metric_or_blank(value: float | None) -> float | str:
    """Keep valid numeric zero values instead of treating them as blanks."""

    return value if value is not None else ""


def discover_esmfold2_structures(esm_dir: Path) -> list[StructureRecord]:
    records: list[StructureRecord] = []
    for json_path in sorted(esm_dir.glob("*.json")):
        if json_path.name in {"planned_folds.json", "esmfold2_run_manifest.json", "row_metadata.json"}:
            continue
        payload = read_json(json_path)
        samples = payload.get("samples", [])
        for sample_idx, sample in enumerate(samples):
            raw_cif = sample.get("cif")
            if not raw_cif:
                continue
            suffix = f"_diffusion{sample_idx}" if len(samples) > 1 else ""
            records.append(
                StructureRecord(
                    task_id=f"{payload['task_id']}{suffix}",
                    complex_kind=str(payload["complex_kind"]),
                    design_key=str(payload["design_key"]),
                    model_index=int(payload["model_index"]),
                    batch_index=int(payload["batch_index"]),
                    design_index=int(payload["design_index"]),
                    raw_cif=str(normalize_project_path(raw_cif)),
                    json_path=str(json_path),
                    plddt_raw=scalar_or_none(sample.get("plddt_mean")),
                    iptm_raw=scalar_or_none(sample.get("iptm")),
                    mean_ipae_raw=scalar_or_none(sample.get("mean_ipae")),
                    mean_pae_raw=scalar_or_none(sample.get("mean_pae")),
                )
            )
    return records


class StructureCache:
    def __init__(self) -> None:
        self._cache: dict[str, dict[str, tuple[list[int], np.ndarray]]] = {}

    def load_ca(self, path: str | Path) -> dict[str, tuple[list[int], np.ndarray]]:
        normalized = str(normalize_project_path(path))
        if normalized not in self._cache:
            from Bio.PDB import MMCIFParser, PDBParser

            structure_path = Path(normalized)
            parser = (
                MMCIFParser(QUIET=True)
                if structure_path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
                else PDBParser(QUIET=True)
            )
            with open_text(structure_path) as handle:
                structure = parser.get_structure("structure", handle)
            model = next(structure.get_models())
            chains: dict[str, tuple[list[int], np.ndarray]] = {}
            for chain in model:
                records: list[tuple[int, np.ndarray]] = []
                seen: set[int] = set()
                for residue in chain:
                    if "CA" not in residue:
                        continue
                    res_id = int(residue.id[1])
                    if res_id in seen:
                        continue
                    coord = residue["CA"].get_coord().astype(float)
                    if np.isfinite(coord).all():
                        seen.add(res_id)
                        records.append((res_id, coord))
                if records:
                    records.sort(key=lambda item: item[0])
                    chains[str(chain.id)] = (
                        [res_id for res_id, _ in records],
                        np.stack([coord for _, coord in records], axis=0),
                    )
            self._cache[normalized] = chains
        return self._cache[normalized]


def matched_ca(
    cache: StructureCache,
    mobile_path: str,
    mobile_chain: str,
    target_path: str,
    target_chain: str,
) -> tuple[np.ndarray, np.ndarray]:
    mobile = cache.load_ca(mobile_path)
    target = cache.load_ca(target_path)
    if mobile_chain not in mobile:
        raise ValueError(f"missing mobile chain {mobile_chain} in {mobile_path}")
    if target_chain not in target:
        raise ValueError(f"missing target chain {target_chain} in {target_path}")
    mobile_res, mobile_coords = mobile[mobile_chain]
    target_res, target_coords = target[target_chain]
    if mobile_res != target_res:
        raise ValueError(
            f"CA residue IDs differ for {mobile_chain}->{target_chain}: "
            f"{len(mobile_res)} vs {len(target_res)}"
        )
    return mobile_coords, target_coords


def alignment_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center
    u, _, vt = np.linalg.svd(mobile_centered.T @ target_centered)
    handedness = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ np.diag([1.0, 1.0, handedness]) @ vt
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def ca_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((mobile - target) ** 2, axis=1))))


def aligned_ca_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    rotation, translation = alignment_transform(mobile, target)
    return ca_rmsd(mobile @ rotation + translation, target)


def rmsd_or_error(
    cache: StructureCache,
    complex_path: str,
    complex_chain: str,
    monomer_path: str,
    monomer_chain: str,
) -> tuple[float | str, str]:
    try:
        complex_ca, monomer_ca = matched_ca(
            cache,
            complex_path,
            complex_chain,
            monomer_path,
            monomer_chain,
        )
        return aligned_ca_rmsd(complex_ca, monomer_ca), ""
    except Exception as error:
        return "", str(error)


def run_score_folded(args: argparse.Namespace) -> None:
    esm_dir = normalize_project_path(args.esmfold2_output_dir)
    out_dir = normalize_project_path(args.out_dir)
    summary_csv = normalize_project_path(args.summary_csv or out_dir / "folded_bindcraft_metrics_long.csv")
    relaxed_dir = out_dir / "relaxed"
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    records = discover_esmfold2_structures(esm_dir)
    relaxed_paths: dict[str, str] = {}
    rows_by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        relaxed_path = relaxed_dir / f"{record.task_id}_relaxed.pdb"
        pr_relax(normalize_project_path(record.raw_cif), relaxed_path)
        relaxed_paths[record.task_id] = str(relaxed_path)
        rows_by_task[record.task_id] = {
            "task_id": record.task_id,
            "complex_kind": record.complex_kind,
            "design_key": record.design_key,
            "model_index": record.model_index,
            "batch_index": record.batch_index,
            "design_index": record.design_index,
            "esmfold2_json": record.json_path,
            "esmfold2_cif": record.raw_cif,
            "relaxed_pdb": str(relaxed_path),
            "pLDDT_raw": record.plddt_raw if record.plddt_raw is not None else "",
            "pLDDT": metric_or_blank(plddt_to_bindcraft(record.plddt_raw)),
            "i_pTM": record.iptm_raw if record.iptm_raw is not None else "",
            "i_pAE_raw": record.mean_ipae_raw if record.mean_ipae_raw is not None else "",
            "i_pAE": metric_or_blank(pae_to_bindcraft(record.mean_ipae_raw)),
            "mean_pae_raw": record.mean_pae_raw if record.mean_pae_raw is not None else "",
        }

    cache = StructureCache()
    records_by_design_kind = {
        (record.design_key, record.complex_kind): record for record in records
    }
    for record in records:
        row = rows_by_task[record.task_id]
        if record.complex_kind in PAIR_CHAINS:
            target_chain, binder_chain = PAIR_CHAINS[record.complex_kind]
            row |= interface_metrics(Path(relaxed_paths[record.task_id]), target_chain, binder_chain)
            binder_monomer_kind = binder_chain
            shared_monomer_kind = "A_SEP" if target_chain == "A" else "D_SER"
            binder_record = records_by_design_kind.get((record.design_key, binder_monomer_kind))
            shared_record = records_by_design_kind.get((record.design_key, shared_monomer_kind))
            binder_rmsd, binder_error = ("", "missing binder monomer")
            shared_rmsd, shared_error = ("", "missing shared monomer")
            if binder_record is not None:
                binder_rmsd, binder_error = rmsd_or_error(
                    cache,
                    relaxed_paths[record.task_id],
                    binder_chain,
                    relaxed_paths[binder_record.task_id],
                    binder_chain,
                )
            if shared_record is not None:
                shared_rmsd, shared_error = rmsd_or_error(
                    cache,
                    relaxed_paths[record.task_id],
                    target_chain,
                    relaxed_paths[shared_record.task_id],
                    target_chain,
                )
            row["Binder_RMSD"] = binder_rmsd
            row["Binder_RMSD_error"] = binder_error
            row["Shared_RMSD"] = shared_rmsd
            row["Shared_RMSD_error"] = shared_error
            failures = threshold_failures(row, FINAL_THRESHOLDS)
            row["bindcraft_paper_pass"] = not failures
            row["bindcraft_paper_failed_filters"] = ";".join(failures)
        elif record.complex_kind in MONOMER_KINDS:
            row["bindcraft_paper_pass"] = ""
            row["bindcraft_paper_failed_filters"] = ""

    rows = [rows_by_task[record.task_id] for record in records]
    write_csv(summary_csv, rows)
    write_json(
        out_dir / "folded_bindcraft_metrics_manifest.json",
        {
            "esmfold2_output_dir": str(esm_dir),
            "summary_csv": str(summary_csv),
            "record_count": len(rows),
            "pair_count": sum(1 for row in rows if row.get("complex_kind") in PAIR_CHAINS),
            "thresholds": FINAL_THRESHOLDS,
        },
    )
    print(f"Wrote {len(rows)} folded metric row(s) to {summary_csv}")


def main() -> None:
    args = parse_args()
    if args.command == "prefilter":
        run_prefilter(args)
    elif args.command == "score-folded":
        run_score_folded(args)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
