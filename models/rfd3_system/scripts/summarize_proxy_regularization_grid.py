#!/usr/bin/env python3
"""Summarize an rfd3_system proxy-kappa regularization grid.

Each experiment row must contain ``run_metadata.json`` and one coupled output
JSON with per-step diagnostics. The script writes a machine-readable CSV,
cross-grid PNGs, and a compact Markdown summary under the requested output
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path.cwd() / "runtime" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PREFERRED_MODE_ORDER = (
    "default",
    "ode",
    "binder",
    "gamma02_step15",
)
MODE_LABELS = {
    "gamma02_step15": "gamma0.2 / step1.5",
}


SUMMARY_FIELDS = (
    "row_id",
    "sampling_mode",
    "rho",
    "step_scale",
    "gamma_0",
    "job_id",
    "gpu_model",
    "n_steps",
    "n_samples",
    "degenerate_fraction",
    "raw_outside_clamp_fraction",
    "regularized_outside_clamp_fraction",
    "final_clamped_fraction",
    "reliability_q10",
    "reliability_median",
    "reliability_q90",
    "relative_denominator_q10",
    "relative_denominator_median",
    "relative_denominator_q90",
    "normalized_regularized_residual_median",
    "normalized_regularized_residual_q90",
    "normalized_final_residual_median",
    "normalized_final_residual_q90",
    "raw_kappa_step_variation_mean",
    "regularized_kappa_step_variation_mean",
    "applied_kappa_step_variation_mean",
    "cosine_track1_mix_median",
    "cosine_track2_mix_median",
    "late_track_delta_cosine_median",
    "late_worst_track_mix_cosine_median",
    "cif_count",
    "diagnostic_json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment_root>/summary.",
    )
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else experiment_root / "summary"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _summarize_run(metadata_path)
        for metadata_path in sorted(experiment_root.rglob("run_metadata.json"))
    ]
    if not rows:
        raise SystemExit(f"No run_metadata.json files found under {experiment_root}.")

    rows.sort(key=lambda row: (str(row["sampling_mode"]), float(row["rho"])))
    csv_path = out_dir / "proxy_kappa_regularization_summary.csv"
    _write_csv(csv_path, rows)
    plot_paths = _write_summary_plots(out_dir, rows)
    markdown_path = out_dir / "README.md"
    _write_markdown(markdown_path, rows, csv_path, plot_paths)

    print(f"WROTE {csv_path}")
    for path in plot_paths:
        print(f"WROTE {path}")
    print(f"WROTE {markdown_path}")
    return 0


def _summarize_run(metadata_path: Path) -> dict[str, Any]:
    run_dir = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    diagnostic_path = _choose_diagnostic_json(run_dir)
    output = json.loads(diagnostic_path.read_text())
    coupling = output.get("coupling", {})
    diagnostics = coupling.get("diagnostics", {})
    if not diagnostics:
        raise ValueError(f"No coupling diagnostics in {diagnostic_path}.")

    raw = _array(diagnostics, "raw_kappa")
    regularized = _array(diagnostics, "regularized_kappa")
    applied = _array(diagnostics, "kappa")
    _require_same_shape(raw, regularized, applied)

    denominator = _array(diagnostics, "denominator")
    scale = _array(diagnostics, "regularization_scale")
    reliability = _array(diagnostics, "reliability")
    relative_denominator = _array(diagnostics, "relative_denominator")
    degenerate = _array(diagnostics, "degenerate").astype(bool)
    regularized_residual = _array(diagnostics, "regularized_proxy_residual")
    final_residual = _array(diagnostics, "proxy_residual")
    _require_same_shape(
        raw,
        denominator,
        scale,
        reliability,
        relative_denominator,
        degenerate,
        regularized_residual,
        final_residual,
    )

    kappa_min = float(coupling.get("proxy_kappa_min", -1.0))
    kappa_max = float(coupling.get("proxy_kappa_max", 2.0))
    normalized_regularized_residual = np.abs(regularized_residual) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )
    normalized_final_residual = np.abs(final_residual) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )

    cosine_1 = _optional_array(
        diagnostics,
        "cosine_delta_1_mix_all_shared",
    )
    cosine_2 = _optional_array(
        diagnostics,
        "cosine_delta_2_mix_all_shared",
    )
    delta_1_norm = _array(diagnostics, "delta_1_norm")
    delta_2_norm = _array(diagnostics, "delta_2_norm")
    _require_same_shape(raw, delta_1_norm, delta_2_norm)
    normalized_t = np.asarray(diagnostics["normalized_t"], dtype=float)
    if normalized_t.ndim != 1 or normalized_t.shape[0] != raw.shape[0]:
        raise ValueError(
            "Diagnostic normalized_t must contain one value per denoising step; "
            f"got {normalized_t.shape} for kappa shape {raw.shape}."
        )
    late_mask = normalized_t >= 0.8
    if not np.any(late_mask):
        raise ValueError("No diagnostics found in the late window t >= 0.8.")

    proxy_eps = float(coupling.get("proxy_eps", 1e-8))
    track_dot = 0.5 * (
        np.square(delta_1_norm)
        + np.square(delta_2_norm)
        - denominator
    )
    track_delta_cosine = track_dot / (
        delta_1_norm * delta_2_norm + proxy_eps
    )
    if cosine_1 is None or cosine_2 is None:
        late_worst_track_mix_cosine = None
    else:
        late_worst_track_mix_cosine = np.minimum(cosine_1, cosine_2)[late_mask]

    return {
        "row_id": metadata["row_id"],
        "sampling_mode": metadata["sampling_mode"],
        "rho": float(metadata["rho"]),
        "step_scale": float(metadata["step_scale"]),
        "gamma_0": float(metadata["gamma_0"]),
        "job_id": str(metadata["job_id"]),
        "gpu_model": str(metadata.get("gpu_model", "")),
        "n_steps": raw.shape[0],
        "n_samples": raw.shape[1],
        "degenerate_fraction": float(np.mean(degenerate)),
        "raw_outside_clamp_fraction": float(
            np.mean((raw < kappa_min) | (raw > kappa_max))
        ),
        "regularized_outside_clamp_fraction": float(
            np.mean((regularized < kappa_min) | (regularized > kappa_max))
        ),
        "final_clamped_fraction": float(
            np.mean(~np.isclose(regularized, applied, rtol=0.0, atol=1e-7))
        ),
        "reliability_q10": _quantile(reliability, 0.1),
        "reliability_median": _quantile(reliability, 0.5),
        "reliability_q90": _quantile(reliability, 0.9),
        "relative_denominator_q10": _quantile(relative_denominator, 0.1),
        "relative_denominator_median": _quantile(relative_denominator, 0.5),
        "relative_denominator_q90": _quantile(relative_denominator, 0.9),
        "normalized_regularized_residual_median": _quantile(
            normalized_regularized_residual,
            0.5,
        ),
        "normalized_regularized_residual_q90": _quantile(
            normalized_regularized_residual,
            0.9,
        ),
        "normalized_final_residual_median": _quantile(
            normalized_final_residual,
            0.5,
        ),
        "normalized_final_residual_q90": _quantile(
            normalized_final_residual,
            0.9,
        ),
        "raw_kappa_step_variation_mean": _mean_step_variation(raw),
        "regularized_kappa_step_variation_mean": _mean_step_variation(
            regularized
        ),
        "applied_kappa_step_variation_mean": _mean_step_variation(applied),
        "cosine_track1_mix_median": _nanmedian(cosine_1),
        "cosine_track2_mix_median": _nanmedian(cosine_2),
        "late_track_delta_cosine_median": _nanmedian(
            track_delta_cosine[late_mask]
        ),
        "late_worst_track_mix_cosine_median": _nanmedian(
            late_worst_track_mix_cosine
        ),
        "cif_count": len(list(run_dir.glob("*.cif*"))),
        "diagnostic_json": str(diagnostic_path),
    }


def _choose_diagnostic_json(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*merged*_model_0.json"))
    if not candidates:
        candidates = sorted(run_dir.glob("*track1_model_0.json"))
    if not candidates:
        candidates = [
            path
            for path in sorted(run_dir.glob("*.json"))
            if path.name != "run_metadata.json"
        ]
    for path in candidates:
        payload = json.loads(path.read_text())
        if payload.get("coupling", {}).get("diagnostics"):
            return path
    raise ValueError(f"No diagnostic output JSON found in {run_dir}.")


def _array(diagnostics: dict, key: str) -> np.ndarray:
    if key not in diagnostics:
        raise KeyError(f"Missing required diagnostic {key!r}.")
    array = np.asarray(diagnostics[key], dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Diagnostic {key!r} has shape {array.shape}.")
    return array


def _optional_array(diagnostics: dict, key: str) -> np.ndarray | None:
    return _array(diagnostics, key) if key in diagnostics else None


def _require_same_shape(first: np.ndarray, *others: np.ndarray) -> None:
    for array in others:
        if array.shape != first.shape:
            raise ValueError(
                f"Diagnostic shape mismatch: {first.shape} versus {array.shape}."
            )


def _quantile(array: np.ndarray, quantile: float) -> float:
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, quantile)) if finite.size else math.nan


def _nanmedian(array: np.ndarray | None) -> float:
    if array is None:
        return math.nan
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


def _mean_step_variation(array: np.ndarray) -> float:
    if array.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(array, axis=0))))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_plots(
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    specs = (
        (
            "final_clamped_fraction",
            "Final clamp fraction",
            "proxy_kappa_regularization_clamp_fraction.png",
        ),
        (
            "applied_kappa_step_variation_mean",
            "Mean absolute applied-kappa step variation",
            "proxy_kappa_regularization_variation.png",
        ),
        (
            "reliability_median",
            "Median reliability",
            "proxy_kappa_regularization_reliability.png",
        ),
        (
            "normalized_final_residual_median",
            "Median |final proxy residual| / S",
            "proxy_kappa_regularization_residual.png",
        ),
        (
            "late_track_delta_cosine_median",
            "Median late cos(delta_1, delta_2)",
            "proxy_kappa_regularization_late_track_agreement.png",
        ),
        (
            "late_worst_track_mix_cosine_median",
            "Median late worst-track cos(delta_i, delta_mix)",
            "proxy_kappa_regularization_late_mix_alignment.png",
        ),
    )
    paths = []
    for field, ylabel, filename in specs:
        path = out_dir / filename
        _plot_metric_by_rho(path, rows, field=field, ylabel=ylabel)
        paths.append(path)
    return paths


def _plot_metric_by_rho(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    field: str,
    ylabel: str,
) -> None:
    rho_values = sorted({float(row["rho"]) for row in rows})
    labels = [f"{rho:g}" for rho in rho_values]
    x = np.arange(len(rho_values), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    available_modes = {str(row["sampling_mode"]) for row in rows}
    ordered_modes = [
        mode for mode in PREFERRED_MODE_ORDER if mode in available_modes
    ]
    ordered_modes.extend(sorted(available_modes - set(ordered_modes)))
    for mode in ordered_modes:
        by_rho = {
            float(row["rho"]): float(row[field])
            for row in rows
            if row["sampling_mode"] == mode
        }
        if not by_rho:
            continue
        values = [by_rho.get(rho, math.nan) for rho in rho_values]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            label=MODE_LABELS.get(mode, mode),
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("proxy_kappa_regularization_rho")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    csv_path: Path,
    plot_paths: list[Path],
) -> None:
    lines = [
        "# Proxy-Kappa Regularization Grid",
        "",
        "This automatically generated summary covers the completed experiment rows.",
        "Interpretive conclusions belong in the tracked rfd3_system results document.",
        "",
        f"- Summary CSV: `{csv_path.name}`",
        f"- Completed rows: {len(rows)}",
        "",
        "## Conditions",
        "",
        "| Mode | rho | Final clamp fraction | Median reliability | "
        "Median normalized residual | Applied-kappa variation | "
        "Late track agreement | Late worst-track alignment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sampling_mode']} | {row['rho']:g} | "
            f"{row['final_clamped_fraction']:.4f} | "
            f"{row['reliability_median']:.4f} | "
            f"{row['normalized_final_residual_median']:.3e} | "
            f"{row['applied_kappa_step_variation_mean']:.4f} | "
            f"{row['late_track_delta_cosine_median']:.4f} | "
            f"{row['late_worst_track_mix_cosine_median']:.4f} |"
        )
    lines.extend(["", "## Cross-Grid Plots", ""])
    lines.extend(f"- `{plot.name}`" for plot in plot_paths)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
