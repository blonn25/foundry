#!/usr/bin/env python3
"""Plot rfd3_system shared-chain coupling diagnostics from output JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

# Keep Matplotlib cache/config writes inside the project when the script is run
# from the project root on CoreHPC.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path.cwd() / "runtime" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG plots for rfd3_system kappa and proxy-residual "
            "diagnostics after inference has finished."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Output directories or specific rfd3_system JSON metadata files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for plots. By default, plots are written next "
            "to each input JSON file."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for JSON files.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Plot diagnostics for every model_N JSON. By default, only "
            "model_0 JSONs are used because diagnostics are batch-level."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution. Default: 180.",
    )
    args = parser.parse_args()

    json_paths = list(_iter_json_paths(args.paths, recursive=args.recursive))
    if not json_paths:
        raise SystemExit("No JSON files found.")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for json_path in json_paths:
        try:
            result = _plot_one_json(
                json_path,
                out_dir=args.out_dir,
                all_models=args.all_models,
                dpi=args.dpi,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch plotting robust.
            skipped += 1
            print(f"SKIP {json_path}: {exc}")
            continue
        if result:
            written += len(result)
            for path in result:
                print(f"WROTE {path}")
        else:
            skipped += 1

    print(f"Done. Wrote {written} plot(s); skipped {skipped} JSON file(s).")
    return 0


def _iter_json_paths(paths: Iterable[Path], *, recursive: bool) -> Iterable[Path]:
    """Yield JSON files from user-provided files or output directories."""

    pattern = "**/*.json" if recursive else "*.json"
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob(pattern))
        elif path.is_file() and path.suffix == ".json":
            yield path
        else:
            print(f"SKIP {path}: not a JSON file or directory")


def _plot_one_json(
    json_path: Path,
    *,
    out_dir: Path | None,
    all_models: bool,
    dpi: int,
) -> list[Path]:
    """Create kappa and proxy-residual PNGs for one coupled output JSON."""

    metadata = json.loads(json_path.read_text())
    coupling = metadata.get("coupling", {})
    output_kind = str(coupling.get("output", ""))
    if not output_kind.startswith("merged_A_plus_all_partners"):
        return []

    diagnostics = coupling.get("diagnostics", {})
    if not diagnostics:
        return []

    plot_prefix = _plot_prefix_for_json(json_path, all_models=all_models)
    if plot_prefix is None:
        return []

    if out_dir is not None:
        plot_prefix = out_dir / plot_prefix.name

    shared = str(coupling.get("shared_chain_id", "A"))
    track_1_label = _complex_label(shared, coupling.get("complex_1_partners", ["B"]))
    track_2_label = _complex_label(shared, coupling.get("complex_2_partners", ["C"]))

    paths: list[Path] = []
    if "kappa" in diagnostics:
        kappa = _as_step_sample_array(diagnostics["kappa"], "kappa")
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            kappa.shape[0],
        )
        kappa_path = plot_prefix.with_name(f"{plot_prefix.name}_kappa.png")
        _plot_kappa(
            kappa_path,
            normalized_t=normalized_t,
            kappa=kappa,
            track_1_label=track_1_label,
            track_2_label=track_2_label,
            kappa_min=float(coupling.get("proxy_kappa_min", -1.0)),
            kappa_max=float(coupling.get("proxy_kappa_max", 2.0)),
            dpi=dpi,
        )
        paths.append(kappa_path)

    if "proxy_residual" in diagnostics:
        residual = _as_step_sample_array(
            diagnostics["proxy_residual"],
            "proxy_residual",
        )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            residual.shape[0],
        )
        residual_path = plot_prefix.with_name(
            f"{plot_prefix.name}_proxy_residual.png"
        )
        _plot_proxy_residual(
            residual_path,
            normalized_t=normalized_t,
            residual=residual,
            dpi=dpi,
        )
        paths.append(residual_path)

    return paths


def _plot_prefix_for_json(json_path: Path, *, all_models: bool) -> Path | None:
    """Return the batch-level plot prefix for an rfd3_system output JSON."""

    stem = json_path.with_suffix("")
    marker = "_model_"
    stem_text = str(stem)
    if marker not in stem_text:
        return stem

    prefix, suffix = stem_text.rsplit(marker, 1)
    if suffix == "0" or all_models:
        return Path(prefix)
    return None


def _as_step_sample_array(values, name: str) -> np.ndarray:
    """Coerce list diagnostics to a [steps, samples] float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"unexpected {name} diagnostic shape {array.shape}")
    return array


def _normalized_t_axis(values, n_steps: int) -> np.ndarray:
    """Return one normalized denoising-progress value per diagnostic step."""

    if values is None:
        return _fallback_t_axis(n_steps)
    axis = np.asarray(values, dtype=float).reshape(-1)
    if axis.size != n_steps:
        return _fallback_t_axis(n_steps)
    return axis


def _fallback_t_axis(n_steps: int) -> np.ndarray:
    """Construct a 0-to-1 t axis when metadata is missing or malformed."""

    if n_steps <= 1:
        return np.zeros(n_steps, dtype=float)
    return np.linspace(0.0, 1.0, n_steps)


def _plot_kappa(
    path: Path,
    *,
    normalized_t: np.ndarray,
    kappa: np.ndarray,
    track_1_label: str,
    track_2_label: str,
    kappa_min: float,
    kappa_max: float,
    dpi: int,
) -> None:
    """Plot kappa trajectories with track-leaning reference lines."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, kappa)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="0.45", linestyle=":", linewidth=1.2)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_title("Shared-chain coupling weight over denoising")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("kappa in delta_mix")
    ax.set_xlim(0.0, 1.0)
    y_min = min(kappa_min, float(np.nanmin(kappa)), 0.0, 0.5, 1.0)
    y_max = max(kappa_max, float(np.nanmax(kappa)), 0.0, 0.5, 1.0)
    ax.set_ylim(_with_padding(y_min, y_max))
    ax.text(
        1.01,
        1.0,
        f"kappa=1: track 1 ({track_1_label}) only",
        transform=ax.get_yaxis_transform(),
        va="center",
        fontsize=8,
    )
    ax.text(
        1.01,
        0.5,
        "kappa=0.5: equal mix",
        transform=ax.get_yaxis_transform(),
        va="center",
        fontsize=8,
    )
    ax.text(
        1.01,
        0.0,
        f"kappa=0: track 2 ({track_2_label}) only",
        transform=ax.get_yaxis_transform(),
        va="center",
        fontsize=8,
    )
    ax.text(
        0.0,
        -0.22,
        (
            f"kappa > 0.5 leans toward {track_1_label}; "
            f"kappa < 0.5 leans toward {track_2_label}."
        ),
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_proxy_residual(
    path: Path,
    *,
    normalized_t: np.ndarray,
    residual: np.ndarray,
    dpi: int,
) -> None:
    """Plot post-clamp proxy residual trajectories."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, residual)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_title("Proxy residual over denoising")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("proxy residual after kappa clamping")
    ax.set_xlim(0.0, 1.0)
    max_abs = float(np.nanmax(np.abs(residual))) if residual.size else 1.0
    max_abs = max(max_abs, 1e-6)
    ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    ax.text(
        0.0,
        -0.22,
        (
            "Residual near zero means the implemented proxy equalization "
            "condition was better satisfied."
        ),
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_samples(ax, normalized_t: np.ndarray, values: np.ndarray) -> None:
    """Plot one line per diffusion-batch sample."""

    for sample_idx in range(values.shape[1]):
        ax.plot(
            normalized_t,
            values[:, sample_idx],
            linewidth=1.6,
            label=f"sample {sample_idx}",
        )


def _with_padding(y_min: float, y_max: float) -> tuple[float, float]:
    """Pad a y-axis range while handling flat values."""

    if abs(y_max - y_min) < 1e-12:
        return y_min - 0.5, y_max + 0.5
    padding = max((y_max - y_min) * 0.05, 0.02)
    return y_min - padding, y_max + padding


def _complex_label(shared_chain_id: str, partner_chain_ids: Iterable[str]) -> str:
    """Return a compact label such as A+B or A+C."""

    partners = [str(partner) for partner in partner_chain_ids]
    return "+".join([str(shared_chain_id)] + partners)


if __name__ == "__main__":
    raise SystemExit(main())
