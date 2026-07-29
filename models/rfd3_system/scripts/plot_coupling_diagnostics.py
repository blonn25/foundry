#!/usr/bin/env python3
"""Plot rfd3_system shared-chain coupling diagnostics from output JSON files."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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
from matplotlib.colors import to_rgb


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG plots for rfd3_system kappa regularization, "
            "proxy-residual, and delta/mix cosine diagnostics after inference "
            "has finished."
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
            "Deprecated compatibility flag; ignored. By default the plotter "
            "writes one batch-level plot set. Use --model-index N for a "
            "single generated model."
        ),
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=None,
        help=(
            "Optional generated model index to plot as a one-off. By default, "
            "all diffusion-batch samples are shown together."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution. Default: 180.",
    )
    parser.add_argument(
        "--max-cosine-samples",
        type=int,
        default=3,
        help=(
            "Maximum number of diffusion-batch samples to show on each "
            "batch-level cosine plot. Default: 3."
        ),
    )
    args = parser.parse_args()
    if args.max_cosine_samples <= 0:
        raise SystemExit("--max-cosine-samples must be positive.")

    json_paths = list(_iter_json_paths(args.paths, recursive=args.recursive))
    if not json_paths:
        raise SystemExit("No JSON files found.")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_models:
        print("NOTE: --all-models is deprecated and ignored; use --model-index N.")

    sources, skipped = _collect_diagnostic_sources(json_paths)
    written = 0
    for source in sources:
        try:
            result = _plot_one_source(
                source,
                out_dir=args.out_dir,
                model_index=args.model_index,
                max_cosine_samples=args.max_cosine_samples,
                dpi=args.dpi,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch plotting robust.
            skipped += 1
            print(f"SKIP {source.json_path}: {exc}")
            continue
        if result:
            written += len(result)
            for path in result:
                print(f"WROTE {path}")
        else:
            skipped += 1

    print(
        f"Done. Wrote {written} plot(s) from {len(sources)} batch source(s); "
        f"skipped {skipped} JSON file(s)."
    )
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


@dataclass(frozen=True)
class DiagnosticSource:
    """A single JSON chosen to provide diagnostics for one diffusion batch."""

    json_path: Path
    plot_prefix: Path
    source_model_index: int | None
    coupling: dict
    diagnostics: dict


def _collect_diagnostic_sources(
    json_paths: Iterable[Path],
) -> tuple[list[DiagnosticSource], int]:
    """Choose one coupled diagnostic JSON for each diffusion batch.

    rfd3_system writes identical coupling diagnostics into track and merged JSON
    files, and also repeats those diagnostics for each model_N output in the
    same diffusion batch.  Grouping by normalized batch prefix prevents duplicate
    plot sets when a run writes track1, track2, and merged variants.
    """

    sources: dict[Path, DiagnosticSource] = {}
    skipped = 0
    for json_path in json_paths:
        try:
            source = _diagnostic_source_from_json(json_path)
        except Exception as exc:  # noqa: BLE001 - keep directory scans robust.
            skipped += 1
            print(f"SKIP {json_path}: {exc}")
            continue
        if source is None:
            skipped += 1
            continue

        existing = sources.get(source.plot_prefix)
        if existing is None or _source_rank(source) < _source_rank(existing):
            if existing is not None:
                skipped += 1
            sources[source.plot_prefix] = source
        else:
            skipped += 1

    return list(sources.values()), skipped


def _diagnostic_source_from_json(json_path: Path) -> DiagnosticSource | None:
    """Return a diagnostic source if the JSON contains coupling diagnostics."""

    metadata = json.loads(json_path.read_text())
    coupling = metadata.get("coupling", {})
    diagnostics = coupling.get("diagnostics", {})
    if not diagnostics:
        return None

    plot_prefix, source_model_index = _plot_prefix_for_json(json_path)
    return DiagnosticSource(
        json_path=json_path,
        plot_prefix=plot_prefix,
        source_model_index=source_model_index,
        coupling=coupling,
        diagnostics=diagnostics,
    )


def _source_rank(source: DiagnosticSource) -> int:
    """Prefer merged JSONs, then track 1, then track 2 when duplicates exist."""

    output_kind = str(source.coupling.get("output", ""))
    model_penalty = 0 if source.source_model_index in (None, 0) else 10
    if output_kind.startswith("merged_A_plus_all_partners"):
        return model_penalty
    if output_kind == "track_1_A_plus_partners":
        return model_penalty + 1
    if output_kind == "track_2_A_plus_partners":
        return model_penalty + 2
    return model_penalty + 3


def _plot_one_source(
    source: DiagnosticSource,
    *,
    out_dir: Path | None,
    model_index: int | None,
    max_cosine_samples: int,
    dpi: int,
) -> list[Path]:
    """Create kappa and proxy-residual PNGs for one diffusion batch."""

    coupling = source.coupling
    diagnostics = source.diagnostics
    plot_prefix = source.plot_prefix
    if model_index is not None:
        plot_prefix = Path(f"{plot_prefix}_model_{model_index}")
    if out_dir is not None:
        plot_prefix = out_dir / plot_prefix.name

    shared = str(coupling.get("shared_chain_id", "A"))
    track_1_label = _complex_label(shared, coupling.get("complex_1_partners", ["B"]))
    track_2_label = _complex_label(shared, coupling.get("complex_2_partners", ["C"]))

    paths: list[Path] = []
    if "kappa" in diagnostics:
        kappa, labels = _diagnostic_array_for_model(
            diagnostics["kappa"],
            "kappa",
            model_index,
        )
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
            sample_labels=labels,
            dpi=dpi,
        )
        paths.append(kappa_path)

    if all(
        key in diagnostics
        for key in ("raw_kappa", "regularized_kappa", "kappa")
    ):
        raw_kappa, labels = _diagnostic_array_for_model(
            diagnostics["raw_kappa"],
            "raw_kappa",
            model_index,
        )
        regularized_kappa, regularized_labels = _diagnostic_array_for_model(
            diagnostics["regularized_kappa"],
            "regularized_kappa",
            model_index,
        )
        applied_kappa, applied_labels = _diagnostic_array_for_model(
            diagnostics["kappa"],
            "kappa",
            model_index,
        )
        if labels != regularized_labels or labels != applied_labels:
            raise ValueError("kappa stage diagnostic labels do not match")
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            raw_kappa.shape[0],
        )
        stages_path = plot_prefix.with_name(
            f"{plot_prefix.name}_kappa_stages.png"
        )
        _plot_kappa_stages(
            stages_path,
            normalized_t=normalized_t,
            raw_kappa=raw_kappa,
            regularized_kappa=regularized_kappa,
            applied_kappa=applied_kappa,
            sample_labels=labels,
            kappa_min=float(coupling.get("proxy_kappa_min", -1.0)),
            kappa_max=float(coupling.get("proxy_kappa_max", 2.0)),
            dpi=dpi,
        )
        paths.append(stages_path)

    if "reliability" in diagnostics:
        reliability, labels = _diagnostic_array_for_model(
            diagnostics["reliability"],
            "reliability",
            model_index,
        )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            reliability.shape[0],
        )
        reliability_path = plot_prefix.with_name(
            f"{plot_prefix.name}_kappa_reliability.png"
        )
        _plot_kappa_reliability(
            reliability_path,
            normalized_t=normalized_t,
            reliability=reliability,
            sample_labels=labels,
            rho=float(
                coupling.get("proxy_kappa_regularization_rho", 0.0)
            ),
            dpi=dpi,
        )
        paths.append(reliability_path)

    if "relative_denominator" in diagnostics:
        relative_denominator, labels = _diagnostic_array_for_model(
            diagnostics["relative_denominator"],
            "relative_denominator",
            model_index,
        )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            relative_denominator.shape[0],
        )
        relative_denominator_path = plot_prefix.with_name(
            f"{plot_prefix.name}_kappa_relative_denominator.png"
        )
        _plot_relative_denominator(
            relative_denominator_path,
            normalized_t=normalized_t,
            relative_denominator=relative_denominator,
            sample_labels=labels,
            rho=float(
                coupling.get("proxy_kappa_regularization_rho", 0.0)
            ),
            dpi=dpi,
        )
        paths.append(relative_denominator_path)

    if "proxy_residual" in diagnostics:
        residual, labels = _diagnostic_array_for_model(
            diagnostics["proxy_residual"],
            "proxy_residual",
            model_index,
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
            sample_labels=labels,
            dpi=dpi,
        )
        paths.append(residual_path)

    if "regularized_proxy_residual" in diagnostics:
        regularized_residual, labels = _diagnostic_array_for_model(
            diagnostics["regularized_proxy_residual"],
            "regularized_proxy_residual",
            model_index,
        )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            regularized_residual.shape[0],
        )
        regularized_residual_path = plot_prefix.with_name(
            f"{plot_prefix.name}_regularized_proxy_residual.png"
        )
        _plot_proxy_residual(
            regularized_residual_path,
            normalized_t=normalized_t,
            residual=regularized_residual,
            sample_labels=labels,
            dpi=dpi,
            stage_label="after regularization, before clamping",
        )
        paths.append(regularized_residual_path)

    if "denominator" in diagnostics:
        denominator, labels = _diagnostic_array_for_model(
            diagnostics["denominator"],
            "denominator",
            model_index,
        )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            denominator.shape[0],
        )
        denominator_path = plot_prefix.with_name(
            f"{plot_prefix.name}_kappa_denominator.png"
        )
        _plot_kappa_denominator(
            denominator_path,
            normalized_t=normalized_t,
            denominator=denominator,
            sample_labels=labels,
            dpi=dpi,
        )
        paths.append(denominator_path)

    if "numerator" in diagnostics:
        numerator, labels = _diagnostic_array_for_model(
            diagnostics["numerator"],
            "numerator",
            model_index,
        )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            numerator.shape[0],
        )
        numerator_path = plot_prefix.with_name(
            f"{plot_prefix.name}_kappa_numerator.png"
        )
        _plot_kappa_numerator(
            numerator_path,
            normalized_t=normalized_t,
            numerator=numerator,
            sample_labels=labels,
            dpi=dpi,
        )
        paths.append(numerator_path)

    cosine_specs = (
        (
            "kappa_subset",
            "cosine_delta_1_mix_kappa_subset",
            "cosine_delta_2_mix_kappa_subset",
            "Kappa-subset update/mix cosine similarity",
        ),
        (
            "all_shared",
            "cosine_delta_1_mix_all_shared",
            "cosine_delta_2_mix_all_shared",
            "All-shared-atom update/mix cosine similarity",
        ),
    )
    for suffix, track_1_key, track_2_key, title in cosine_specs:
        if track_1_key not in diagnostics or track_2_key not in diagnostics:
            continue
        cosine_1, labels = _diagnostic_array_for_model(
            diagnostics[track_1_key],
            track_1_key,
            model_index,
        )
        cosine_2, labels_2 = _diagnostic_array_for_model(
            diagnostics[track_2_key],
            track_2_key,
            model_index,
        )
        if cosine_1.shape != cosine_2.shape:
            raise ValueError(
                f"cosine diagnostic shape mismatch for {suffix}: "
                f"{cosine_1.shape} versus {cosine_2.shape}"
            )
        if labels != labels_2:
            raise ValueError(
                f"cosine diagnostic labels differ for {suffix}: "
                f"{labels!r} versus {labels_2!r}"
            )
        normalized_t = _normalized_t_axis(
            diagnostics.get("normalized_t"),
            cosine_1.shape[0],
        )
        cosine_path = plot_prefix.with_name(
            f"{plot_prefix.name}_cosine_{suffix}.png"
        )
        _plot_cosine_similarity(
            cosine_path,
            normalized_t=normalized_t,
            cosine_1=cosine_1,
            cosine_2=cosine_2,
            sample_labels=labels,
            title=title,
            track_1_label=track_1_label,
            track_2_label=track_2_label,
            max_samples=max_cosine_samples,
            dpi=dpi,
        )
        paths.append(cosine_path)

    return paths


def _plot_prefix_for_json(json_path: Path) -> tuple[Path, int | None]:
    """Return a normalized batch plot prefix and source model index."""

    stem = json_path.with_suffix("")
    marker = "_model_"
    stem_text = str(stem)
    if marker not in stem_text:
        return _coupling_plot_prefix(stem), None

    prefix, suffix = stem_text.rsplit(marker, 1)
    try:
        model_index = int(suffix)
    except ValueError:
        return _coupling_plot_prefix(stem), None
    return _coupling_plot_prefix(Path(prefix)), model_index


def _coupling_plot_prefix(prefix: Path) -> Path:
    """Collapse track/merged output variants to one coupling plot prefix."""

    prefix_text = str(prefix)
    for output_suffix in (
        "_merged_track1",
        "_merged_track2",
        "_merged",
        "_track1",
        "_track2",
    ):
        if prefix_text.endswith(output_suffix):
            return Path(f"{prefix_text[: -len(output_suffix)]}_coupling")
    return prefix


def _diagnostic_array_for_model(
    values,
    name: str,
    model_index: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Return diagnostic values and labels for a batch or one generated model."""

    array = _as_step_sample_array(values, name)
    if model_index is None:
        labels = [f"sample {idx}" for idx in range(array.shape[1])]
        return array, labels
    if model_index < 0:
        raise ValueError(f"model index must be non-negative, got {model_index}")
    if model_index >= array.shape[1]:
        raise ValueError(
            f"model index {model_index} is out of range for {name} "
            f"diagnostics with {array.shape[1]} sample(s)"
        )
    return array[:, model_index : model_index + 1], [f"model {model_index}"]


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
    sample_labels: list[str],
    dpi: int,
) -> None:
    """Plot kappa trajectories with track-leaning reference lines."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, kappa, sample_labels)
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


def _plot_kappa_stages(
    path: Path,
    *,
    normalized_t: np.ndarray,
    raw_kappa: np.ndarray,
    regularized_kappa: np.ndarray,
    applied_kappa: np.ndarray,
    sample_labels: list[str],
    kappa_min: float,
    kappa_max: float,
    dpi: int,
) -> None:
    """Compare raw, regularized, and applied kappa trajectories."""

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    stage_specs = (
        ("Raw kappa", raw_kappa, True),
        ("Regularized kappa before clamp", regularized_kappa, True),
        ("Applied kappa after clamp", applied_kappa, False),
    )
    for ax, (title, values, use_symlog) in zip(axes, stage_specs):
        _plot_samples(ax, normalized_t, values, sample_labels)
        ax.axhline(0.5, color="0.45", linestyle=":", linewidth=1.2)
        ax.axhline(kappa_min, color="0.35", linestyle="--", linewidth=1.0)
        ax.axhline(kappa_max, color="0.35", linestyle="--", linewidth=1.0)
        if use_symlog:
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(title)
        ax.set_ylabel("kappa")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(
        "Normalized denoising progress t (0 = noisiest, 1 = final)"
    )
    axes[-1].set_xlim(0.0, 1.0)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_kappa_reliability(
    path: Path,
    *,
    normalized_t: np.ndarray,
    reliability: np.ndarray,
    sample_labels: list[str],
    rho: float,
    dpi: int,
) -> None:
    """Plot the scale-aware reliability applied to raw kappa."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, reliability, sample_labels)
    ax.axhline(0.5, color="0.45", linestyle=":", linewidth=1.2)
    ax.set_title(f"Kappa reliability over denoising (rho={rho:g})")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("D / (D + rho S)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_relative_denominator(
    path: Path,
    *,
    normalized_t: np.ndarray,
    relative_denominator: np.ndarray,
    sample_labels: list[str],
    rho: float,
    dpi: int,
) -> None:
    """Plot D/S, the dimensionless conditioning measure for the solve."""

    positive_values = np.where(
        relative_denominator > 0,
        relative_denominator,
        np.nan,
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, positive_values, sample_labels)
    if rho > 0:
        ax.axhline(
            rho,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label=f"rho={rho:g} (reliability=0.5)",
        )
    ax.set_yscale("log")
    ax.set_title("Relative kappa denominator over denoising")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("D / S")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_proxy_residual(
    path: Path,
    *,
    normalized_t: np.ndarray,
    residual: np.ndarray,
    sample_labels: list[str],
    dpi: int,
    stage_label: str = "after kappa clamping",
) -> None:
    """Plot post-clamp proxy residual trajectories."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, residual, sample_labels)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_title("Proxy residual over denoising")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel(f"proxy residual {stage_label}")
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


def _plot_kappa_denominator(
    path: Path,
    *,
    normalized_t: np.ndarray,
    denominator: np.ndarray,
    sample_labels: list[str],
    dpi: int,
) -> None:
    """Plot kappa denominator trajectories."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, denominator, sample_labels)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_title("Kappa denominator over denoising")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("kappa denominator")
    ax.set_xlim(0.0, 1.0)
    max_abs = float(np.nanmax(np.abs(denominator))) if denominator.size else 1.0
    max_abs = max(max_abs, 1e-6)
    # ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=12)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_kappa_numerator(
    path: Path,
    *,
    normalized_t: np.ndarray,
    numerator: np.ndarray,
    sample_labels: list[str],
    dpi: int,
) -> None:
    """Plot kappa numerator trajectories."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _plot_samples(ax, normalized_t, numerator, sample_labels)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_title("Kappa numerator over denoising")
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("kappa numerator")
    ax.set_xlim(0.0, 1.0)
    max_abs = float(np.nanmax(np.abs(numerator))) if numerator.size else 1.0
    max_abs = max(max_abs, 1e-6)
    ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    ax.legend(loc="best", fontsize=12)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



def _plot_cosine_similarity(
    path: Path,
    *,
    normalized_t: np.ndarray,
    cosine_1: np.ndarray,
    cosine_2: np.ndarray,
    sample_labels: list[str],
    title: str,
    track_1_label: str,
    track_2_label: str,
    max_samples: int,
    dpi: int,
) -> None:
    """Plot paired delta/mix cosine trajectories with one hue per sample."""

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    n_samples = cosine_1.shape[1]
    n_plot = min(n_samples, max_samples)
    cmap = plt.get_cmap("tab10")

    for sample_idx in range(n_plot):
        label = (
            sample_labels[sample_idx]
            if sample_idx < len(sample_labels)
            else f"sample {sample_idx}"
        )
        base_color = cmap(sample_idx % cmap.N)
        light_color = _blend_with_white(base_color, 0.62)
        dark_color = _blend_with_black(base_color, 0.18)
        ax.plot(
            normalized_t,
            cosine_1[:, sample_idx],
            color=light_color,
            linewidth=1.8,
            label=f"{label}: cos(delta_1, delta_mix)",
        )
        ax.plot(
            normalized_t,
            cosine_2[:, sample_idx],
            color=dark_color,
            linewidth=1.8,
            label=f"{label}: cos(delta_2, delta_mix)",
        )

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
    ax.axhline(-1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Normalized denoising progress t (0 = noisiest, 1 = final)")
    ax.set_ylabel("cosine similarity")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    note = (
        f"Light shade: cos(delta_1 from {track_1_label}, delta_mix). "
        f"Dark shade: cos(delta_2 from {track_2_label}, delta_mix). "
        "Hue identifies the diffusion-batch sample."
    )
    if n_plot < n_samples:
        note += f" Showing first {n_plot} of {n_samples} samples."
    ax.text(0.0, -0.24, note, transform=ax.transAxes, fontsize=8)
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_samples(
    ax,
    normalized_t: np.ndarray,
    values: np.ndarray,
    sample_labels: list[str],
) -> None:
    """Plot one line per diffusion-batch sample."""

    for sample_idx in range(values.shape[1]):
        label = (
            sample_labels[sample_idx]
            if sample_idx < len(sample_labels)
            else f"sample {sample_idx}"
        )
        ax.plot(
            normalized_t,
            values[:, sample_idx],
            linewidth=1.6,
            label=label,
        )


def _blend_with_white(color, fraction: float):
    """Return `color` blended toward white by `fraction`."""

    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - fraction) + fraction)


def _blend_with_black(color, fraction: float):
    """Return `color` blended toward black by `fraction`."""

    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - fraction))


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
