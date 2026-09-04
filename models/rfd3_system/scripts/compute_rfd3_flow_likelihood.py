#!/usr/bin/env python3
"""Estimate an RFD3 coordinate log likelihood with the probability-flow ODE.

This is a post-hoc research diagnostic, not a calibrated thermodynamic energy.
The supplied structure is the low-noise endpoint.  Fixed atoms are conditioning;
all remaining RFD3 coordinate slots form the random variable being scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class LikelihoodSettings:
    """Numerical settings after defaults have been resolved."""

    sequence_conditioning: str
    integration_intervals: int = 50
    hutchinson_probes: int = 5
    probe_seed: int = 123
    precision: str = "float32"
    gauge: str = "auto"
    sigma_min: float | None = None
    sigma_max: float | None = None
    schedule_power: float | None = None
    progress_every: int = 5


def load_config(path: Path) -> tuple[str, dict[str, Any], LikelihoodSettings]:
    """Load the deliberately small YAML interface and reject ambiguous options."""

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("The scoring config must be a YAML mapping.")
    allowed_top = {"checkpoint_path", "specification", "likelihood"}
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown_top)}")

    checkpoint = raw.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint_path must be a non-empty string.")
    specification = raw.get("specification", {})
    if not isinstance(specification, dict):
        raise ValueError("specification must be a YAML mapping.")
    if "select_fixed_atoms" not in specification:
        raise ValueError(
            "specification.select_fixed_atoms is required; use false to score "
            "all atoms."
        )

    # These fields would alter, discard, or regenerate the supplied coordinates.
    managed = {
        "input",
        "atom_array_input",
        "partial_t",
        "contig",
        "unindex",
        "length",
        "ori_token",
        "infer_ori_strategy",
        "symmetry",
    }
    overlap = managed & set(specification)
    if overlap:
        raise ValueError(
            "The likelihood scorer manages exact-structure loading and centering; "
            f"remove these specification keys: {sorted(overlap)}"
        )

    likelihood = raw.get("likelihood")
    if not isinstance(likelihood, dict):
        raise ValueError("likelihood must be a YAML mapping.")
    allowed_likelihood = set(LikelihoodSettings.__dataclass_fields__)
    unknown_likelihood = set(likelihood) - allowed_likelihood
    if unknown_likelihood:
        raise ValueError(f"Unknown likelihood keys: {sorted(unknown_likelihood)}")
    if "sequence_conditioning" not in likelihood:
        raise ValueError(
            "likelihood.sequence_conditioning must explicitly be masked or observed."
        )
    settings = LikelihoodSettings(**likelihood)
    validate_settings(settings)

    expected_unfixed = settings.sequence_conditioning == "masked"
    supplied_unfixed = specification.get("select_unfixed_sequence")
    if supplied_unfixed is not None and supplied_unfixed is not expected_unfixed:
        raise ValueError(
            "specification.select_unfixed_sequence conflicts with "
            f"sequence_conditioning={settings.sequence_conditioning!r}."
        )
    specification = dict(specification)
    specification["select_unfixed_sequence"] = expected_unfixed
    return checkpoint, specification, settings


def validate_settings(settings: LikelihoodSettings) -> None:
    if settings.sequence_conditioning not in {"masked", "observed"}:
        raise ValueError("sequence_conditioning must be masked or observed.")
    if settings.integration_intervals < 1:
        raise ValueError("integration_intervals must be at least 1.")
    if settings.hutchinson_probes < 1:
        raise ValueError("hutchinson_probes must be at least 1.")
    if settings.precision not in {"float32", "bf16_mixed"}:
        raise ValueError("precision must be float32 or bf16_mixed.")
    if settings.gauge not in {"auto", "as_supplied"}:
        raise ValueError("gauge must be auto or as_supplied.")
    if settings.progress_every < 1:
        raise ValueError("progress_every must be at least 1.")
    for name in ("sigma_min", "sigma_max", "schedule_power"):
        value = getattr(settings, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and greater than zero.")
    if (
        settings.sigma_min is not None
        and settings.sigma_max is not None
        and settings.sigma_min >= settings.sigma_max
    ):
        raise ValueError("sigma_min must be smaller than sigma_max.")


def karras_sigma(
    u: float | torch.Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigma(u) and d sigma / du for low-to-high-noise integration."""

    u_tensor = torch.as_tensor(u, dtype=torch.float64)
    low = sigma_min ** (1.0 / power)
    high = sigma_max ** (1.0 / power)
    root = low + u_tensor * (high - low)
    sigma = root**power
    derivative = power * (high - low) * root ** (power - 1.0)
    return sigma, derivative


def make_rademacher_probes(
    shape: torch.Size,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Create fixed probes; reusing them makes the estimated ODE deterministic."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    probes = []
    for _ in range(count):
        values = torch.randint(
            0, 2, shape, generator=generator, device=device, dtype=torch.int8
        )
        probes.append(values.to(torch.float32).mul_(2).sub_(1))
    return probes


def field_and_hutchinson_divergence(
    field_fn: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    probes: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a field once and estimate its divergence by reverse-mode VJPs."""

    x = coordinates.detach().requires_grad_(True)
    field = field_fn(x)
    if field.shape != x.shape:
        raise ValueError(
            f"Field shape {tuple(field.shape)} does not match {tuple(x.shape)}."
        )
    if not torch.isfinite(field).all():
        raise FloatingPointError("The probability-flow field is non-finite.")
    estimates = []
    for index, probe in enumerate(probes):
        if probe.shape != x.shape:
            raise ValueError("Every Hutchinson probe must match the coordinate shape.")
        vector_jacobian = torch.autograd.grad(
            outputs=field,
            inputs=x,
            grad_outputs=probe.to(field.dtype),
            retain_graph=index + 1 < len(probes),
            create_graph=False,
        )[0]
        estimates.append(torch.sum(probe * vector_jacobian.float()))
    divergence = torch.stack(estimates).mean()
    if not torch.isfinite(divergence):
        raise FloatingPointError("The Hutchinson divergence estimate is non-finite.")
    return field.detach().float(), divergence.detach().float()


def integrate_probability_flow_rk4(
    initial_coordinates: torch.Tensor,
    *,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_min: float,
    sigma_max: float,
    schedule_power: float,
    intervals: int,
    probes: list[torch.Tensor],
    progress_every: int = 0,
) -> tuple[torch.Tensor, float, list[dict[str, float | int]]]:
    """Integrate coordinates and the density correction from data to noise."""

    x = initial_coordinates.detach().float()
    correction = torch.zeros((), device=x.device, dtype=torch.float32)
    trace: list[dict[str, float | int]] = []
    step_size = 1.0 / intervals

    def rhs(x_at_u: torch.Tensor, u: float) -> tuple[torch.Tensor, torch.Tensor]:
        sigma, derivative = karras_sigma(
            u,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            power=schedule_power,
        )
        sigma_value = float(sigma)
        derivative_value = float(derivative)

        def sigma_field(active: torch.Tensor) -> torch.Tensor:
            denoised = denoiser_fn(active, sigma_value)
            return (active - denoised.float()) / sigma_value

        field, divergence = field_and_hutchinson_divergence(
            sigma_field, x_at_u, probes
        )
        return derivative_value * field, derivative_value * divergence

    for step in range(intervals):
        u0 = step * step_size
        u_mid = u0 + 0.5 * step_size
        u1 = u0 + step_size
        k1_x, k1_d = rhs(x, u0)
        k2_x, k2_d = rhs(x + 0.5 * step_size * k1_x, u_mid)
        k3_x, k3_d = rhs(x + 0.5 * step_size * k2_x, u_mid)
        k4_x, k4_d = rhs(x + step_size * k3_x, u1)

        weighted_divergence = (k1_d + 2 * k2_d + 2 * k3_d + k4_d) / 6
        x = x + step_size * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        correction = correction + step_size * weighted_divergence
        if not torch.isfinite(x).all() or not torch.isfinite(correction):
            raise FloatingPointError(
                f"Probability-flow integration became non-finite at interval {step + 1}."
            )
        sigma0, _ = karras_sigma(
            u0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            power=schedule_power,
        )
        sigma1, _ = karras_sigma(
            u1,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            power=schedule_power,
        )
        trace.append(
            {
                "step": step + 1,
                "u_start": u0,
                "u_end": u1,
                "sigma_start": float(sigma0),
                "sigma_end": float(sigma1),
                "rk4_weighted_divergence_du": float(weighted_divergence),
                "cumulative_log_density_correction": float(correction),
                "active_coordinate_norm": float(torch.linalg.vector_norm(x)),
            }
        )
        if progress_every and (
            (step + 1) % progress_every == 0 or step + 1 == intervals
        ):
            print(
                f"ODE interval {step + 1}/{intervals}: sigma={float(sigma1):.6g}, "
                f"correction={float(correction):.6g}",
                flush=True,
            )

    return x, float(correction), trace


def gaussian_log_probability(coordinates: torch.Tensor, sigma: float) -> float:
    """Log probability under N(0, sigma^2 I) over every scalar coordinate."""

    flat = coordinates.double().reshape(-1)
    dimension = flat.numel()
    value = -0.5 * (
        torch.sum(flat.square()) / sigma**2
        + dimension * math.log(2 * math.pi * sigma**2)
    )
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_stem(path: Path) -> str:
    name = path.name
    for suffix in (".cif.gz", ".pdb.gz", ".cif", ".mmcif", ".pdb"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _find_rfd3_model(wrapped: torch.nn.Module) -> torch.nn.Module:
    """Resolve Foundry's optional Fabric/EMA wrappers to the inference network."""

    candidates = [wrapped]
    seen: set[int] = set()
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        for attribute in ("module", "shadow", "model", "_forward_module"):
            value = getattr(candidate, attribute, None)
            if value is not None and id(value) not in seen:
                candidates.append(value)
    for candidate in candidates:
        if hasattr(candidate, "token_initializer") and hasattr(
            candidate, "diffusion_module"
        ):
            return candidate
    raise RuntimeError("Could not locate the RFD3 inference network in model wrappers.")


def _resolve_schedule(
    model: torch.nn.Module, settings: LikelihoodSettings
) -> tuple[float, float, float, float]:
    sampler = getattr(model.inference_sampler, "sampler", model.inference_sampler)
    sigma_data = float(sampler.sigma_data)
    native_min = sigma_data * float(sampler.s_min)
    native_max = sigma_data * float(sampler.s_max)
    sigma_min = settings.sigma_min or native_min
    sigma_max = settings.sigma_max or native_max
    power = settings.schedule_power or float(sampler.p)
    if sigma_min >= sigma_max:
        raise ValueError("Resolved sigma_min must be smaller than sigma_max.")
    return sigma_data, sigma_min, sigma_max, power


def _prepare_pipeline_input(
    engine: Any,
    model: torch.nn.Module,
    input_path: Path,
    specification: dict[str, Any],
    settings: LikelihoodSettings,
    sigma_min: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use native RFD3 preprocessing while preserving the supplied structure."""

    from rfd3_system.inference.input_parsing import DesignInputSpecification

    spec_args = dict(specification)
    spec_args.update(input=str(input_path.resolve()), partial_t=sigma_min)
    spec = DesignInputSpecification.safe_init(**spec_args)

    source = spec.atom_array_input
    if source is None or len(source) == 0:
        raise ValueError("The input structure contains no atoms.")
    if not np.isfinite(source.coord).all():
        raise ValueError("The input contains non-finite physical coordinates.")

    fixed_source = source.is_motif_atom_with_fixed_coord.astype(bool)
    if settings.gauge == "auto":
        center_mask = fixed_source if fixed_source.any() else ~fixed_source
        center = np.mean(source.coord[center_mask], axis=0)
    else:
        center = np.zeros(3, dtype=float)
    spec.ori_token = center.astype(float).tolist()

    pipeline_output = engine.pipeline(
        spec.to_pipeline_input(example_id=input_path.stem)
    )
    pipeline_output = engine.trainer.fabric.to_device(pipeline_output)
    coordinates = pipeline_output["coord_atom_lvl_to_be_noised"]
    if coordinates.shape[0] != 1 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Unexpected RFD3 coordinate shape: {tuple(coordinates.shape)}"
        )
    if not torch.isfinite(coordinates).all():
        raise ValueError("RFD3 preprocessing produced non-finite coordinates.")

    features = pipeline_output["feats"]
    fixed = features["is_motif_atom_with_fixed_coord"].bool()
    virtual = features["is_virtual"].bool()
    if fixed.ndim != 1 or virtual.ndim != 1:
        raise ValueError("Expected one-dimensional RFD3 atom masks.")
    active = ~fixed
    if not torch.any(active):
        raise ValueError("No active coordinates remain after select_fixed_atoms.")

    with torch.no_grad(), engine.trainer.fabric.autocast():
        initializer_outputs = model.token_initializer(features)

    metadata = {
        "gauge": settings.gauge,
        "center_subtracted_angstrom": center.astype(float).tolist(),
        "atom_slots_total": int(active.numel()),
        "active_atom_slots": int(active.sum()),
        "fixed_atom_slots": int(fixed.sum()),
        "active_coordinate_dimension": int(active.sum()) * 3,
        "virtual_atom_slots": int(virtual.sum()),
        "active_virtual_atom_slots": int((active & virtual).sum()),
        "active_physical_atom_slots": int((active & ~virtual).sum()),
        "virtual_completion": (
            "canonical_rfd3_atom14_padding" if torch.any(virtual) else "none"
        ),
    }
    return (
        {
            "coordinates": coordinates.detach().float(),
            "features": features,
            "fixed_mask": fixed,
            "active_mask": active,
            "initializer_outputs": initializer_outputs,
        },
        metadata,
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = args.input.resolve()
    config_path = args.config.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    checkpoint_text, specification, settings = load_config(config_path)

    # Imports stay local so analytical tests do not need the complete Foundry stack.
    from rfd3_system.engine import RFD3InferenceConfig, RFD3InferenceEngine

    torch.set_float32_matmul_precision("high")
    engine_config = RFD3InferenceConfig(
        ckpt_path=checkpoint_text,
        diffusion_batch_size=1,
        inference_sampler={
            "kind": "default",
            "use_classifier_free_guidance": False,
            "gamma_0": 0.0,
            "step_scale": 1.0,
            "allow_realignment": False,
            "s_jitter_origin": 0.0,
        },
        specification={},
        seed=settings.probe_seed,
    )
    engine = RFD3InferenceEngine(**engine_config)
    precision_name = "32-true" if settings.precision == "float32" else "bf16-mixed"
    engine._assign_override("trainer.precision", precision_name)
    engine.initialize()
    model = _find_rfd3_model(engine.trainer.state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sigma_data, sigma_min, sigma_max, power = _resolve_schedule(model, settings)
    prepared, input_metadata = _prepare_pipeline_input(
        engine, model, input_path, specification, settings, sigma_min
    )
    full_initial = prepared["coordinates"]
    active_mask = prepared["active_mask"]
    active_initial = full_initial[:, active_mask, :].contiguous()
    fixed_template = full_initial.detach()
    features = prepared["features"]
    initializer_outputs = prepared["initializer_outputs"]
    diffusion_module = model.diffusion_module
    native_sampler = getattr(
        model.inference_sampler, "sampler", model.inference_sampler
    )
    autocast = engine.trainer.fabric.autocast

    def denoiser(active_coordinates: torch.Tensor, sigma: float) -> torch.Tensor:
        full = fixed_template.index_copy(
            1, torch.where(active_mask)[0], active_coordinates
        )
        sigma_tensor = torch.tensor(
            [sigma], device=full.device, dtype=full.dtype
        )
        # Neighbor selection is intentionally piecewise constant: RFD3 builds the
        # sparse attention graph under no_grad at each field evaluation.
        with autocast() if callable(autocast) else nullcontext():
            output = diffusion_module(
                X_noisy_L=full,
                t=sigma_tensor,
                f=dict(features),
                n_recycle=native_sampler.n_recycle,
                **initializer_outputs,
            )
        denoised = output["X_L"] if isinstance(output, dict) else output
        return denoised[:, active_mask, :]

    probes = make_rademacher_probes(
        active_initial.shape,
        count=settings.hutchinson_probes,
        seed=settings.probe_seed,
        device=active_initial.device,
    )
    started = time.time()
    terminal, correction, trace = integrate_probability_flow_rk4(
        active_initial,
        denoiser_fn=denoiser,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        schedule_power=power,
        intervals=settings.integration_intervals,
        probes=probes,
        progress_every=settings.progress_every,
    )
    runtime_seconds = time.time() - started
    terminal_log_prior = gaussian_log_probability(terminal, sigma_max)
    log_likelihood = terminal_log_prior + correction
    dimension = terminal.numel()
    checkpoint_path = Path(engine.ckpt_path)

    result = {
        "method": "rfd3_probability_flow_ode_hutchinson_v1",
        "interpretation": "approximate conditional coordinate log likelihood",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "specification": specification,
        "likelihood_settings": asdict(settings),
        "resolved_schedule": {
            "sigma_data": sigma_data,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "schedule_power": power,
        },
        "state": input_metadata,
        "terminal_log_prior": terminal_log_prior,
        "log_density_correction": correction,
        "estimated_log_likelihood": log_likelihood,
        "estimated_negative_log_likelihood": -log_likelihood,
        "log_likelihood_per_active_coordinate": log_likelihood / dimension,
        "negative_log_likelihood_per_active_coordinate": -log_likelihood / dimension,
        "terminal_normalized_squared_radius": float(
            terminal.double().square().sum() / (dimension * sigma_max**2)
        ),
        "runtime_seconds": runtime_seconds,
        "warnings": [
            "This is a learned-flow energy, not a calibrated thermodynamic "
            "free energy.",
            "The scalar isotropic EDM interpretation omits RFD3's correlated "
            "COM training perturbation.",
            "Hutchinson probes approximate the high-dimensional divergence.",
            "Cleaned structures reconstruct omitted virtual atom14 slots canonically.",
            "Sparse attention neighbor identities are treated as piecewise constant.",
            "Compare scores only with identical topology, conditioning, schedule, "
            "checkpoint, and probe seed.",
        ],
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(input_path) + "_rfd3_flow_likelihood"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_trace.csv"
    if not args.overwrite and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(
            f"Output already exists for {stem}; pass --overwrite to replace it."
        )
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDB, CIF, or compressed CIF to score")
    parser.add_argument("--config", type=Path, required=True, help="Scoring YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    json_path, csv_path = run(parse_args())
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
