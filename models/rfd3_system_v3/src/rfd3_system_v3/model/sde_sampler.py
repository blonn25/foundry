"""Strict reverse-SDE samplers for rfd3_system_v3.

The single-track and coupled samplers share RFD3's schedule, denoiser call, and
output conventions, but all state changes are explicit Euler-Maruyama updates.
Native churn, heuristic step scaling, and per-step rigid realignment are not
part of these trajectories.
"""

from __future__ import annotations

from typing import Any

import torch

from foundry.common import exists
from foundry.utils.ddp import RankedLogger
from rfd3_system_v3.model.inference_sampler import SampleDiffusionWithMotif
from rfd3_system_v3.system.chains import normalize_kappa_atom_subset
from rfd3_system_v3.system.sequence_control import mix_shared_sequence_logits
from rfd3_system_v3.system.stochastic_control import (
    build_guided_base_field,
    cosine_similarity_by_sample,
    norm_by_sample,
    reverse_sde_increment,
    reverse_sde_noise_scale,
    solve_stochastic_superdiff_kappa,
)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _print_progress(message: str) -> None:
    """Print sampler progress once even when Foundry suppresses INFO logging."""

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message, flush=True)


def _append_diagnostics(
    destination: dict[str, list[torch.Tensor]],
    values: dict[str, torch.Tensor],
) -> None:
    """Detach one step of per-sample diagnostics onto CPU."""

    for key in destination:
        destination[key].append(values[key].detach().cpu())


def _sequence_entropy(outs: dict[str, Any]) -> torch.Tensor | None:
    logits = outs.get("sequence_logits_I")
    if not exists(logits):
        return None
    probabilities = torch.softmax(logits, dim=-1).cpu()
    return -torch.sum(
        probabilities * torch.log(probabilities + 1e-10),
        dim=-1,
    )


class StrictReverseSDEMixin:
    """Validation shared by the single- and two-track reverse-SDE samplers."""

    def _validate_strict_reverse_sde(self) -> None:
        unsupported = []
        if self.use_classifier_free_guidance:
            unsupported.append("use_classifier_free_guidance")
        if self.allow_realignment:
            unsupported.append("allow_realignment")
        if self.s_jitter_origin > 0:
            unsupported.append("s_jitter_origin")
        if self.gamma_0 != 0:
            unsupported.append("gamma_0")
        if self.gamma_min != 0:
            unsupported.append("gamma_min")
        if self.noise_scale != 1:
            unsupported.append("noise_scale")
        if self.step_scale != 1:
            unsupported.append("step_scale")
        if unsupported:
            raise ValueError(
                "Strict reverse-SDE sampling does not support native RFD3 "
                "sampling heuristics: "
                + ", ".join(unsupported)
                + ". Use gamma_0=0, gamma_min=0, noise_scale=1, and "
                "step_scale=1."
            )


class SampleDiffusionReverseSDE(StrictReverseSDEMixin, SampleDiffusionWithMotif):
    """Single-track Euler-Maruyama sampler for RFD3's isotropic VE process."""

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        initializer_outputs: dict[str, Any],
        **_,
    ) -> dict[str, Any]:
        self._validate_strict_reverse_sde()
        device = coord_atom_lvl_to_be_noised.device
        fixed = f["is_motif_atom_with_fixed_coord"].to(
            device=device,
            dtype=torch.bool,
        )
        schedule = self._construct_inference_noise_schedule(
            device=device,
            partial_t=f.get("partial_t", None),
        )
        n_steps = len(schedule) - 1
        if n_steps < 1:
            raise ValueError("At least two schedule values are required.")

        batch_size = diffusion_batch_size
        X = self._get_initial_structure(
            schedule[0],
            batch_size,
            f["ref_element"].shape[0],
            coord_atom_lvl_to_be_noised.clone(),
            fixed,
        )
        normalized_t = self._noise_schedule_to_normalized_t(schedule)[:-1]
        progress_interval = max(1, n_steps // 20)
        noisy_traj: list[torch.Tensor] = []
        denoised_traj: list[torch.Tensor] = []
        sequence_entropy_traj: list[torch.Tensor] = []
        t_hats: list[torch.Tensor] = []
        diagnostic_names = (
            "sigma",
            "normalized_t",
            "delta_sigma",
            "brownian_scale",
            "velocity_norm",
            "score_norm",
            "drift_norm",
            "noise_norm",
            "update_norm",
            "noise_to_drift_ratio",
        )
        diagnostics = {name: [] for name in diagnostic_names}
        outs: dict[str, Any] = {}

        _print_progress(
            "Starting rfd3_system_v3 single-track reverse SDE: "
            f"{n_steps} steps, batch={batch_size}."
        )
        for step_num, (sigma, sigma_next) in enumerate(
            zip(schedule, schedule[1:])
        ):
            d_sigma = sigma_next - sigma
            outs = self._denoise_once(
                X_noisy_L=X,
                t_hat=sigma,
                D=batch_size,
                f=f,
                diffusion_module=diffusion_module,
                initializer_outputs=initializer_outputs,
                step_num=step_num,
            )
            X_denoised = outs["X_L"] if isinstance(outs, dict) else outs
            velocity = ((X - X_denoised) / sigma).clone()
            velocity[:, fixed, :] = 0

            brownian_scale = reverse_sde_noise_scale(sigma, d_sigma)
            noise = brownian_scale * torch.randn_like(X)
            noise[:, fixed, :] = 0
            increment = reverse_sde_increment(
                velocity,
                d_sigma=d_sigma,
                noise=noise,
            )
            X_next = X + increment
            X_next[:, fixed, :] = X[:, fixed, :]

            entropy = _sequence_entropy(outs)
            if entropy is not None:
                sequence_entropy_traj.append(entropy)
            noisy_traj.append(
                self.sigma_data * X / torch.sqrt(sigma**2 + self.sigma_data**2)
            )
            denoised_traj.append(X_denoised)
            t_hats.append(sigma)

            drift = increment - noise
            drift_norm = norm_by_sample(drift)
            noise_norm = norm_by_sample(noise)
            values = {
                "sigma": sigma.expand(batch_size),
                "normalized_t": normalized_t[step_num].expand(batch_size),
                "delta_sigma": d_sigma.expand(batch_size),
                "brownian_scale": brownian_scale.expand(batch_size),
                "velocity_norm": norm_by_sample(velocity),
                "score_norm": norm_by_sample(velocity) / sigma,
                "drift_norm": drift_norm,
                "noise_norm": noise_norm,
                "update_norm": norm_by_sample(increment),
                "noise_to_drift_ratio": noise_norm
                / drift_norm.clamp_min(torch.finfo(torch.float32).eps),
            }
            _append_diagnostics(diagnostics, values)
            X = X_next

            if (
                step_num == 0
                or step_num == n_steps - 1
                or (step_num + 1) % progress_interval == 0
            ):
                _print_progress(
                    "rfd3_system_v3 reverse-SDE step "
                    f"{step_num + 1}/{n_steps} | "
                    f"t={float(normalized_t[step_num]):.3f} | "
                    f"sigma={float(sigma):.4g} | "
                    "mean noise/drift="
                    f"{float(values['noise_to_drift_ratio'].mean()):.3g}"
                )

        return {
            "X_L": X,
            "X_noisy_L_traj": noisy_traj,
            "X_denoised_L_traj": denoised_traj,
            "t_hats": t_hats,
            "sequence_logits_I": outs.get("sequence_logits_I"),
            "sequence_indices_I": outs.get("sequence_indices_I"),
            "sequence_entropy_traj": sequence_entropy_traj,
            "sampling_metadata": {
                "implementation": "isotropic VE reverse SDE",
                "discretization": "Euler-Maruyama",
                "native_churn": False,
                "score_source": (
                    "RFD3 denoiser-derived score under the primary isotropic "
                    "Gaussian corruption."
                ),
                "auxiliary_com_perturbation_sde": "not modeled",
                "terminal_sigma": float(schedule[-1]),
                "diagnostics": diagnostics,
            },
        }


class SampleDiffusionWithSuperDiffSharedChainSDE(
    StrictReverseSDEMixin,
    SampleDiffusionWithMotif,
):
    """Stochastic SuperDiff AND sampler for two contexts sharing one chain."""

    def __init__(
        self,
        *,
        superdiff_guidance_scale: float = 1.0,
        superdiff_lift: float = 0.0,
        kappa_min: float | None = -1.0,
        kappa_max: float | None = 2.0,
        kappa_eps: float = 1e-8,
        kappa_atom_subset: str = "ALL",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.superdiff_guidance_scale = float(superdiff_guidance_scale)
        self.superdiff_lift = float(superdiff_lift)
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_eps = float(kappa_eps)
        self.kappa_atom_subset = normalize_kappa_atom_subset(kappa_atom_subset)

    def sample_diffusion_like_af3(self, **_):
        raise ValueError(
            "inference_sampler.kind='superdiff_shared_chain_sde' requires "
            "coupling_mode='superdiff_shared_chain_sde'."
        )

    def _validate_coupled_settings(self, reference: dict[str, Any] | None) -> None:
        self._validate_strict_reverse_sde()
        if self.superdiff_guidance_scale <= 0:
            raise ValueError("superdiff_guidance_scale must be greater than zero.")
        if self.superdiff_guidance_scale != 1.0 and reference is None:
            raise ValueError(
                "An isolated shared-chain reference is required when "
                "superdiff_guidance_scale != 1."
            )
        if self.kappa_eps <= 0:
            raise ValueError("kappa_eps must be greater than zero.")
        if (
            self.kappa_min is not None
            and self.kappa_max is not None
            and self.kappa_min > self.kappa_max
        ):
            raise ValueError("kappa_min cannot exceed kappa_max.")

    def _field_once(
        self,
        *,
        X: torch.Tensor,
        sigma: torch.Tensor,
        batch_size: int,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        initializer_outputs: dict[str, Any],
        step_num: int,
        fixed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        outs = self._denoise_once(
            X_noisy_L=X,
            t_hat=sigma,
            D=batch_size,
            f=f,
            diffusion_module=diffusion_module,
            initializer_outputs=initializer_outputs,
            step_num=step_num,
        )
        X_denoised = outs["X_L"] if isinstance(outs, dict) else outs
        velocity = ((X - X_denoised) / sigma).clone()
        velocity[:, fixed, :] = 0
        return velocity, X_denoised, outs

    def sample_coupled_superdiff_sde(
        self,
        *,
        track_1: dict[str, Any],
        track_2: dict[str, Any],
        reference: dict[str, Any] | None,
        shared_update_atom_indices_1: torch.Tensor,
        shared_update_atom_indices_2: torch.Tensor,
        shared_kappa_atom_indices_1: torch.Tensor,
        shared_kappa_atom_indices_2: torch.Tensor,
        shared_update_token_indices_1: torch.Tensor,
        shared_update_token_indices_2: torch.Tensor,
        reference_update_atom_indices: torch.Tensor | None,
        reference_kappa_atom_indices: torch.Tensor | None,
        reference_update_token_indices: torch.Tensor | None,
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coupling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_coupled_settings(reference)
        f1 = track_1["f"]
        f2 = track_2["f"]
        init1 = track_1["initializer_outputs"]
        init2 = track_2["initializer_outputs"]
        coord1 = track_1["coord_atom_lvl_to_be_noised"]
        coord2 = track_2["coord_atom_lvl_to_be_noised"]
        device = coord1.device

        update1 = shared_update_atom_indices_1.to(device=device, dtype=torch.long)
        update2 = shared_update_atom_indices_2.to(device=device, dtype=torch.long)
        kappa1 = shared_kappa_atom_indices_1.to(device=device, dtype=torch.long)
        kappa2 = shared_kappa_atom_indices_2.to(device=device, dtype=torch.long)
        token1 = shared_update_token_indices_1.to(device=device, dtype=torch.long)
        token2 = shared_update_token_indices_2.to(device=device, dtype=torch.long)
        if update1.numel() == 0 or kappa1.numel() == 0:
            raise ValueError("Shared update and kappa selections must be non-empty.")
        if update1.shape != update2.shape or kappa1.shape != kappa2.shape:
            raise ValueError("Paired shared-chain atom selections must have equal shapes.")
        if token1.shape != token2.shape:
            raise ValueError("Paired shared-chain token selections must have equal shapes.")

        fixed1 = f1["is_motif_atom_with_fixed_coord"].to(
            device=device, dtype=torch.bool
        )
        fixed2 = f2["is_motif_atom_with_fixed_coord"].to(
            device=device, dtype=torch.bool
        )
        if torch.any(fixed1[update1]) or torch.any(fixed2[update2]):
            raise ValueError("Shared update indices cannot include fixed motif atoms.")
        if torch.any(fixed1[kappa1]) or torch.any(fixed2[kappa2]):
            raise ValueError("Kappa indices cannot include fixed motif atoms.")

        use_reference = self.superdiff_guidance_scale != 1.0
        if use_reference:
            f0 = reference["f"]
            init0 = reference["initializer_outputs"]
            coord0 = reference["coord_atom_lvl_to_be_noised"]
            fixed0 = f0["is_motif_atom_with_fixed_coord"].to(
                device=device, dtype=torch.bool
            )
            update0 = reference_update_atom_indices.to(
                device=device, dtype=torch.long
            )
            kappa0 = reference_kappa_atom_indices.to(
                device=device, dtype=torch.long
            )
            token0 = reference_update_token_indices.to(
                device=device, dtype=torch.long
            )
            if (
                update0.shape != update1.shape
                or kappa0.shape != kappa1.shape
                or token0.shape != token1.shape
            ):
                raise ValueError(
                    "The isolated-A reference maps do not match the shared track maps."
                )
        else:
            f0 = init0 = coord0 = fixed0 = update0 = kappa0 = token0 = None

        schedule1 = self._construct_inference_noise_schedule(
            device=device,
            partial_t=f1.get("partial_t", None),
        )
        schedule2 = self._construct_inference_noise_schedule(
            device=device,
            partial_t=f2.get("partial_t", None),
        )
        if schedule1.shape != schedule2.shape or not torch.allclose(
            schedule1, schedule2
        ):
            raise ValueError("Both conditional tracks must use the same noise schedule.")
        schedule = schedule1
        batch_size = diffusion_batch_size
        n_steps = len(schedule) - 1
        if n_steps < 1:
            raise ValueError("At least two schedule values are required.")

        X1 = self._get_initial_structure(
            schedule[0],
            batch_size,
            f1["ref_element"].shape[0],
            coord1.clone(),
            fixed1,
        )
        X2 = self._get_initial_structure(
            schedule[0],
            batch_size,
            f2["ref_element"].shape[0],
            coord2.clone(),
            fixed2,
        )
        X2[:, update2, :] = X1[:, update1, :]
        if use_reference:
            X0 = self._get_initial_structure(
                schedule[0],
                batch_size,
                f0["ref_element"].shape[0],
                coord0.clone(),
                fixed0,
            )
            X0[:, update0, :] = X1[:, update1, :]
        else:
            X0 = None

        normalized_t = self._noise_schedule_to_normalized_t(schedule)[:-1]
        progress_interval = max(1, n_steps // 20)
        noisy1_traj: list[torch.Tensor] = []
        noisy2_traj: list[torch.Tensor] = []
        denoised1_traj: list[torch.Tensor] = []
        denoised2_traj: list[torch.Tensor] = []
        entropy1_traj: list[torch.Tensor] = []
        entropy2_traj: list[torch.Tensor] = []
        t_hats: list[torch.Tensor] = []
        diagnostic_names = (
            "sigma",
            "normalized_t",
            "delta_sigma",
            "brownian_scale",
            "raw_kappa",
            "kappa",
            "numerator",
            "denominator",
            "field_difference_norm_sq",
            "relative_field_difference",
            "degenerate",
            "clamped",
            "raw_delta_log_q_1",
            "raw_delta_log_q_2",
            "raw_density_difference",
            "raw_density_residual",
            "delta_log_q_1",
            "delta_log_q_2",
            "density_difference",
            "density_residual",
            "target_density_difference",
            "noise_projection_on_field_difference",
            "v_0_norm",
            "v_1_norm",
            "v_2_norm",
            "mixed_field_norm",
            "cosine_v1_mixed_kappa_subset",
            "cosine_v2_mixed_kappa_subset",
            "cosine_v1_mixed_all_shared",
            "cosine_v2_mixed_all_shared",
            "score_1_norm_all_shared",
            "score_2_norm_all_shared",
            "drift_A_norm",
            "noise_A_norm",
            "update_A_norm",
            "noise_to_drift_A_ratio",
            "density_residual_per_coordinate",
        )
        diagnostics = {name: [] for name in diagnostic_names}
        outs1: dict[str, Any] = {}
        outs2: dict[str, Any] = {}
        outs0: dict[str, Any] = {}
        final_kappa: torch.Tensor | None = None

        _print_progress(
            "Starting rfd3_system_v3 stochastic SuperDiff AND: "
            f"{n_steps} steps, batch={batch_size}, "
            f"subset={self.kappa_atom_subset}, "
            f"guidance={self.superdiff_guidance_scale:g}."
        )
        for step_num, (sigma, sigma_next) in enumerate(
            zip(schedule, schedule[1:])
        ):
            d_sigma = sigma_next - sigma
            X2[:, update2, :] = X1[:, update1, :]
            v1_full, X1_denoised, outs1 = self._field_once(
                X=X1,
                sigma=sigma,
                batch_size=batch_size,
                f=f1,
                diffusion_module=diffusion_module,
                initializer_outputs=init1,
                step_num=step_num,
                fixed=fixed1,
            )
            v2_full, X2_denoised, outs2 = self._field_once(
                X=X2,
                sigma=sigma,
                batch_size=batch_size,
                f=f2,
                diffusion_module=diffusion_module,
                initializer_outputs=init2,
                step_num=step_num,
                fixed=fixed2,
            )
            if use_reference:
                X0[:, update0, :] = X1[:, update1, :]
                v0_full, _, outs0 = self._field_once(
                    X=X0,
                    sigma=sigma,
                    batch_size=batch_size,
                    f=f0,
                    diffusion_module=diffusion_module,
                    initializer_outputs=init0,
                    step_num=step_num,
                    fixed=fixed0,
                )
                v0_update = v0_full[:, update0, :]
                v0_kappa = v0_full[:, kappa0, :]
            else:
                v0_update = v0_kappa = None

            brownian_scale = reverse_sde_noise_scale(sigma, d_sigma)
            noise1 = brownian_scale * torch.randn_like(X1)
            noise2 = brownian_scale * torch.randn_like(X2)
            shared_noise = brownian_scale * torch.randn_like(X1[:, update1, :])
            noise1[:, fixed1, :] = 0
            noise2[:, fixed2, :] = 0
            noise1[:, update1, :] = shared_noise
            noise2[:, update2, :] = shared_noise

            v1_kappa = v1_full[:, kappa1, :]
            v2_kappa = v2_full[:, kappa2, :]
            mixed_kappa, kappa_diag = solve_stochastic_superdiff_kappa(
                v1_kappa,
                v2_kappa,
                shared_noise=noise1[:, kappa1, :],
                sigma=sigma,
                d_sigma=d_sigma,
                guidance_scale=self.superdiff_guidance_scale,
                v_0=v0_kappa,
                lift=self.superdiff_lift,
                num_steps=n_steps,
                kappa_min=self.kappa_min,
                kappa_max=self.kappa_max,
                eps=self.kappa_eps,
            )
            final_kappa = kappa_diag.kappa
            kappa_view = final_kappa.to(v1_full.dtype).reshape(
                batch_size, 1, 1
            )
            v1_update = v1_full[:, update1, :]
            v2_update = v2_full[:, update2, :]
            base_update = build_guided_base_field(
                v2_update,
                guidance_scale=self.superdiff_guidance_scale,
                v_0=v0_update,
            )
            mixed_update = base_update + (
                self.superdiff_guidance_scale
                * kappa_view
                * (v1_update - v2_update)
            )

            increment1 = reverse_sde_increment(
                v1_full,
                d_sigma=d_sigma,
                noise=noise1,
            )
            increment2 = reverse_sde_increment(
                v2_full,
                d_sigma=d_sigma,
                noise=noise2,
            )
            shared_increment = reverse_sde_increment(
                mixed_update,
                d_sigma=d_sigma,
                noise=shared_noise,
            )
            X1_next = X1 + increment1
            X2_next = X2 + increment2
            shared_next = X1[:, update1, :] + shared_increment
            X1_next[:, update1, :] = shared_next
            X2_next[:, update2, :] = shared_next
            X1_next[:, fixed1, :] = X1[:, fixed1, :]
            X2_next[:, fixed2, :] = X2[:, fixed2, :]

            entropy1 = _sequence_entropy(outs1)
            entropy2 = _sequence_entropy(outs2)
            if entropy1 is not None:
                entropy1_traj.append(entropy1)
            if entropy2 is not None:
                entropy2_traj.append(entropy2)
            noisy1_traj.append(
                self.sigma_data * X1 / torch.sqrt(sigma**2 + self.sigma_data**2)
            )
            noisy2_traj.append(
                self.sigma_data * X2 / torch.sqrt(sigma**2 + self.sigma_data**2)
            )
            denoised1_traj.append(X1_denoised)
            denoised2_traj.append(X2_denoised)
            t_hats.append(sigma)

            A_drift = shared_increment - shared_noise
            A_drift_norm = norm_by_sample(A_drift)
            A_noise_norm = norm_by_sample(shared_noise)
            values = {
                field: getattr(kappa_diag, field)
                for field in kappa_diag.__dataclass_fields__
            }
            values.update(
                {
                    "sigma": sigma.expand(batch_size),
                    "normalized_t": normalized_t[step_num].expand(batch_size),
                    "delta_sigma": d_sigma.expand(batch_size),
                    "brownian_scale": brownian_scale.expand(batch_size),
                    "cosine_v1_mixed_kappa_subset": cosine_similarity_by_sample(
                        v1_kappa, mixed_kappa, eps=self.kappa_eps
                    ),
                    "cosine_v2_mixed_kappa_subset": cosine_similarity_by_sample(
                        v2_kappa, mixed_kappa, eps=self.kappa_eps
                    ),
                    "cosine_v1_mixed_all_shared": cosine_similarity_by_sample(
                        v1_update, mixed_update, eps=self.kappa_eps
                    ),
                    "cosine_v2_mixed_all_shared": cosine_similarity_by_sample(
                        v2_update, mixed_update, eps=self.kappa_eps
                    ),
                    "score_1_norm_all_shared": norm_by_sample(v1_update) / sigma,
                    "score_2_norm_all_shared": norm_by_sample(v2_update) / sigma,
                    "drift_A_norm": A_drift_norm,
                    "noise_A_norm": A_noise_norm,
                    "update_A_norm": norm_by_sample(shared_increment),
                    "noise_to_drift_A_ratio": A_noise_norm
                    / A_drift_norm.clamp_min(torch.finfo(torch.float32).eps),
                    "density_residual_per_coordinate": (
                        kappa_diag.density_residual / float(kappa1.numel() * 3)
                    ),
                }
            )
            _append_diagnostics(diagnostics, values)

            X1, X2 = X1_next, X2_next
            if (
                step_num == 0
                or step_num == n_steps - 1
                or (step_num + 1) % progress_interval == 0
            ):
                _print_progress(
                    "rfd3_system_v3 stochastic AND step "
                    f"{step_num + 1}/{n_steps} | "
                    f"t={float(normalized_t[step_num]):.3f} | "
                    f"sigma={float(sigma):.4g} | "
                    f"kappa={float(final_kappa.mean()):.3f} "
                    f"[{float(final_kappa.min()):.3f},"
                    f"{float(final_kappa.max()):.3f}] | "
                    "mean |Ito residual|="
                    f"{float(kappa_diag.density_residual.abs().mean()):.3e}"
                )

        if final_kappa is None:
            raise RuntimeError("Coupled reverse-SDE loop produced no kappa value.")
        logits1 = outs1.get("sequence_logits_I")
        logits2 = outs2.get("sequence_logits_I")
        indices1 = outs1.get("sequence_indices_I")
        indices2 = outs2.get("sequence_indices_I")
        if logits1 is not None and logits2 is not None and token1.numel() > 0:
            logits1, logits2 = mix_shared_sequence_logits(
                logits1,
                logits2,
                token_indices_1=token1,
                token_indices_2=token2,
                kappa=final_kappa,
                guidance_scale=self.superdiff_guidance_scale,
                logits_0=outs0.get("sequence_logits_I") if use_reference else None,
                token_indices_0=token0,
            )
            indices1 = (
                indices1.clone()
                if indices1 is not None
                else torch.argmax(logits1, dim=-1)
            )
            indices2 = (
                indices2.clone()
                if indices2 is not None
                else torch.argmax(logits2, dim=-1)
            )
            indices1[:, token1] = torch.argmax(logits1[:, token1, :], dim=-1)
            indices2[:, token2] = torch.argmax(logits2[:, token2, :], dim=-1)

        metadata = {
            **coupling_metadata,
            "implementation": "stochastic SuperDiff AND on isotropic VE reverse SDE",
            "density_estimator": "finite-step Ito estimator",
            "discretization": "Euler-Maruyama",
            "native_churn": False,
            "superdiff_guidance_scale": self.superdiff_guidance_scale,
            "superdiff_lift": self.superdiff_lift,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "kappa_eps": self.kappa_eps,
            "kappa_atom_subset": self.kappa_atom_subset,
            "terminal_sigma": float(schedule[-1]),
            "diagnostics": diagnostics,
        }
        return {
            "track_1": {
                "X_L": X1,
                "X_noisy_L_traj": noisy1_traj,
                "X_denoised_L_traj": denoised1_traj,
                "t_hats": t_hats,
                "sequence_logits_I": logits1,
                "sequence_indices_I": indices1,
                "sequence_entropy_traj": entropy1_traj,
            },
            "track_2": {
                "X_L": X2,
                "X_noisy_L_traj": noisy2_traj,
                "X_denoised_L_traj": denoised2_traj,
                "t_hats": t_hats,
                "sequence_logits_I": logits2,
                "sequence_indices_I": indices2,
                "sequence_entropy_traj": entropy2_traj,
            },
            "coupling_metadata": metadata,
        }
