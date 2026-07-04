import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
from jaxtyping import Float
from rfd3_system.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from rfd3_system.model.cfg_utils import strip_X
from rfd3_system.system.chains import normalize_kappa_atom_subset
from rfd3_system.system.proxy import (
    cosine_similarity_by_sample,
    solve_two_track_proxy_kappa,
)

from foundry.common import exists
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from foundry.utils.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(kw_only=True)
class SampleDiffusionConfig:
    kind: Literal["default", "symmetry", "superdiff_shared_chain"] = "default"

    # Standard EDM args
    num_timesteps: int = 200
    min_t: int = 0
    max_t: int = 1
    sigma_data: int = 16
    s_min: float = 4e-4
    s_max: int = 160
    p: int = 7
    gamma_0: float = 0.6
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    solver: Literal["af3"] = "af3"

    # RFD3 / design args
    center_option: str = "all"
    s_trans: float = 1.0
    s_jitter_origin: float = 0.0
    fraction_of_steps_to_fix_motif: float = 0.0
    skip_few_diffusion_steps: bool = False
    allow_realignment: bool = False
    insert_motif_at_end: bool = True
    use_classifier_free_guidance: bool = False
    cfg_scale: float = 2.0
    cfg_t_max: float | None = None

    # Recycling
    n_recycle: int | None = None  # Override model default n_recycle for inference


class SampleDiffusionWithMotif(SampleDiffusionConfig):
    """Diffusion sampler that supports optional motif alignment."""

    def _construct_inference_noise_schedule(
        self, device: torch.device, partial_t: float = None
    ) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        if partial_t is not None:
            # For now, partial t is a global parameter
            partial_t = float(partial_t.mean())
            noise_schedule = t_hat
            ranked_logger.info("Using partial diffusion with t={}".format(partial_t))

            # Debug the noise schedule filtering
            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(f"Filtered range: [{new_min:.3f}, {new_max:.3f}]")
            else:
                ranked_logger.warning(
                    f"No noise schedule steps found with t <= {partial_t}!"
                )
                ranked_logger.info(
                    f"Original schedule range: [{original_min:.3f}, {original_max:.3f}]"
                )
                # Fallback to smallest available step
                noise_schedule_original = self._construct_inference_noise_schedule(
                    device=device
                )
                noise_schedule = noise_schedule_original[-1:]  # Just use the final step
                ranked_logger.info(
                    f"Using fallback: final step with t={noise_schedule[0].item():.6f}"
                )
        else:
            noise_schedule = t_hat

        return noise_schedule

    def _noise_schedule_to_normalized_t(
        self,
        noise_schedule: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the AF3/RFD3 noise schedule back to the normalized t value.

        RFD3 stores and passes the physical noise scale `t_hat` to the denoiser,
        while users usually interpret the schedule by its normalized construction
        variable `t`.  This helper is only used for diagnostics and plotting.
        """

        base = torch.clamp(noise_schedule / self.sigma_data, min=0.0)
        numerator = base ** (1 / self.p) - self.s_max ** (1 / self.p)
        denominator = self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p)
        return numerator / denominator

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_coord, :] = 0  # Zero out noise going in
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def _denoise_once(
        self,
        *,
        X_noisy_L: torch.Tensor,
        t_hat: torch.Tensor,
        D: int,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        initializer_outputs: dict[str, Any],
        step_num: int,
    ) -> dict[str, Any]:
        """Run one denoiser call, preserving the existing chunked-mode path."""

        if "chunked_pairwise_embedder" in initializer_outputs:
            # Chunked mode: explicitly provide P_LL=None.  The pairwise embedder
            # object is stateful enough that we keep it in initializer_outputs.
            tic = time.time()
            chunked_embedder = initializer_outputs["chunked_pairwise_embedder"]
            other_outputs = {
                k: v
                for k, v in initializer_outputs.items()
                if k != "chunked_pairwise_embedder"
            }
            outs = diffusion_module(
                X_noisy_L=X_noisy_L,
                t=t_hat.tile(D),
                f=f,
                P_LL=None,
                chunked_pairwise_embedder=chunked_embedder,
                initializer_outputs=other_outputs,
                n_recycle=self.n_recycle,
                **other_outputs,
            )
            toc = time.time()
            ranked_logger.info(f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms")
            return outs

        return diffusion_module(
            X_noisy_L=X_noisy_L,
            t=t_hat.tile(D),
            f=f,
            n_recycle=self.n_recycle,
            **initializer_outputs,
        )

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]

        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )  # (D, L, 3)

        if self.s_jitter_origin > 0.0:
            X_L[:, is_motif_atom_with_fixed_coord, :] += torch.normal(
                mean=0.0,
                std=self.s_jitter_origin,
                size=(D, 1, 3),
                device=X_L.device,
            )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        threshold_step = (len(noise_schedule) - 1) * self.fraction_of_steps_to_fix_motif

        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, _ = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                    center_option=self.center_option,
                    # If centering_affects_motif is True, the model's predictions from (step_num-1) might affect the motif
                    centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                    # If keeping the motif position wrt the origin fixed, we can't do translational augmentation
                    # We want to keep this position fixed in the interval where the model is not allowed to change it
                    s_trans=self.s_trans if step_num >= threshold_step else 0.0,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0  # No noise injection for fixed atoms
            )
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            # Handle chunked mode vs standard mode
            if "chunked_pairwise_embedder" in initializer_outputs:
                # Chunked mode: explicitly provide P_LL=None
                tic = time.time()
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    n_recycle=self.n_recycle,
                    **initializer_outputs,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            if self.use_classifier_free_guidance and (
                self.cfg_t_max is None or c_t > self.cfg_t_max
            ):
                X_noisy_L_stripped = strip_X(X_noisy_L, f_ref)

                # unconditional forward pass
                outs_ref = diffusion_module(
                    X_noisy_L=X_noisy_L_stripped,  # modify X
                    t=t_hat.tile(D),
                    f=f_ref,  # modified f
                    n_recycle=self.n_recycle,
                    **ref_initializer_outputs,
                )

                X_denoised_L_stripped = outs_ref["X_L"]

                delta_L_ref = (
                    X_noisy_L_stripped - X_denoised_L_stripped
                ) / t_hat  # gradient of x wrt. t at x_t_hat

                # pad delta_L_ref with zeros to match delta_L (for the unindexed atoms)
                if delta_L_ref.shape[1] < delta_L.shape[1]:
                    delta_L_ref = torch.cat(
                        [
                            delta_L_ref,
                            torch.zeros_like(delta_L[:, delta_L_ref.shape[1] :, :]),
                        ],
                        dim=1,
                    )

                # apply CFG
                delta_L = delta_L + (self.cfg_scale - 1) * (delta_L - delta_L_ref)

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class SampleDiffusionWithSymmetry(SampleDiffusionWithMotif):
    """
    This class is a wrapper around the SampleDiffusionWithMotif class.
    It is used to sample diffusion with symmetry.
    """

    def __init__(self, sym_step_frac: float = 0.9, **kwargs):
        assert (
            kwargs.get("gamma_0") > 0.5
        ), "gamma_0 must be greater than 0.5 for symmetry sampling"
        self.sym_step_frac = sym_step_frac
        super().__init__(**kwargs)

    def apply_symmetry_to_X_L(self, X_L, f):
        # check that we are doing symmetric inference

        assert "sym_transform" in f.keys(), "Symmetry transform not found in f"

        # update symmetric frames to correct for change in global frame
        symmetry_feats = {k: v for k, v in f.items() if "sym" in k}

        # apply symmetry frame shift to X_L
        X_L = apply_symmetry_to_xyz_atomwise(
            X_L, symmetry_feats, partial_diffusion=("partial_t" in f)
        )

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        **_,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )  # (D, L, 3)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        # symmetrize X_L until the step gamma = gamma_min_sym
        gamma_min_sym_idx = min(
            int(len(noise_schedule) * self.sym_step_frac), len(noise_schedule) - 1
        )
        gamma_min_sym = noise_schedule[gamma_min_sym_idx]

        ranked_logger.info(f"gamma_min_sym: {gamma_min_sym}")
        ranked_logger.info(f"gamma_min: {self.gamma_min}")
        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, R = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0  # No noise injection for fixed atoms
            )

            # NOTE: no symmetry applied to the noisy structure
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            # Handle chunked mode vs standard mode (same as default sampler)
            if "chunked_pairwise_embedder" in initializer_outputs:
                # Chunked mode: explicitly provide P_LL=None
                tic = time.time()
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    n_recycle=self.n_recycle,
                    **initializer_outputs,
                )
            # apply symmetry to X_denoised_L
            if "X_L" in outs and c_t > gamma_min_sym:
                # outs["original_X_L"] = outs["X_L"].clone()
                outs["X_L"] = self.apply_symmetry_to_X_L(outs["X_L"], f)

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            # NOTE: no classifier-free guidance for symmetry

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            # delta_L should be symmetric
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, R = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # apply symmetry frame shift to X_L
            X_L = self.apply_symmetry_to_X_L(X_L, f)

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class SampleDiffusionWithSuperDiffSharedChainProxy(SampleDiffusionWithMotif):
    """Approximate two-track shared-chain sampler.

    This is a research heuristic inspired by SuperDiff's AND operation.  It uses
    RFD3 denoiser deltas as score-like update proxies and solves a local proxy
    equation on the shared chain.  It is not exact SuperDiff density control.
    """

    def __init__(
        self,
        *,
        proxy_norm_weight: float = 1.0,
        proxy_kappa_min: float = -1.0,
        proxy_kappa_max: float = 2.0,
        proxy_eps: float = 1e-8,
        kappa_atom_subset: str = "ALL",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proxy_norm_weight = proxy_norm_weight
        self.proxy_kappa_min = proxy_kappa_min
        self.proxy_kappa_max = proxy_kappa_max
        self.proxy_eps = proxy_eps
        self.kappa_atom_subset = normalize_kappa_atom_subset(kappa_atom_subset)

    def sample_diffusion_like_af3(self, **_):
        raise ValueError(
            "inference_sampler.kind='superdiff_shared_chain' requires "
            "coupling_mode='superdiff_shared_chain'. It cannot run as an ordinary "
            "single-track sampler."
        )

    def _validate_proxy_sampler_settings(self):
        unsupported = []
        if self.use_classifier_free_guidance:
            unsupported.append("use_classifier_free_guidance")
        if self.allow_realignment:
            unsupported.append("allow_realignment")
        if self.s_jitter_origin > 0:
            unsupported.append("s_jitter_origin")
        if unsupported:
            raise ValueError(
                "Approximate shared-chain coupling currently does not support "
                + ", ".join(unsupported)
                + ". Disable these options or derive their coupled behavior first."
            )

    def sample_coupled_superdiff_proxy(
        self,
        *,
        track_1: dict[str, Any],
        track_2: dict[str, Any],
        shared_update_atom_indices_1: torch.Tensor,
        shared_update_atom_indices_2: torch.Tensor,
        shared_kappa_atom_indices_1: torch.Tensor | None = None,
        shared_kappa_atom_indices_2: torch.Tensor | None = None,
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coupling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run approximate coupled sampling for two context-specific tracks."""

        self._validate_proxy_sampler_settings()

        f1 = track_1["f"]
        f2 = track_2["f"]
        initializer_outputs_1 = track_1["initializer_outputs"]
        initializer_outputs_2 = track_2["initializer_outputs"]
        coord_1 = track_1["coord_atom_lvl_to_be_noised"]
        coord_2 = track_2["coord_atom_lvl_to_be_noised"]

        device = coord_1.device
        shared_update_atom_indices_1 = shared_update_atom_indices_1.to(
            device=device, dtype=torch.long
        )
        shared_update_atom_indices_2 = shared_update_atom_indices_2.to(
            device=device, dtype=torch.long
        )
        if shared_kappa_atom_indices_1 is None:
            shared_kappa_atom_indices_1 = shared_update_atom_indices_1
        else:
            shared_kappa_atom_indices_1 = shared_kappa_atom_indices_1.to(
                device=device, dtype=torch.long
            )
        if shared_kappa_atom_indices_2 is None:
            shared_kappa_atom_indices_2 = shared_update_atom_indices_2
        else:
            shared_kappa_atom_indices_2 = shared_kappa_atom_indices_2.to(
                device=device, dtype=torch.long
            )
        if shared_update_atom_indices_1.numel() == 0:
            raise ValueError(
                "No non-fixed shared-chain atoms were provided for coupled denoising."
            )
        if shared_kappa_atom_indices_1.numel() == 0:
            raise ValueError(
                "No non-fixed shared-chain atoms were provided for the kappa solve."
            )
        fixed_1 = f1["is_motif_atom_with_fixed_coord"].to(device=device, dtype=torch.bool)
        fixed_2 = f2["is_motif_atom_with_fixed_coord"].to(device=device, dtype=torch.bool)

        if shared_update_atom_indices_1.shape != shared_update_atom_indices_2.shape:
            raise ValueError(
                "Shared update atom index arrays must have the same shape."
            )
        if shared_kappa_atom_indices_1.shape != shared_kappa_atom_indices_2.shape:
            raise ValueError("Shared kappa atom index arrays must have the same shape.")
        if torch.any(fixed_1[shared_update_atom_indices_1]) or torch.any(
            fixed_2[shared_update_atom_indices_2]
        ):
            raise ValueError(
                "Shared update atom indices include fixed motif atoms. Fixed "
                "motifs must be excluded from the coupled kappa solve."
            )
        if torch.any(fixed_1[shared_kappa_atom_indices_1]) or torch.any(
            fixed_2[shared_kappa_atom_indices_2]
        ):
            raise ValueError(
                "Shared kappa atom indices include fixed motif atoms. Fixed "
                "motifs must be excluded from the coupled kappa solve."
            )

        noise_schedule_1 = self._construct_inference_noise_schedule(
            device=device,
            partial_t=f1.get("partial_t", None),
        )
        noise_schedule_2 = self._construct_inference_noise_schedule(
            device=device,
            partial_t=f2.get("partial_t", None),
        )
        if noise_schedule_1.shape != noise_schedule_2.shape or not torch.allclose(
            noise_schedule_1, noise_schedule_2
        ):
            raise ValueError(
                "Track noise schedules differ. Approximate shared-chain coupling "
                "requires the same timestep schedule for both contexts."
            )
        noise_schedule = noise_schedule_1

        D = diffusion_batch_size
        n_update_steps = len(noise_schedule) - 1
        if n_update_steps <= 0:
            raise ValueError(
                "Approximate shared-chain coupling requires at least two noise "
                "schedule values so that one denoising update can be performed."
            )
        normalized_t_values = self._noise_schedule_to_normalized_t(noise_schedule)[:-1]
        progress_interval = max(1, n_update_steps // 20)
        ranked_logger.info(
            "Starting rfd3_system shared-chain coupled denoising: "
            f"{n_update_steps} steps, diffusion_batch_size={D}, "
            f"kappa_atom_subset={self.kappa_atom_subset}, "
            f"kappa_solve_atoms={shared_kappa_atom_indices_1.numel()}, "
            f"shared_update_atoms={shared_update_atom_indices_1.numel()}."
        )

        X1_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=f1["ref_element"].shape[0],
            coord_atom_lvl_to_be_noised=coord_1.clone(),
            is_motif_atom_with_fixed_coord=fixed_1,
        )
        X2_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=f2["ref_element"].shape[0],
            coord_atom_lvl_to_be_noised=coord_2.clone(),
            is_motif_atom_with_fixed_coord=fixed_2,
        )

        # Track 1 owns the shared state. Track 2's shared coordinates are
        # overwritten at initialization and before every denoiser call.
        X2_L[:, shared_update_atom_indices_2, :] = X1_L[
            :, shared_update_atom_indices_1, :
        ]

        X1_noisy_traj = []
        X2_noisy_traj = []
        X1_denoised_traj = []
        X2_denoised_traj = []
        seq_entropy_traj_1 = []
        seq_entropy_traj_2 = []
        t_hats = []
        proxy_diag = {
            "kappa": [],
            "raw_kappa": [],
            "denominator": [],
            "degenerate": [],
            "proxy_residual": [],
            "delta_1_norm": [],
            "delta_2_norm": [],
            "cosine_delta_1_mix_kappa_subset": [],
            "cosine_delta_2_mix_kappa_subset": [],
            "cosine_delta_1_mix_all_shared": [],
            "cosine_delta_2_mix_all_shared": [],
        }
        proxy_diag_fields = (
            "kappa",
            "raw_kappa",
            "denominator",
            "degenerate",
            "proxy_residual",
            "delta_1_norm",
            "delta_2_norm",
        )

        outs1: dict[str, Any] = {}
        outs2: dict[str, Any] = {}
        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X1_L.requires_grad and not X2_L.requires_grad

            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            t_hat = c_t_minus_1 * (gamma + 1)
            step_scale = self.step_scale
            d_t = c_t - t_hat

            eps_scale = self.noise_scale * torch.sqrt(
                torch.square(t_hat) - torch.square(c_t_minus_1)
            )
            epsilon_1 = eps_scale * torch.normal(
                mean=0.0, std=1.0, size=X1_L.shape, device=device
            )
            epsilon_2 = eps_scale * torch.normal(
                mean=0.0, std=1.0, size=X2_L.shape, device=device
            )
            epsilon_shared = eps_scale * torch.normal(
                mean=0.0,
                std=1.0,
                size=X1_L[:, shared_update_atom_indices_1, :].shape,
                device=device,
            )
            epsilon_1[:, fixed_1, :] = 0
            epsilon_2[:, fixed_2, :] = 0
            epsilon_1[:, shared_update_atom_indices_1, :] = epsilon_shared
            epsilon_2[:, shared_update_atom_indices_2, :] = epsilon_shared

            X2_L[:, shared_update_atom_indices_2, :] = X1_L[
                :, shared_update_atom_indices_1, :
            ]
            X1_noisy_L = X1_L + epsilon_1
            X2_noisy_L = X2_L + epsilon_2

            outs1 = self._denoise_once(
                X_noisy_L=X1_noisy_L,
                t_hat=t_hat,
                D=D,
                f=f1,
                diffusion_module=diffusion_module,
                initializer_outputs=initializer_outputs_1,
                step_num=step_num,
            )
            outs2 = self._denoise_once(
                X_noisy_L=X2_noisy_L,
                t_hat=t_hat,
                D=D,
                f=f2,
                diffusion_module=diffusion_module,
                initializer_outputs=initializer_outputs_2,
                step_num=step_num,
            )

            X1_denoised_L = outs1["X_L"] if "X_L" in outs1 else outs1
            X2_denoised_L = outs2["X_L"] if "X_L" in outs2 else outs2

            delta_1 = (X1_noisy_L - X1_denoised_L) / t_hat
            delta_2 = (X2_noisy_L - X2_denoised_L) / t_hat
            delta_A_1 = delta_1[:, shared_update_atom_indices_1, :]
            delta_A_2 = delta_2[:, shared_update_atom_indices_2, :]
            delta_A_1_kappa = delta_1[:, shared_kappa_atom_indices_1, :]
            delta_A_2_kappa = delta_2[:, shared_kappa_atom_indices_2, :]

            diag = solve_two_track_proxy_kappa(
                delta_A_1_kappa,
                delta_A_2_kappa,
                norm_weight=self.proxy_norm_weight,
                kappa_min=self.proxy_kappa_min,
                kappa_max=self.proxy_kappa_max,
                eps=self.proxy_eps,
            )
            kappa_view = diag.kappa.reshape((D, 1, 1))
            delta_A_mix = kappa_view * delta_A_1 + (1 - kappa_view) * delta_A_2
            delta_A_mix_kappa = kappa_view * delta_A_1_kappa + (
                1 - kappa_view
            ) * delta_A_2_kappa
            cosine_delta_1_mix_kappa_subset = cosine_similarity_by_sample(
                delta_A_1_kappa,
                delta_A_mix_kappa,
                eps=self.proxy_eps,
            )
            cosine_delta_2_mix_kappa_subset = cosine_similarity_by_sample(
                delta_A_2_kappa,
                delta_A_mix_kappa,
                eps=self.proxy_eps,
            )
            cosine_delta_1_mix_all_shared = cosine_similarity_by_sample(
                delta_A_1,
                delta_A_mix,
                eps=self.proxy_eps,
            )
            cosine_delta_2_mix_all_shared = cosine_similarity_by_sample(
                delta_A_2,
                delta_A_mix,
                eps=self.proxy_eps,
            )

            if (
                step_num == 0
                or step_num == n_update_steps - 1
                or (step_num + 1) % progress_interval == 0
            ):
                kappa_values = diag.kappa.detach().cpu()
                residual_values = diag.proxy_residual.detach().abs().cpu()
                normalized_t = float(normalized_t_values[step_num].detach().cpu())
                ranked_logger.info(
                    "rfd3_system coupled denoising "
                    f"step {step_num + 1}/{n_update_steps} | "
                    f"t={normalized_t:.3f} | "
                    f"t_hat={float(t_hat.detach().cpu()):.4g} | "
                    "kappa mean/min/max="
                    f"{float(kappa_values.mean()):.3f}/"
                    f"{float(kappa_values.min()):.3f}/"
                    f"{float(kappa_values.max()):.3f} | "
                    f"mean |proxy_residual|={float(residual_values.mean()):.3e}"
                )

            X1_next = X1_noisy_L + step_scale * d_t * delta_1
            X2_next = X2_noisy_L + step_scale * d_t * delta_2
            shared_next = X1_noisy_L[:, shared_update_atom_indices_1, :] + (
                step_scale * d_t * delta_A_mix
            )
            X1_next[:, shared_update_atom_indices_1, :] = shared_next
            X2_next[:, shared_update_atom_indices_2, :] = shared_next

            X1_L = X1_next
            X2_L = X2_next

            if exists(outs1.get("sequence_logits_I")):
                p1 = torch.softmax(outs1["sequence_logits_I"], dim=-1).cpu()
                seq_entropy_traj_1.append(-torch.sum(p1 * torch.log(p1 + 1e-10), dim=-1))
            if exists(outs2.get("sequence_logits_I")):
                p2 = torch.softmax(outs2["sequence_logits_I"], dim=-1).cpu()
                seq_entropy_traj_2.append(-torch.sum(p2 * torch.log(p2 + 1e-10), dim=-1))

            X1_noisy_traj.append(
                self.sigma_data
                * X1_noisy_L
                / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )
            X2_noisy_traj.append(
                self.sigma_data
                * X2_noisy_L
                / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )
            X1_denoised_traj.append(X1_denoised_L)
            X2_denoised_traj.append(X2_denoised_L)
            t_hats.append(t_hat)

            for key in proxy_diag_fields:
                proxy_diag[key].append(getattr(diag, key).detach().cpu())
            proxy_diag["cosine_delta_1_mix_kappa_subset"].append(
                cosine_delta_1_mix_kappa_subset.detach().cpu()
            )
            proxy_diag["cosine_delta_2_mix_kappa_subset"].append(
                cosine_delta_2_mix_kappa_subset.detach().cpu()
            )
            proxy_diag["cosine_delta_1_mix_all_shared"].append(
                cosine_delta_1_mix_all_shared.detach().cpu()
            )
            proxy_diag["cosine_delta_2_mix_all_shared"].append(
                cosine_delta_2_mix_all_shared.detach().cpu()
            )

        proxy_diag["t_hat"] = [t_hat.detach().cpu() for t_hat in t_hats]
        proxy_diag["normalized_t"] = [
            t_value.detach().cpu() for t_value in normalized_t_values
        ]
        metadata = {
            **coupling_metadata,
            "approximation": (
                "RFD3 denoiser deltas are used as score-like proxies; this is "
                "not exact SuperDiff density control."
            ),
            "proxy_norm_weight": self.proxy_norm_weight,
            "proxy_kappa_min": self.proxy_kappa_min,
            "proxy_kappa_max": self.proxy_kappa_max,
            "proxy_eps": self.proxy_eps,
            "kappa_atom_subset": self.kappa_atom_subset,
            "kappa_solve_atom_count": int(shared_kappa_atom_indices_1.numel()),
            "shared_update_atom_count": int(shared_update_atom_indices_1.numel()),
            "diagnostics": proxy_diag,
        }

        return {
            "track_1": dict(
                X_L=X1_L,
                X_noisy_L_traj=X1_noisy_traj,
                X_denoised_L_traj=X1_denoised_traj,
                t_hats=t_hats,
                sequence_logits_I=outs1.get("sequence_logits_I"),
                sequence_indices_I=outs1.get("sequence_indices_I"),
                sequence_entropy_traj=seq_entropy_traj_1,
            ),
            "track_2": dict(
                X_L=X2_L,
                X_noisy_L_traj=X2_noisy_traj,
                X_denoised_L_traj=X2_denoised_traj,
                t_hats=t_hats,
                sequence_logits_I=outs2.get("sequence_logits_I"),
                sequence_indices_I=outs2.get("sequence_indices_I"),
                sequence_entropy_traj=seq_entropy_traj_2,
            ),
            "coupling_metadata": metadata,
        }


class ConditionalDiffusionSampler:
    """
    Conditional diffusion sampler, chooses at construction time which sampler to use,
    then forwards `sample_diffusion_like_af3` to the chosen sampler.
    If you write a new sampler, you best add it to the registry below
    and inference_sampler.kind in inference_engine config.
    """

    _registry = {
        "default": SampleDiffusionWithMotif,
        "symmetry": SampleDiffusionWithSymmetry,
        "superdiff_shared_chain": SampleDiffusionWithSuperDiffSharedChainProxy,
    }

    def __init__(self, kind="default", **kwargs):
        ranked_logger.info(
            f"Initializing ConditionalDiffusionSampler with kind: {kind}"
        )
        try:
            SamplerCls = self._registry[kind]
            # remove kwargs that the sampler cannot take
            init_args = self.get_class_init_args(SamplerCls)
            kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        except KeyError:
            raise ValueError(
                f"Invalid sampler kind: {kind}, must be one of {list(self._registry.keys())}"
            )
        self.sampler = SamplerCls(**kwargs)

    def sample_diffusion_like_af3(self, **kwargs):
        return self.sampler.sample_diffusion_like_af3(**kwargs)

    def sample_coupled_superdiff_proxy(self, **kwargs):
        if not hasattr(self.sampler, "sample_coupled_superdiff_proxy"):
            raise ValueError(
                "Coupled proxy sampling requires "
                "inference_sampler.kind='superdiff_shared_chain'."
            )
        return self.sampler.sample_coupled_superdiff_proxy(**kwargs)

    def get_class_init_args(self, cls):
        arg_names = []
        if hasattr(cls, "__init__") and callable(cls.__init__):
            for p_cls in cls.__mro__:
                if "__init__" in p_cls.__dict__ and p_cls is not object:
                    signature = inspect.signature(p_cls.__init__)
                    arg_names.extend(
                        [param.name for param in signature.parameters.values()]
                    )
        return arg_names


def centre_random_augment_around_motif(
    X_L: torch.Tensor,  # (D, L, 3) noisy diffused coordinates
    coord_atom_lvl_to_be_noised: torch.Tensor,  # (D, L, 3) original coordinates
    is_motif_atom_with_fixed_coord: torch.Tensor,  # (D, L) indices in original coordinates to be kept constant
    s_trans: float = 1.0,
    center_option: str = "all",
    centering_affects_motif: bool = True,
    reinsert_motif=True,
):
    D, L, _ = X_L.shape

    if reinsert_motif and torch.any(is_motif_atom_with_fixed_coord):
        # ... Align original coordinates to the prediction
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_coord, :],
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_coord, :],
        )

        # ... Insert original coordinates into X_L
        X_L[..., is_motif_atom_with_fixed_coord, :] = coords_with_gt_aligned

    # ... Centering
    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center = torch.mean(
                X_L[..., is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of motif atoms
        elif center_option == "diffuse":
            center = torch.mean(
                X_L[..., ~is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of diffused atoms

        else:
            center = torch.mean(X_L, dim=-2, keepdim=True)
    else:
        center = torch.mean(X_L, dim=-2, keepdim=True)

    # ... Center
    if centering_affects_motif:
        X_L = X_L - center
    else:
        X_L[..., ~is_motif_atom_with_fixed_coord, :] = (
            X_L[..., ~is_motif_atom_with_fixed_coord, :] - center
        )

    # ... Random augmentation
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = (
        torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans
    )  # (D, 1, 3)
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L, R
