import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
from jaxtyping import Float
from rfd3_system_v2.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from rfd3_system_v2.model.cfg_utils import strip_X
from rfd3_system_v2.system.chains import normalize_kappa_atom_subset
from rfd3_system_v2.system.density_control import (
    build_guided_base_field,
    cosine_similarity_by_sample as density_cosine_similarity_by_sample,
    evaluate_density_rates,
    solve_deterministic_density_kappa,
)
from rfd3_system_v2.system.proxy import (
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
    kind: Literal[
        "default",
        "symmetry",
        "superdiff_shared_chain",
        "superdiff_shared_chain_density",
    ] = "default"

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
            "Starting rfd3_system_v2 shared-chain coupled denoising: "
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
                    "rfd3_system_v2 coupled denoising "
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


class SampleDiffusionWithSuperDiffSharedChainDensity(SampleDiffusionWithMotif):
    """Post-churn deterministic SuperDiff AND sampler for a shared chain.

    The two conditional denoiser fields are differentiated with forward-mode
    automatic differentiation. A shared Hutchinson probe estimates each field
    divergence on the configured shared-chain atom subset. The resulting kappa
    equalizes the estimated conditional density rates for the deterministic
    update that follows RFD3's stochastic churn.
    """

    def __init__(
        self,
        *,
        superdiff_guidance_scale: float = 1.0,
        superdiff_lift: float = 0.0,
        kappa_min: float | None = -1.0,
        kappa_max: float | None = 2.0,
        kappa_eps: float = 1e-8,
        kappa_atom_subset: str = "ALL",
        density_hutchinson_probes: int = 1,
        density_validation_probes: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.superdiff_guidance_scale = float(superdiff_guidance_scale)
        self.superdiff_lift = float(superdiff_lift)
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_eps = float(kappa_eps)
        self.kappa_atom_subset = normalize_kappa_atom_subset(kappa_atom_subset)
        self.density_hutchinson_probes = int(density_hutchinson_probes)
        self.density_validation_probes = int(density_validation_probes)

    def sample_diffusion_like_af3(self, **_):
        raise ValueError(
            "inference_sampler.kind='superdiff_shared_chain_density' requires "
            "coupling_mode='superdiff_shared_chain_density'."
        )

    def _validate_density_sampler_settings(self, reference: dict[str, Any] | None):
        unsupported = []
        if self.use_classifier_free_guidance:
            unsupported.append("use_classifier_free_guidance")
        if self.allow_realignment:
            unsupported.append("allow_realignment")
        if self.s_jitter_origin > 0:
            unsupported.append("s_jitter_origin")
        if unsupported:
            raise ValueError(
                "Density-controlled shared-chain coupling currently does not "
                "support "
                + ", ".join(unsupported)
                + "."
            )
        if self.superdiff_guidance_scale <= 0:
            raise ValueError("superdiff_guidance_scale must be greater than zero.")
        if self.superdiff_guidance_scale != 1.0 and reference is None:
            raise ValueError(
                "An isolated shared-chain reference is required when "
                "superdiff_guidance_scale != 1."
            )
        if self.density_hutchinson_probes < 1:
            raise ValueError("density_hutchinson_probes must be at least 1.")
        if self.density_validation_probes < 0:
            raise ValueError("density_validation_probes cannot be negative.")
        if self.kappa_eps <= 0:
            raise ValueError("kappa_eps must be greater than zero.")
        if (
            self.kappa_min is not None
            and self.kappa_max is not None
            and self.kappa_min > self.kappa_max
        ):
            raise ValueError("kappa_min cannot exceed kappa_max.")

    @staticmethod
    def _rademacher_like(value: torch.Tensor) -> torch.Tensor:
        """Sample a Rademacher vector in the field's floating-point dtype."""

        return (
            torch.empty_like(value)
            .bernoulli_(0.5)
            .mul_(2)
            .sub_(1)
        )

    def _field_and_divergence_jvp(
        self,
        *,
        X_noisy_L: torch.Tensor,
        selected_atom_indices: torch.Tensor,
        probes: list[torch.Tensor],
        t_hat: torch.Tensor,
        D: int,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        initializer_outputs: dict[str, Any],
        step_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Evaluate one conditional field and estimate ``-div(field)`` by JVP."""

        selected_coords = X_noisy_L[:, selected_atom_indices, :]
        field_selected = None
        outs = None
        dlog_estimates = []

        def selected_field_fn(coords):
            X_eval = X_noisy_L.index_copy(1, selected_atom_indices, coords)
            denoiser_out = self._denoise_once(
                X_noisy_L=X_eval,
                t_hat=t_hat,
                D=D,
                f=dict(f),
                diffusion_module=diffusion_module,
                initializer_outputs=initializer_outputs,
                step_num=step_num,
            )
            X_denoised = (
                denoiser_out["X_L"]
                if isinstance(denoiser_out, dict)
                else denoiser_out
            )
            field = (X_eval - X_denoised) / t_hat
            return field[:, selected_atom_indices, :], denoiser_out

        for probe in probes:
            try:
                primal, tangent, aux = torch.func.jvp(
                    selected_field_fn,
                    (selected_coords,),
                    (probe,),
                    has_aux=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Forward-mode JVP failed while estimating the RFD3 shared-chain "
                    "density change. rfd3_system_v2 intentionally has no VJP or "
                    "proxy fallback; this model/runtime combination must be "
                    "investigated before density coupling can be used."
                ) from exc
            if field_selected is None:
                field_selected = primal
                outs = aux
            dlog_estimates.append(
                -torch.sum(
                    probe.float() * tangent.float(),
                    dim=tuple(range(1, tangent.ndim)),
                )
            )

        return (
            field_selected,
            torch.stack(dlog_estimates, dim=0).mean(dim=0),
            outs,
        )

    def _field_once(
        self,
        *,
        X_noisy_L: torch.Tensor,
        t_hat: torch.Tensor,
        D: int,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        initializer_outputs: dict[str, Any],
        step_num: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        outs = self._denoise_once(
            X_noisy_L=X_noisy_L,
            t_hat=t_hat,
            D=D,
            f=dict(f),
            diffusion_module=diffusion_module,
            initializer_outputs=initializer_outputs,
            step_num=step_num,
        )
        X_denoised = outs["X_L"] if isinstance(outs, dict) else outs
        return (X_noisy_L - X_denoised) / t_hat, outs

    def sample_coupled_superdiff_density(
        self,
        *,
        track_1: dict[str, Any],
        track_2: dict[str, Any],
        reference: dict[str, Any] | None,
        shared_update_atom_indices_1: torch.Tensor,
        shared_update_atom_indices_2: torch.Tensor,
        shared_kappa_atom_indices_1: torch.Tensor,
        shared_kappa_atom_indices_2: torch.Tensor,
        reference_update_atom_indices: torch.Tensor | None,
        reference_kappa_atom_indices: torch.Tensor | None,
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coupling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run deterministic density-rate coupling after each stochastic churn."""

        self._validate_density_sampler_settings(reference)
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
        if update1.numel() == 0 or kappa1.numel() == 0:
            raise ValueError("Shared update and kappa atom selections must be non-empty.")
        if update1.shape != update2.shape or kappa1.shape != kappa2.shape:
            raise ValueError("Paired shared-chain atom selections must have equal shapes.")

        fixed1 = f1["is_motif_atom_with_fixed_coord"].to(device=device, dtype=torch.bool)
        fixed2 = f2["is_motif_atom_with_fixed_coord"].to(device=device, dtype=torch.bool)
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
            if update0.shape != update1.shape or kappa0.shape != kappa1.shape:
                raise ValueError(
                    "The isolated-A reference atom map does not match the shared "
                    "track atom maps."
                )
        else:
            f0 = init0 = coord0 = fixed0 = update0 = kappa0 = None

        schedule1 = self._construct_inference_noise_schedule(
            device=device, partial_t=f1.get("partial_t", None)
        )
        schedule2 = self._construct_inference_noise_schedule(
            device=device, partial_t=f2.get("partial_t", None)
        )
        if schedule1.shape != schedule2.shape or not torch.allclose(
            schedule1, schedule2
        ):
            raise ValueError("Both conditional tracks must use the same noise schedule.")
        schedule = schedule1
        D = diffusion_batch_size
        n_steps = len(schedule) - 1
        if n_steps < 1:
            raise ValueError("At least two schedule values are required.")

        X1 = self._get_initial_structure(
            schedule[0], D, f1["ref_element"].shape[0], coord1.clone(), fixed1
        )
        X2 = self._get_initial_structure(
            schedule[0], D, f2["ref_element"].shape[0], coord2.clone(), fixed2
        )
        X2[:, update2, :] = X1[:, update1, :]
        if use_reference:
            X0 = self._get_initial_structure(
                schedule[0], D, f0["ref_element"].shape[0], coord0.clone(), fixed0
            )
            X0[:, update0, :] = X1[:, update1, :]
        else:
            X0 = None

        normalized_t = self._noise_schedule_to_normalized_t(schedule)[:-1]
        progress_interval = max(1, n_steps // 20)
        noisy1_traj, noisy2_traj = [], []
        denoised1_traj, denoised2_traj = [], []
        entropy1_traj, entropy2_traj = [], []
        t_hats = []
        diagnostics = {
            key: []
            for key in (
                "kappa",
                "raw_kappa",
                "denominator",
                "field_difference_norm_sq",
                "degenerate",
                "clamped",
                "density_rate_1",
                "density_rate_2",
                "target_rate_difference",
                "density_rate_residual",
                "delta_log_q_1",
                "delta_log_q_2",
                "dlog_1",
                "dlog_2",
                "v_0_norm",
                "v_1_norm",
                "v_2_norm",
                "cosine_v1_path_kappa_subset",
                "cosine_v2_path_kappa_subset",
                "cosine_v1_path_all_shared",
                "cosine_v2_path_all_shared",
                "validation_density_rate_residual",
            )
        }
        outs1 = outs2 = {}

        ranked_logger.info(
            "Starting rfd3_system_v2 deterministic density coupling: "
            f"{n_steps} steps, batch={D}, subset={self.kappa_atom_subset}, "
            f"solve_probes={self.density_hutchinson_probes}, "
            f"validation_probes={self.density_validation_probes}, "
            f"guidance={self.superdiff_guidance_scale:g}."
        )

        for step_num, (sigma_before_churn, sigma_next) in enumerate(
            zip(schedule, schedule[1:])
        ):
            gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
            sigma = sigma_before_churn * (gamma + 1)
            d_sigma = sigma_next - sigma
            epsilon_scale = self.noise_scale * torch.sqrt(
                torch.square(sigma) - torch.square(sigma_before_churn)
            )
            epsilon1 = epsilon_scale * torch.randn_like(X1)
            epsilon2 = epsilon_scale * torch.randn_like(X2)
            epsilon_shared = epsilon_scale * torch.randn_like(X1[:, update1, :])
            epsilon1[:, fixed1, :] = 0
            epsilon2[:, fixed2, :] = 0
            epsilon1[:, update1, :] = epsilon_shared
            epsilon2[:, update2, :] = epsilon_shared

            X2[:, update2, :] = X1[:, update1, :]
            X1_hat = X1 + epsilon1
            X2_hat = X2 + epsilon2
            if use_reference:
                X0[:, update0, :] = X1[:, update1, :]
                X0_hat = X0.clone()
                X0_hat[:, update0, :] = X1_hat[:, update1, :]
            else:
                X0_hat = None

            solve_probes = [
                self._rademacher_like(X1_hat[:, kappa1, :])
                for _ in range(self.density_hutchinson_probes)
            ]
            v1_kappa, dlog1, outs1 = self._field_and_divergence_jvp(
                X_noisy_L=X1_hat,
                selected_atom_indices=kappa1,
                probes=solve_probes,
                t_hat=sigma,
                D=D,
                f=f1,
                diffusion_module=diffusion_module,
                initializer_outputs=init1,
                step_num=step_num,
            )
            v2_kappa, dlog2, outs2 = self._field_and_divergence_jvp(
                X_noisy_L=X2_hat,
                selected_atom_indices=kappa2,
                probes=solve_probes,
                t_hat=sigma,
                D=D,
                f=f2,
                diffusion_module=diffusion_module,
                initializer_outputs=init2,
                step_num=step_num,
            )
            X1_denoised = outs1["X_L"] if isinstance(outs1, dict) else outs1
            X2_denoised = outs2["X_L"] if isinstance(outs2, dict) else outs2
            v1_full = (X1_hat - X1_denoised) / sigma
            v2_full = (X2_hat - X2_denoised) / sigma

            if use_reference:
                v0_full, _ = self._field_once(
                    X_noisy_L=X0_hat,
                    t_hat=sigma,
                    D=D,
                    f=f0,
                    diffusion_module=diffusion_module,
                    initializer_outputs=init0,
                    step_num=step_num,
                )
                v0_kappa = v0_full[:, kappa0, :]
                v0_update = v0_full[:, update0, :]
            else:
                v0_full = v0_kappa = v0_update = None

            diag = solve_deterministic_density_kappa(
                v1_kappa,
                v2_kappa,
                dlog1,
                dlog2,
                sigma=sigma,
                d_sigma=d_sigma,
                step_scale=self.step_scale,
                guidance_scale=self.superdiff_guidance_scale,
                v_0=v0_kappa,
                lift=self.superdiff_lift,
                num_steps=n_steps,
                kappa_min=self.kappa_min,
                kappa_max=self.kappa_max,
                eps=self.kappa_eps,
            )
            kappa_view = diag.kappa.to(v1_full.dtype).reshape(D, 1, 1)
            v1_update = v1_full[:, update1, :]
            v2_update = v2_full[:, update2, :]
            base_update = build_guided_base_field(
                v2_update,
                guidance_scale=self.superdiff_guidance_scale,
                v_0=v0_update,
            )
            v_mix_update = base_update + (
                self.superdiff_guidance_scale
                * kappa_view
                * (v1_update - v2_update)
            )
            path_update = self.step_scale * v_mix_update

            base_kappa = build_guided_base_field(
                v2_kappa,
                guidance_scale=self.superdiff_guidance_scale,
                v_0=v0_kappa,
            )
            v_mix_kappa = base_kappa + (
                self.superdiff_guidance_scale
                * kappa_view
                * (v1_kappa - v2_kappa)
            )
            path_kappa = self.step_scale * v_mix_kappa

            validation_residual = torch.full_like(diag.kappa, torch.nan)
            if self.density_validation_probes:
                validation_probes = [
                    self._rademacher_like(X1_hat[:, kappa1, :])
                    for _ in range(self.density_validation_probes)
                ]
                _, validation_dlog1, _ = self._field_and_divergence_jvp(
                    X_noisy_L=X1_hat,
                    selected_atom_indices=kappa1,
                    probes=validation_probes,
                    t_hat=sigma,
                    D=D,
                    f=f1,
                    diffusion_module=diffusion_module,
                    initializer_outputs=init1,
                    step_num=step_num,
                )
                _, validation_dlog2, _ = self._field_and_divergence_jvp(
                    X_noisy_L=X2_hat,
                    selected_atom_indices=kappa2,
                    probes=validation_probes,
                    t_hat=sigma,
                    D=D,
                    f=f2,
                    diffusion_module=diffusion_module,
                    initializer_outputs=init2,
                    step_num=step_num,
                )
                validation_residual = evaluate_density_rates(
                    v1_kappa,
                    v2_kappa,
                    validation_dlog1,
                    validation_dlog2,
                    sigma=sigma,
                    d_sigma=d_sigma,
                    path_field=path_kappa,
                    target_rate_difference=diag.target_rate_difference,
                ).density_rate_residual

            X1_next = X1_hat + self.step_scale * d_sigma * v1_full
            X2_next = X2_hat + self.step_scale * d_sigma * v2_full
            shared_next = X1_hat[:, update1, :] + d_sigma * path_update
            X1_next[:, update1, :] = shared_next
            X2_next[:, update2, :] = shared_next
            if use_reference:
                X0_next = X0_hat + self.step_scale * d_sigma * v0_full
                X0_next[:, update0, :] = shared_next
                X0 = X0_next
            X1, X2 = X1_next, X2_next

            if exists(outs1.get("sequence_logits_I")):
                p1 = torch.softmax(outs1["sequence_logits_I"], dim=-1).cpu()
                entropy1_traj.append(-torch.sum(p1 * torch.log(p1 + 1e-10), dim=-1))
            if exists(outs2.get("sequence_logits_I")):
                p2 = torch.softmax(outs2["sequence_logits_I"], dim=-1).cpu()
                entropy2_traj.append(-torch.sum(p2 * torch.log(p2 + 1e-10), dim=-1))
            noisy1_traj.append(
                self.sigma_data * X1_hat / torch.sqrt(sigma**2 + self.sigma_data**2)
            )
            noisy2_traj.append(
                self.sigma_data * X2_hat / torch.sqrt(sigma**2 + self.sigma_data**2)
            )
            denoised1_traj.append(X1_denoised)
            denoised2_traj.append(X2_denoised)
            t_hats.append(sigma)

            value_map = {
                "dlog_1": dlog1,
                "dlog_2": dlog2,
                "validation_density_rate_residual": validation_residual,
                "cosine_v1_path_kappa_subset": density_cosine_similarity_by_sample(
                    v1_kappa, path_kappa, eps=self.kappa_eps
                ),
                "cosine_v2_path_kappa_subset": density_cosine_similarity_by_sample(
                    v2_kappa, path_kappa, eps=self.kappa_eps
                ),
                "cosine_v1_path_all_shared": density_cosine_similarity_by_sample(
                    v1_update, path_update, eps=self.kappa_eps
                ),
                "cosine_v2_path_all_shared": density_cosine_similarity_by_sample(
                    v2_update, path_update, eps=self.kappa_eps
                ),
            }
            for key in diagnostics:
                value = value_map.get(key, getattr(diag, key, None))
                diagnostics[key].append(value.detach().cpu())

            if (
                step_num == 0
                or step_num == n_steps - 1
                or (step_num + 1) % progress_interval == 0
            ):
                ranked_logger.info(
                    "rfd3_system_v2 density step "
                    f"{step_num + 1}/{n_steps} | "
                    f"t={float(normalized_t[step_num]):.3f} | "
                    f"sigma={float(sigma):.4g} | "
                    f"kappa={float(diag.kappa.mean()):.3f} "
                    f"[{float(diag.kappa.min()):.3f},"
                    f"{float(diag.kappa.max()):.3f}] | "
                    "mean |density residual|="
                    f"{float(diag.density_rate_residual.abs().mean()):.3e}"
                )

        diagnostics["t_hat"] = [value.detach().cpu() for value in t_hats]
        diagnostics["normalized_t"] = [
            value.detach().cpu() for value in normalized_t
        ]
        metadata = {
            **coupling_metadata,
            "implementation": "post-churn deterministic SuperDiff density-rate estimate",
            "density_estimator": "shared-probe Hutchinson JVP",
            "churn_density_accounting": "excluded",
            "superdiff_guidance_scale": self.superdiff_guidance_scale,
            "superdiff_lift": self.superdiff_lift,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "kappa_eps": self.kappa_eps,
            "kappa_atom_subset": self.kappa_atom_subset,
            "density_hutchinson_probes": self.density_hutchinson_probes,
            "density_validation_probes": self.density_validation_probes,
            "diagnostics": diagnostics,
        }
        return {
            "track_1": dict(
                X_L=X1,
                X_noisy_L_traj=noisy1_traj,
                X_denoised_L_traj=denoised1_traj,
                t_hats=t_hats,
                sequence_logits_I=outs1.get("sequence_logits_I"),
                sequence_indices_I=outs1.get("sequence_indices_I"),
                sequence_entropy_traj=entropy1_traj,
            ),
            "track_2": dict(
                X_L=X2,
                X_noisy_L_traj=noisy2_traj,
                X_denoised_L_traj=denoised2_traj,
                t_hats=t_hats,
                sequence_logits_I=outs2.get("sequence_logits_I"),
                sequence_indices_I=outs2.get("sequence_indices_I"),
                sequence_entropy_traj=entropy2_traj,
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
        "superdiff_shared_chain_density": SampleDiffusionWithSuperDiffSharedChainDensity,
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

    def sample_coupled_superdiff_density(self, **kwargs):
        if not hasattr(self.sampler, "sample_coupled_superdiff_density"):
            raise ValueError(
                "Deterministic density coupling requires "
                "inference_sampler.kind='superdiff_shared_chain_density'."
            )
        return self.sampler.sample_coupled_superdiff_density(**kwargs)

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
