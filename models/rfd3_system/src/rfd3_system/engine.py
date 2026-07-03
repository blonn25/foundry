import json
import logging
import os
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import yaml
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack
from omegaconf import DictConfig, ListConfig
from toolz import merge_with

from foundry.common import exists
from foundry.inference_engines.base import BaseInferenceEngine
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from rfd3_system.constants import SAVED_CONDITIONING_ANNOTATIONS
from rfd3_system.inference.datasets import (
    assemble_distributed_inference_loader_from_json,
)
from rfd3_system.inference.input_parsing import (
    DesignInputSpecification,
    ensure_input_is_abspath,
)
from rfd3_system.model.inference_sampler import SampleDiffusionConfig
from rfd3_system.system.chains import (
    assert_matching_shared_chain,
    build_shared_update_atom_map,
    chain_mask,
    merge_tracks_with_shared_source,
    normalize_kappa_atom_subset,
    relabel_nonshared_chains,
    select_kappa_solve_atom_indices,
    subset_by_chains,
)
from rfd3_system.utils.inference import (
    ensure_inference_sampler_matches_design_spec,
)
from rfd3_system.utils.io import (
    CIF_LIKE_EXTENSIONS,
    build_stack_from_atom_array_and_batched_coords,
    extract_example_id_from_path,
    find_files_with_extension,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(kw_only=True)
class RFD3InferenceConfig:
    ckpt_path: str | Path = (
        "rfd3"  # Defaults to foundry installation upon instantiation
    )
    diffusion_batch_size: int = 16

    # RFD3 specific
    skip_existing: bool = True
    json_keys_subset: Optional[List[str]] = None
    specification: Optional[dict] = field(default_factory=dict)
    inference_sampler: SampleDiffusionConfig | dict = field(default_factory=dict)

    # Approximate shared-chain coupling prototype. This mode is off by default.
    coupling_mode: Optional[str] = None
    shared_chain_id: str = "A"
    complex_1_partners: List[str] = field(default_factory=lambda: ["B"])
    complex_2_partners: List[str] = field(default_factory=lambda: ["C"])
    track_1_specification: Optional[dict] = field(default_factory=dict)
    track_2_specification: Optional[dict] = field(default_factory=dict)
    merged_output_policy: Literal["track1", "track2", "both", "none"] = "track1"

    # Saving args
    cleanup_guideposts: bool = True
    cleanup_virtual_atoms: bool = True
    read_sequence_from_sequence_head: bool = True
    output_full_json: bool = True

    # Prefix to add to all output samples
    # Default: None      -> f'{jsonfilebasename}_{jsonkey}_{batch}_{model}'
    # Otherwise: string  -> f'{string}{jsonkey}_{batch}_{model}'
    # e.g. Empty string  -> f'{jsonkey}_{batch}_{model}'
    # e.g. Chunk string  -> f'{chunkprefix_}{jsonkey}_{batch}_{model}' (pipelines usage)
    global_prefix: Optional[str] = None
    dump_prediction_metadata_json: bool = True
    dump_trajectories: bool = False
    align_trajectory_structures: bool = False
    prevalidate_inputs: bool = True
    low_memory_mode: bool = (
        False  # False for standard mode, True for memory efficient tokenization mode
    )

    # Other:
    num_nodes: int = 1
    devices_per_node: int = 1
    verbose: bool = False
    seed: Optional[int] = None

    # For use as mapping:
    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class RFD3Output:
    atom_array: AtomArray
    metadata: dict
    example_id: str
    denoised_trajectory_stack: Optional[AtomArrayStack] = None
    noisy_trajectory_stack: Optional[AtomArrayStack] = None

    def dump(
        self,
        out_dir,
        verbose=True,
    ):
        base_path = os.path.join(out_dir, self.example_id)
        base_path = Path(base_path).absolute()
        coupling_output = str(self.metadata.get("coupling", {}).get("output", ""))
        allow_ambiguous_bond_annotations = coupling_output.startswith(
            "merged_A_plus_all_partners"
        )
        to_cif_file(
            self.atom_array,
            base_path,
            file_type="cif.gz",
            include_entity_poly=False,
            extra_fields=SAVED_CONDITIONING_ANNOTATIONS,
            _allow_ambiguous_bond_annotations=allow_ambiguous_bond_annotations,
        )
        if self.metadata:
            with open(f"{base_path}.json", "w") as f:
                json.dump(self.metadata, f, indent=4)

        # Trajectory saving
        denoised_base_path, noisy_base_path = _trajectory_output_paths(base_path)
        if self.denoised_trajectory_stack is not None:
            to_cif_file(
                self.denoised_trajectory_stack,
                denoised_base_path,
                file_type="cif.gz",
                include_entity_poly=False,
                _allow_ambiguous_bond_annotations=allow_ambiguous_bond_annotations,
            )

        if self.noisy_trajectory_stack is not None:
            to_cif_file(
                self.noisy_trajectory_stack,
                noisy_base_path,
                file_type="cif.gz",
                include_entity_poly=False,
                _allow_ambiguous_bond_annotations=allow_ambiguous_bond_annotations,
            )

        if verbose:
            ranked_logger.info(f"Outputs for {self.example_id} written to {base_path}.")


class RFD3InferenceEngine(BaseInferenceEngine):
    """Inference engine for RFdiffusion3"""

    def __init__(
        self,
        *,
        # Default input handling args
        skip_existing: bool,
        json_keys_subset: None | List[str],
        prevalidate_inputs: bool,
        # Base inference engine args
        diffusion_batch_size: int,
        inference_sampler: dict,
        specification: dict | None,
        # Structure dumping arguments
        global_prefix: str | None,
        cleanup_guideposts: bool,
        cleanup_virtual_atoms: bool,
        read_sequence_from_sequence_head: bool,
        output_full_json: bool,
        dump_prediction_metadata_json: bool,
        dump_trajectories: bool,
        align_trajectory_structures: bool,
        low_memory_mode: bool,
        coupling_mode: str | None = None,
        shared_chain_id: str = "A",
        complex_1_partners: list[str] | None = None,
        complex_2_partners: list[str] | None = None,
        track_1_specification: dict | None = None,
        track_2_specification: dict | None = None,
        merged_output_policy: str = "track1",
        **kwargs,
    ):
        super().__init__(
            transform_overrides={"diffusion_batch_size": diffusion_batch_size},
            inference_sampler_overrides={**inference_sampler},
            trainer_overrides={
                "cleanup_guideposts": cleanup_guideposts,
                "cleanup_virtual_atoms": cleanup_virtual_atoms,
                "read_sequence_from_sequence_head": read_sequence_from_sequence_head,
                "output_full_json": output_full_json,
            },
            **kwargs,
        )
        # save
        self.specification_overrides = dict(specification or {})
        self.inference_sampler_overrides = dict(inference_sampler or {})

        # Coupled prototype configuration.  These values are engine-level
        # because the ordinary RFD3 input schema is intentionally strict.
        self.coupling_mode = coupling_mode
        self.shared_chain_id = shared_chain_id
        self.complex_1_partners = list(complex_1_partners or ["B"])
        self.complex_2_partners = list(complex_2_partners or ["C"])
        self.track_1_specification = dict(track_1_specification or {})
        self.track_2_specification = dict(track_2_specification or {})
        self.merged_output_policy = str(merged_output_policy)
        if self.merged_output_policy not in {"track1", "track2", "both", "none"}:
            raise ValueError(
                "merged_output_policy must be one of: track1, track2, both, none."
            )

        # Setup output directories and args
        self.global_prefix = global_prefix
        self.json_keys_subset = json_keys_subset
        self.prevalidate_inputs = prevalidate_inputs
        self.skip_existing = skip_existing

        # Saving / other args
        self.dump_prediction_metadata_json = dump_prediction_metadata_json
        self.dump_trajectories = dump_trajectories
        self.align_trajectory_structures = align_trajectory_structures
        if not cleanup_guideposts:
            ranked_logger.warning(
                "Guideposts will not be cleaned up. This is intended for debugging purposes."
            )
        if not cleanup_virtual_atoms:
            ranked_logger.warning(
                "Virtual atoms will not be cleaned up. Some tools like MPNN may run, but outputs will not be like native structures."
            )

        # Check which example ids already exist in the output directory
        if low_memory_mode:
            ranked_logger.info("Low memory mode enabled.")
            # HACK: Set attribute to the diffusion module
            os.environ["RFD3_LOW_MEMORY_MODE"] = "1"

    def _override_checkpoint_config(self, cfg):
        """Load the RFD3 checkpoint config through the rfd3_system package.

        The public RFD3 checkpoint stores Hydra targets under the original
        `rfd3.*` package. This research copy must instantiate local
        `rfd3_system.*` classes so the coupled sampler registry and engine
        extensions are available while reusing the same checkpoint weights.
        """

        cfg = super()._override_checkpoint_config(cfg)
        self._rewrite_rfd3_targets_to_system(cfg)
        return cfg

    def _rewrite_rfd3_targets_to_system(self, node) -> None:
        if isinstance(node, DictConfig):
            if "_target_" in node and isinstance(node["_target_"], str):
                target = node["_target_"]
                if target.startswith("rfd3."):
                    node["_target_"] = "rfd3_system." + target[len("rfd3.") :]
            for value in node.values():
                self._rewrite_rfd3_targets_to_system(value)
        elif isinstance(node, ListConfig):
            for value in node:
                self._rewrite_rfd3_targets_to_system(value)

    def run(
        self,
        *,
        inputs: str | PathLike | AtomArray | DesignInputSpecification,
        n_batches: int | None = None,
        out_dir: str | PathLike | None = None,
    ):
        self._set_out_dir(out_dir)
        inputs = self._canonicalize_inputs(inputs)
        design_specifications = self._multiply_specifications(
            inputs=inputs,
            n_batches=n_batches,
        )
        if len(design_specifications) == 0:
            ranked_logger.info("No design specifications to run. Skipping.")
            return None
        if self.coupling_mode == "superdiff_shared_chain":
            if self.inference_sampler_overrides.get("kind") != "superdiff_shared_chain":
                raise ValueError(
                    "coupling_mode='superdiff_shared_chain' requires "
                    "inference_sampler.kind='superdiff_shared_chain'."
                )
            # init before
            self.initialize()
            return self._run_superdiff_shared_chain(design_specifications)
        if self.coupling_mode not in (None, "none"):
            raise ValueError(f"Unsupported coupling_mode: {self.coupling_mode!r}")
        ensure_inference_sampler_matches_design_spec(
            design_specifications, self.inference_sampler_overrides
        )
        # init before
        self.initialize()
        outputs = self._run_multi(design_specifications)
        return outputs

    def _set_out_dir(self, out_dir: str | PathLike | None):
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            ranked_logger.info(f"Outputs will be written to {out_dir.resolve()}.")
        self.out_dir = out_dir

    def _run_multi(self, specs) -> None | Dict[str, List[RFD3Output]]:
        # ==============================================================================
        # Prepare pipeline and inference loader
        # ==============================================================================
        loader = assemble_distributed_inference_loader_from_json(
            # Passed directly to ContigJSONDataset
            data=specs,
            transform=self.pipeline,
            name="inference-dataset",
            cif_parser_args=None,
            subset_to_keys=None,
            eval_every_n=1,
            # Sampler args
            world_size=self.trainer.fabric.world_size,
            rank=self.trainer.fabric.global_rank,
        )
        loader = self.trainer.fabric.setup_dataloaders(
            loader,
            use_distributed_sampler=False,
        )

        # ==============================================================================
        # Evaluate, using `validation_step`
        # ==============================================================================
        outputs = {}
        for batch_idx, batch in enumerate(loader):
            pipeline_output = batch[0]
            example_id = pipeline_output["example_id"]

            # Run model
            output_list = self._model_forward(pipeline_output)
            if self.out_dir:
                for output in output_list:
                    output.dump(out_dir=self.out_dir)
            else:
                outputs[example_id] = output_list
        return outputs

    def _model_forward(self, pipeline_output) -> List[RFD3Output]:
        # Wraps around the trainer validation step to create atom arrays for saving.
        t0 = time.time()
        with torch.no_grad():
            pipeline_output = self.trainer.fabric.to_device(pipeline_output)
            output_val = self.trainer.validation_step(
                batch=pipeline_output,
                batch_idx=0,
                compute_metrics=False,
            )
        t_end = time.time()

        # Add additional information to prediction metadata
        if self.dump_trajectories:
            X_noisy_L_traj = torch.stack(
                output_val["network_output"]["X_noisy_L_traj"]
            ).transpose(0, 1)  # [D, N_steps, L, 3]
            X_denoised_L_traj = torch.stack(
                output_val["network_output"]["X_denoised_L_traj"]
            ).transpose(0, 1)  # [D, N_steps, L, 3]

        outputs = []
        for idx in range(len(output_val["predicted_atom_array_stack"])):
            if self.dump_prediction_metadata_json:
                ckpt = Path(self.ckpt_path)
                if ckpt.is_symlink():
                    ckpt = ckpt.resolve(strict=True)  # follow symlink to target
                output_val["prediction_metadata"][idx]["ckpt_path"] = str(ckpt)
                output_val["prediction_metadata"][idx]["seed"] = self.seed

            # Append to outputs
            if self.dump_trajectories:
                X_denoised_L_traj_i = _reshape_trajectory(
                    X_noisy_L_traj[idx], self.align_trajectory_structures
                )
                X_noisy_L_traj_i = _reshape_trajectory(X_denoised_L_traj[idx], False)
                denoised_trajectory_stack = (
                    build_stack_from_atom_array_and_batched_coords(
                        X_denoised_L_traj_i, pipeline_output["atom_array"]
                    )
                )
                noisy_trajectory_stack = build_stack_from_atom_array_and_batched_coords(
                    X_noisy_L_traj_i, pipeline_output["atom_array"]
                )
            else:
                denoised_trajectory_stack = None
                noisy_trajectory_stack = None

            outputs.append(
                RFD3Output(
                    example_id=f"{pipeline_output['example_id']}_model_{idx}",
                    atom_array=output_val["predicted_atom_array_stack"][idx],
                    metadata=output_val["prediction_metadata"][idx]
                    if self.dump_prediction_metadata_json
                    else {},
                    denoised_trajectory_stack=denoised_trajectory_stack,
                    noisy_trajectory_stack=noisy_trajectory_stack,
                )
            )

        ranked_logger.info(f"Finished inference batch in {t_end - t0:.2f} seconds.")
        return outputs

    def _run_superdiff_shared_chain(
        self, specs
    ) -> None | Dict[str, List[RFD3Output]]:
        """Run the approximate A+B/A+C shared-chain prototype."""

        if self.trainer.fabric.world_size != 1:
            raise ValueError(
                "Approximate shared-chain coupling currently supports one process. "
                "Run with a single GPU/process for this prototype."
            )

        outputs = {}
        for example_id, example_spec in specs.items():
            output_list = self._model_forward_coupled(example_id, example_spec)
            if self.out_dir:
                for output in output_list:
                    output.dump(out_dir=self.out_dir)
            else:
                outputs[example_id] = output_list
        return outputs

    def _model_forward_coupled(
        self, example_id: str, example_spec: dict | DesignInputSpecification
    ) -> List[RFD3Output]:
        """Build two track views, run coupled inference, and format outputs."""

        t0 = time.time()
        if self._uses_track_specific_inputs():
            track_1_spec, track_2_spec = self._build_coupled_track_input_specs(
                example_spec
            )
        else:
            base_spec = self._ensure_design_spec(example_spec)
            track_1_spec, track_2_spec = self._build_coupled_track_specs(base_spec)

        track_1_output = self.pipeline(
            track_1_spec.to_pipeline_input(example_id=f"{example_id}_track1")
        )
        track_2_output = self.pipeline(
            track_2_spec.to_pipeline_input(example_id=f"{example_id}_track2")
        )

        shared_atom_map = build_shared_update_atom_map(
            track_1_output["atom_array"],
            track_2_output["atom_array"],
            self.shared_chain_id,
            self._feature_mask_np(track_1_output, "is_motif_atom_with_fixed_coord"),
            self._feature_mask_np(track_2_output, "is_motif_atom_with_fixed_coord"),
            self._feature_mask_np(track_1_output, "is_motif_atom_with_fixed_seq"),
            self._feature_mask_np(track_2_output, "is_motif_atom_with_fixed_seq"),
        )
        kappa_atom_subset = normalize_kappa_atom_subset(
            self.inference_sampler_overrides.get("kappa_atom_subset", "ALL")
        )
        kappa_indices_1_np, kappa_indices_2_np = select_kappa_solve_atom_indices(
            shared_atom_map,
            track_1_output["atom_array"],
            track_2_output["atom_array"],
            kappa_atom_subset,
        )
        self._validate_shared_initial_coordinates(
            track_1_output,
            track_2_output,
            shared_atom_map.update_indices_1,
            shared_atom_map.update_indices_2,
        )

        with torch.no_grad():
            track_1_device = self.trainer.fabric.to_device(track_1_output)
            track_2_device = self.trainer.fabric.to_device(track_2_output)
            device = track_1_device["coord_atom_lvl_to_be_noised"].device
            shared_indices_1 = torch.as_tensor(
                shared_atom_map.update_indices_1,
                dtype=torch.long,
                device=device,
            )
            shared_indices_2 = torch.as_tensor(
                shared_atom_map.update_indices_2,
                dtype=torch.long,
                device=device,
            )
            kappa_indices_1 = torch.as_tensor(
                kappa_indices_1_np,
                dtype=torch.long,
                device=device,
            )
            kappa_indices_2 = torch.as_tensor(
                kappa_indices_2_np,
                dtype=torch.long,
                device=device,
            )

            model = self._get_forward_coupled_model(self.trainer.state["model"])
            network_output = model.forward_coupled(
                track_1_input={"f": track_1_device["feats"]},
                track_2_input={"f": track_2_device["feats"]},
                track_1_coord_atom_lvl_to_be_noised=track_1_device[
                    "coord_atom_lvl_to_be_noised"
                ],
                track_2_coord_atom_lvl_to_be_noised=track_2_device[
                    "coord_atom_lvl_to_be_noised"
                ],
                shared_update_atom_indices_1=shared_indices_1,
                shared_update_atom_indices_2=shared_indices_2,
                shared_kappa_atom_indices_1=kappa_indices_1,
                shared_kappa_atom_indices_2=kappa_indices_2,
                coupling_metadata=self._base_coupling_metadata(
                    shared_atom_map,
                    kappa_atom_subset,
                    int(kappa_indices_1_np.size),
                ),
            )

        track_1_arrays, track_1_metadata = self.trainer._build_predicted_atom_array_stack(
            network_output["track_1"], track_1_device
        )
        track_2_arrays, track_2_metadata = self.trainer._build_predicted_atom_array_stack(
            network_output["track_2"], track_2_device
        )

        outputs = self._build_coupled_rfd3_outputs(
            example_id=example_id,
            track_1_output=track_1_output,
            track_2_output=track_2_output,
            network_output=network_output,
            track_1_arrays=track_1_arrays,
            track_2_arrays=track_2_arrays,
            track_1_metadata=track_1_metadata,
            track_2_metadata=track_2_metadata,
        )
        ranked_logger.info(
            f"Finished coupled inference batch in {time.time() - t0:.2f} seconds."
        )
        return outputs

    @staticmethod
    def _get_forward_coupled_model(model):
        """Return the module that owns `forward_coupled`.

        Foundry checkpoints may wrap RFD3 in the shared EMA module. During
        inference the EMA wrapper's `forward()` dispatches to `shadow`, so the
        coupled path should call `shadow.forward_coupled()` when present.
        """

        if hasattr(model, "forward_coupled"):
            return model
        for attr in ("shadow", "model"):
            wrapped = getattr(model, attr, None)
            if wrapped is not None and hasattr(wrapped, "forward_coupled"):
                return wrapped
        raise AttributeError(
            "Loaded model does not expose forward_coupled directly or through "
            "an EMA shadow/model wrapper."
        )

    def _ensure_design_spec(self, spec: dict | DesignInputSpecification):
        if isinstance(spec, DesignInputSpecification):
            return spec
        return DesignInputSpecification.safe_init(**spec)

    def _uses_track_specific_inputs(self) -> bool:
        has_track_1_input = self._track_spec_has_input(self.track_1_specification)
        has_track_2_input = self._track_spec_has_input(self.track_2_specification)
        if has_track_1_input != has_track_2_input:
            raise ValueError(
                "Track-specific motif input mode requires both "
                "track_1_specification and track_2_specification to provide an "
                "input or atom_array_input."
            )
        return has_track_1_input and has_track_2_input

    @staticmethod
    def _track_spec_has_input(track_spec: dict) -> bool:
        return exists(track_spec.get("input")) or exists(track_spec.get("atom_array_input"))

    def _build_coupled_track_input_specs(
        self,
        example_spec: dict | DesignInputSpecification,
    ) -> tuple[DesignInputSpecification, DesignInputSpecification]:
        """Build complete track specs from user-provided A+B and A+C inputs."""

        base_dict = self._base_track_input_defaults(example_spec)
        track_1_dict = self._track_input_spec_dict(
            base_dict,
            track_name="track_1",
            track_chains=[self.shared_chain_id] + self.complex_1_partners,
            overrides=self.track_1_specification,
        )
        track_2_dict = self._track_input_spec_dict(
            base_dict,
            track_name="track_2",
            track_chains=[self.shared_chain_id] + self.complex_2_partners,
            overrides=self.track_2_specification,
        )

        track_1_spec = DesignInputSpecification.safe_init(**track_1_dict)
        track_2_spec = DesignInputSpecification.safe_init(**track_2_dict)
        if not self._has_origin_controls(track_1_dict) and not self._has_origin_controls(
            track_2_dict
        ):
            common_origin = self._common_origin_from_many(
                [track_1_spec.atom_array_input, track_2_spec.atom_array_input]
            )
            track_1_dict["ori_token"] = common_origin
            track_2_dict["ori_token"] = common_origin
            track_1_spec = DesignInputSpecification.safe_init(**track_1_dict)
            track_2_spec = DesignInputSpecification.safe_init(**track_2_dict)
        return track_1_spec, track_2_spec

    @staticmethod
    def _base_track_input_defaults(
        example_spec: dict | DesignInputSpecification,
    ) -> dict:
        if isinstance(example_spec, DesignInputSpecification):
            spec_dict = example_spec.get_dict_to_save()
            extra = dict(example_spec.extra or {})
        else:
            spec_dict = dict(example_spec or {})
            extra = dict(spec_dict.get("extra", {}) or {})
        spec_dict.pop("input", None)
        spec_dict.pop("atom_array_input", None)
        spec_dict["extra"] = extra
        return spec_dict

    @staticmethod
    def _track_input_spec_dict(
        base_dict: dict,
        *,
        track_name: str,
        track_chains: list[str],
        overrides: dict,
    ) -> dict:
        spec_dict = dict(base_dict)
        spec_dict.update(overrides)
        extra = dict(base_dict.get("extra", {}) or {})
        extra.update(dict(spec_dict.get("extra", {}) or {}))
        extra["coupled_track"] = track_name
        extra["coupled_track_chains"] = track_chains
        extra["track_specific_input_mode"] = True
        spec_dict["extra"] = extra
        return spec_dict

    @staticmethod
    def _has_origin_controls(spec_dict: dict) -> bool:
        return exists(spec_dict.get("ori_token")) or exists(
            spec_dict.get("infer_ori_strategy")
        )

    def _build_coupled_track_specs(
        self, base_spec: DesignInputSpecification
    ) -> tuple[DesignInputSpecification, DesignInputSpecification]:
        if getattr(base_spec, "symmetry", None) is not None:
            raise ValueError(
                "Approximate shared-chain coupling does not currently support symmetry."
            )
        source_atom_array = base_spec.atom_array_input
        if source_atom_array is None:
            # For de novo coupled tests, the base specification may define only
            # a multi-chain contig such as "90,/0,80,/0,100".  Build that once
            # to create the source ABC atom array used only for chain identity,
            # atom ordering, and track splitting.
            source_atom_array, _ = base_spec.build(return_metadata=True)

        track_1_chains = [self.shared_chain_id] + self.complex_1_partners
        track_2_chains = [self.shared_chain_id] + self.complex_2_partners
        track_1_array = subset_by_chains(source_atom_array, track_1_chains)
        track_2_array = subset_by_chains(source_atom_array, track_2_chains)
        assert_matching_shared_chain(track_1_array, track_2_array, self.shared_chain_id)

        common_origin = self._common_origin(source_atom_array)
        track_1_spec = self._make_track_spec(
            base_spec,
            atom_array_input=track_1_array,
            track_name="track_1",
            track_chains=track_1_chains,
            common_origin=common_origin,
            overrides=self.track_1_specification,
        )
        track_2_spec = self._make_track_spec(
            base_spec,
            atom_array_input=track_2_array,
            track_name="track_2",
            track_chains=track_2_chains,
            common_origin=common_origin,
            overrides=self.track_2_specification,
        )
        return track_1_spec, track_2_spec

    def _make_track_spec(
        self,
        base_spec: DesignInputSpecification,
        *,
        atom_array_input: AtomArray,
        track_name: str,
        track_chains: list[str],
        common_origin: list[float],
        overrides: dict,
    ) -> DesignInputSpecification:
        spec_dict = base_spec.get_dict_to_save()
        spec_dict.pop("input", None)
        if "ligand" not in overrides:
            # The chain split already controls which partner chains are present.
            # Keeping a copied ligand field can duplicate ligands during build().
            spec_dict.pop("ligand", None)
        if (
            "ori_token" not in spec_dict
            and "ori_token" not in overrides
            and "infer_ori_strategy" not in spec_dict
            and "infer_ori_strategy" not in overrides
        ):
            spec_dict["ori_token"] = common_origin
        spec_dict.update(overrides)
        spec_dict["atom_array_input"] = atom_array_input

        extra = dict(base_spec.extra or {})
        extra.update(spec_dict.get("extra", {}))
        extra["coupled_track"] = track_name
        extra["coupled_track_chains"] = track_chains
        spec_dict["extra"] = extra
        return DesignInputSpecification.safe_init(**spec_dict)

    @staticmethod
    def _common_origin(atom_array: AtomArray) -> list[float]:
        coord = atom_array.coord
        finite = np.isfinite(coord).all(axis=-1)
        if not np.any(finite):
            return [0.0, 0.0, 0.0]
        return np.mean(coord[finite], axis=0).astype(float).tolist()

    @staticmethod
    def _common_origin_from_many(atom_arrays: list[AtomArray]) -> list[float]:
        coords = []
        for atom_array in atom_arrays:
            coord = atom_array.coord
            finite = np.isfinite(coord).all(axis=-1)
            if np.any(finite):
                coords.append(coord[finite])
        if not coords:
            return [0.0, 0.0, 0.0]
        return np.mean(np.concatenate(coords, axis=0), axis=0).astype(float).tolist()

    def _validate_shared_initial_coordinates(
        self,
        track_1_output: dict,
        track_2_output: dict,
        shared_indices_1: np.ndarray,
        shared_indices_2: np.ndarray,
    ) -> None:
        coord_1 = track_1_output["coord_atom_lvl_to_be_noised"][:, shared_indices_1, :]
        coord_2 = track_2_output["coord_atom_lvl_to_be_noised"][:, shared_indices_2, :]
        if coord_1.shape != coord_2.shape or not torch.allclose(
            coord_1, coord_2, atol=1e-4, rtol=1e-4
        ):
            raise ValueError(
                "Shared non-fixed chain coordinates differ after track pipeline "
                "construction. Provide track-specific inputs in the same frame or "
                "common ori_token/infer_ori_strategy settings."
            )

    @staticmethod
    def _feature_mask_np(pipeline_output: dict, feature_name: str) -> np.ndarray:
        value = pipeline_output["feats"][feature_name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        mask = np.asarray(value, dtype=bool)
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 1:
            raise ValueError(
                f"Expected atom-level feature mask {feature_name!r} to be 1D, "
                f"but received shape {mask.shape}."
            )
        return mask

    def _base_coupling_metadata(
        self,
        shared_atom_map,
        kappa_atom_subset: str,
        kappa_solve_atom_count: int,
    ) -> dict:
        sequence_policy = (
            "Track-specific sequence logits are left uncoupled. Merged outputs "
            f"are controlled by merged_output_policy={self.merged_output_policy!r}."
        )
        return {
            "coupling_mode": self.coupling_mode,
            "shared_chain_id": self.shared_chain_id,
            "complex_1_partners": self.complex_1_partners,
            "complex_2_partners": self.complex_2_partners,
            "merged_output_policy": self.merged_output_policy,
            "implementation": "approximate denoiser-delta proxy",
            "superdiff_exact": False,
            "sequence_policy": sequence_policy,
            "kappa_atom_subset": kappa_atom_subset,
            "kappa_solve_atom_count": int(kappa_solve_atom_count),
            "kappa_solve_policy": (
                "kappa is solved on kappa_atom_subset atoms; the resulting "
                "scalar kappa mixes and updates all non-fixed shared-chain "
                "atoms in the shared update map."
            ),
            "shared_atom_mapping": shared_atom_map.to_metadata(),
        }

    def _build_coupled_rfd3_outputs(
        self,
        *,
        example_id: str,
        track_1_output: dict,
        track_2_output: dict,
        network_output: dict,
        track_1_arrays,
        track_2_arrays,
        track_1_metadata: dict,
        track_2_metadata: dict,
    ) -> List[RFD3Output]:
        outputs = []
        coupling_metadata = _to_jsonable(network_output["coupling_metadata"])
        track_2_template = relabel_nonshared_chains(
            track_2_output["atom_array"],
            self.shared_chain_id,
            self.complex_2_partners,
        )
        merged_sources = self._merged_shared_sources()
        merged_templates = {
            source: merge_tracks_with_shared_source(
                track_1_output["atom_array"],
                track_2_template,
                self.shared_chain_id,
                source,
            )
            for source in merged_sources
        }

        for idx in range(len(track_1_arrays)):
            track_2_array = relabel_nonshared_chains(
                track_2_arrays[idx],
                self.shared_chain_id,
                self.complex_2_partners,
            )
            denoised_1, noisy_1 = self._trajectory_stacks(
                network_output["track_1"], track_1_output["atom_array"], idx
            )
            denoised_2, noisy_2 = self._trajectory_stacks(
                network_output["track_2"], track_2_template, idx
            )

            metadata_1 = dict(track_1_metadata[idx])
            metadata_1["coupling"] = coupling_metadata | {"output": "track_1_A_plus_partners"}
            metadata_2 = dict(track_2_metadata[idx])
            metadata_2["coupling"] = coupling_metadata | {"output": "track_2_A_plus_partners"}

            outputs.extend(
                [
                    RFD3Output(
                        example_id=f"{example_id}_track1_model_{idx}",
                        atom_array=track_1_arrays[idx],
                        metadata=metadata_1,
                        denoised_trajectory_stack=denoised_1,
                        noisy_trajectory_stack=noisy_1,
                    ),
                    RFD3Output(
                        example_id=f"{example_id}_track2_model_{idx}",
                        atom_array=track_2_array,
                        metadata=metadata_2,
                        denoised_trajectory_stack=denoised_2,
                        noisy_trajectory_stack=noisy_2,
                    ),
                ]
            )

            for source in merged_sources:
                merged_array = merge_tracks_with_shared_source(
                    track_1_arrays[idx],
                    track_2_array,
                    self.shared_chain_id,
                    source,
                )
                denoised_merged = None
                noisy_merged = None
                if self.dump_trajectories:
                    merged_network_output = self._merged_network_output(
                        network_output,
                        track_1_output["atom_array"],
                        track_2_template,
                        source,
                    )
                    denoised_merged, noisy_merged = self._trajectory_stacks(
                        merged_network_output, merged_templates[source], idx
                    )

                source_label = source.replace("_", "")
                output_kind = f"merged_A_plus_all_partners_{source_label}"
                if self.merged_output_policy == "track1" and source == "track_1":
                    output_id = f"{example_id}_merged_model_{idx}"
                else:
                    output_id = f"{example_id}_merged_{source_label}_model_{idx}"

                metadata_merged = {
                    "coupling": coupling_metadata
                    | {
                        "output": output_kind,
                        "shared_chain_source": source,
                    },
                    "source_outputs": {
                        "track_1": f"{example_id}_track1_model_{idx}",
                        "track_2": f"{example_id}_track2_model_{idx}",
                    },
                }
                outputs.append(
                    RFD3Output(
                        example_id=output_id,
                        atom_array=merged_array,
                        metadata=metadata_merged,
                        denoised_trajectory_stack=denoised_merged,
                        noisy_trajectory_stack=noisy_merged,
                    )
                )
        return outputs

    def _merged_shared_sources(self) -> list[str]:
        """Return the shared-chain sources requested by merged_output_policy."""

        if self.merged_output_policy == "none":
            return []
        if self.merged_output_policy == "track1":
            return ["track_1"]
        if self.merged_output_policy == "track2":
            return ["track_2"]
        return ["track_1", "track_2"]

    def _merged_network_output(
        self,
        network_output: dict,
        track_1_atom_array: AtomArray,
        track_2_atom_array: AtomArray,
        shared_source: str,
    ) -> dict:
        return {
            "X_noisy_L_traj": [
                self._merge_track_coords(
                    x1, x2, track_1_atom_array, track_2_atom_array, shared_source
                )
                for x1, x2 in zip(
                    network_output["track_1"]["X_noisy_L_traj"],
                    network_output["track_2"]["X_noisy_L_traj"],
                )
            ],
            "X_denoised_L_traj": [
                self._merge_track_coords(
                    x1, x2, track_1_atom_array, track_2_atom_array, shared_source
                )
                for x1, x2 in zip(
                    network_output["track_1"]["X_denoised_L_traj"],
                    network_output["track_2"]["X_denoised_L_traj"],
                )
            ],
        }

    def _merge_track_coords(
        self,
        coords_1: torch.Tensor,
        coords_2: torch.Tensor,
        track_1_atom_array: AtomArray,
        track_2_atom_array: AtomArray,
        shared_source: str,
    ):
        # Match merge_tracks_with_shared_source(): shared chain from the chosen
        # track, then track 1 partners, then track 2 partners.
        if shared_source not in {"track_1", "track_2"}:
            raise ValueError(f"Unsupported shared_source: {shared_source!r}")
        shared_1 = torch.as_tensor(
            chain_mask(track_1_atom_array, self.shared_chain_id),
            dtype=torch.bool,
            device=coords_1.device,
        )
        shared_2 = torch.as_tensor(
            chain_mask(track_2_atom_array, self.shared_chain_id),
            dtype=torch.bool,
            device=coords_2.device,
        )
        shared_coords = (
            coords_1[:, shared_1, :]
            if shared_source == "track_1"
            else coords_2[:, shared_2, :]
        )
        nonshared_1 = ~shared_1
        nonshared_2 = ~shared_2
        return torch.cat(
            [
                shared_coords,
                coords_1[:, nonshared_1, :],
                coords_2[:, nonshared_2, :],
            ],
            dim=1,
        )

    def _trajectory_stacks(self, network_output: dict, atom_array: AtomArray, idx: int):
        if not self.dump_trajectories:
            return None, None
        X_noisy_L_traj = torch.stack(network_output["X_noisy_L_traj"]).transpose(0, 1)
        X_denoised_L_traj = torch.stack(network_output["X_denoised_L_traj"]).transpose(0, 1)
        denoised = build_stack_from_atom_array_and_batched_coords(
            _reshape_trajectory(
                X_denoised_L_traj[idx], self.align_trajectory_structures
            ),
            atom_array,
        )
        noisy = build_stack_from_atom_array_and_batched_coords(
            _reshape_trajectory(X_noisy_L_traj[idx], False),
            atom_array,
        )
        return denoised, noisy

    ###############################################
    # Input merging
    ###############################################

    def _canonicalize_inputs(
        self, inputs
    ) -> Dict[str, dict | DesignInputSpecification]:
        is_json_like = (isinstance(inputs, (str, PathLike, Path))) or (
            isinstance(inputs, list)
            and all([isinstance(i, (str, PathLike, Path)) for i in inputs])
        )
        is_specification_like = isinstance(inputs, DesignInputSpecification) or (
            isinstance(inputs, list)
            and all([isinstance(i, DesignInputSpecification) for i in inputs])
        )
        is_atom_array_like = isinstance(inputs, (AtomArray, list)) or (
            isinstance(inputs, list) and all([isinstance(i, AtomArray) for i in inputs])
        )
        if inputs is None:
            # Create empty specification dictionary
            prefix = str(self.global_prefix or "design").rstrip("_") or "design"
            return {prefix: {**self.specification_overrides}}
        elif is_json_like:
            # List of file paths
            inputs = process_input(
                inputs,
                json_keys_subset=self.json_keys_subset,
                global_prefix=self.global_prefix,
                specification_overrides=self.specification_overrides,
                validate=self.prevalidate_inputs,
            )  # any -> Dict[Name: DesignInputSpecification]
        elif is_specification_like:
            # List of DesignInputSpecifications
            if isinstance(inputs, DesignInputSpecification):
                inputs = [inputs]
            inputs = {f"backbone_{i}": spec for i, spec in enumerate(inputs)}
        elif is_atom_array_like:
            raise NotImplementedError("AtomArray inputs not yet supported.")
        else:
            raise ValueError(
                f"Invalid input type: {type(inputs)}. Expected JSON/YAML file paths, AtomArray, or DesignInputSpecification.\nInput: {inputs}"
            )

        return inputs

    def _multiply_specifications(
        self, inputs: Dict[str, dict | DesignInputSpecification], n_batches=None
    ) -> Dict[str, dict | DesignInputSpecification]:
        # Find existing example IDS in output directory
        if exists(self.out_dir):
            existing_example_ids_ = set(
                extract_example_id_from_path(path, CIF_LIKE_EXTENSIONS)
                for path in find_files_with_extension(self.out_dir, CIF_LIKE_EXTENSIONS)
            )
            existing_example_ids = set(
                [
                    "_model_".join(eid.split("_model_")[:-1])
                    for eid in existing_example_ids_
                ]
            )
            ranked_logger.info(
                f"Found {len(existing_example_ids)} existing example IDs in the output directory ({len(existing_example_ids_)} total)."
            )

        # Based on inputs, construct the specifications to loop through
        design_specifications = {}
        for prefix, example_spec in inputs.items():
            # Record task name in the specification
            if isinstance(example_spec, DesignInputSpecification):
                example_spec.extra = example_spec.extra or {}
                example_spec.extra["task_name"] = prefix
            else:
                if "extra" not in example_spec:
                    example_spec["extra"] = {}
                example_spec["extra"]["task_name"] = prefix

            # ... Create n_batches for example
            for batch_id in range((n_batches) if exists(n_batches) else 1):
                # ... Example ID
                example_id = f"{prefix}_{batch_id}" if exists(n_batches) else prefix
                if (
                    self.skip_existing
                    and exists(self.out_dir)
                    and example_id in existing_example_ids
                ):
                    ranked_logger.info(
                        f"Skipping design specification for example {example_id} | Already exists."
                    )
                    continue
                design_specifications[example_id] = example_spec
        return design_specifications


def normalize_inputs(inputs: str | list | None) -> list[str | None]:
    """
    inputs: str | list[str] | None
        - Can be:
            - A single path to a JSON, YAML, or regular input file (cif or pdb)
            - A comma-separated string of paths (e.g. "a.json,b.json")
            - A list of file paths
            - None or an empty list, in which case a dummy input is added (used for e.g. motif-only design)
        - Returns list of paths or [None] if no inputs are provided
    """
    if inputs is None or (isinstance(inputs, list) and len(inputs) == 0):
        inputs = [None]
    elif isinstance(inputs, str):
        inputs = inputs.split(",")
    elif not isinstance(inputs, list):
        raise ValueError(
            f"Invalid input type: {type(inputs)}. Expected str, list, or None.\nInput: {inputs}"
        )
    return inputs


def _to_jsonable(value):
    """Convert nested tensors/arrays to JSON-friendly Python objects."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _trajectory_output_paths(base_path: Path) -> tuple[str, str]:
    """Build trajectory paths without corrupting names like `merged_model_0`.

    `str.rstrip("_model_")` removes any trailing characters that appear in the
    argument, so `merged_model_0` can become `merg`. Split on the model marker
    instead to preserve the full output kind.
    """

    stem = str(base_path)
    marker = "_model_"
    if marker not in stem:
        return f"{stem}_denoised", f"{stem}_noisy"
    prefix, suffix = stem.rsplit(marker, 1)
    return (
        f"{prefix}_denoised_model_{suffix}",
        f"{prefix}_noisy_model_{suffix}",
    )


def process_input(
    inputs: str | list | None,
    json_keys_subset: str | list | None = None,
    global_prefix: str | None = None,
    specification_overrides: dict | None = None,
    validate: bool = True,
) -> Dict[str, dict]:
    """
    inputs: Any -> list[str | None] (see normalize_inputs)
    json_keys_subset: extract only subset of JSON keys. None will keep all keys
    prefix: If provided, prefix all example ids with said prefix

    returns: Dictionaries of specifcation args pre-batching:
        {
            'jsonfile_jsonkey1': {
                **args_from_key1
            },
            'jsonfile_jsonkey2': {
                **args_from_key2
            }
        }
    """
    specification_overrides = dict(specification_overrides or {})

    def merge_args(example_args: dict) -> dict:
        return merge_with(lambda x: x[-1], example_args, specification_overrides)

    inputs = normalize_inputs(inputs)

    # If global_prefix is not provided, then default to using the basename of the JSON or YAML file (when provided)
    if global_prefix is None:
        use_json_basename_prefix = True
    else:
        use_json_basename_prefix = False

    # ... Convert all inputs to list of inputs (e.g. if comma-separated)
    if exists(inputs) and "," in inputs:
        inputs = inputs.split(",")
    elif not exists(inputs):
        # If inputs is None or empty, we will create a dummy input
        inputs = []
    inputs = inputs if isinstance(inputs, list) else [inputs]
    if len(inputs) == 0:
        inputs = [None]

    # ... Determine prefix of sample to create
    all_specs = {}
    for input in inputs:
        if exists(input) and (input.endswith(".json") or input.endswith(".yaml")):
            # ... Load JSON or YAML file
            with open(input, "r") as f:
                data = json.load(f) if input.endswith(".json") else yaml.safe_load(f)

            # ... Apply any global args for this input file
            if "global_args" in data:
                global_args = data.pop("global_args")
                for example in data:
                    data[example].update(global_args)

            # ... Subset to keys
            if json_keys_subset is not None:
                json_keys_subset = (
                    json_keys_subset.split(",")
                    if isinstance(json_keys_subset, str)
                    else json_keys_subset
                )
                data = {
                    example: data[example]
                    for example in json_keys_subset
                    if example in data
                }

            # ... Extract each accumulated example in data.
            for example, args in data.items():
                args = ensure_input_is_abspath(args, input)
                if use_json_basename_prefix:
                    name = os.path.splitext(os.path.basename(input))[0]
                    prefix = f"{name}_{example}"
                else:
                    prefix = f"{global_prefix}{example}"
                args["extra"] = args.get("extra", {}) | {"example": example}
                all_specs[prefix] = dict(merge_args(args))

        elif exists(input):
            prefix = os.path.basename(os.path.splitext(input)[0])
            if global_prefix is not None:
                prefix = f"{global_prefix}{prefix}"
            all_specs[prefix] = dict(merge_args({"input": input}))
        else:
            all_specs["backbone"] = dict(specification_overrides)

    if validate:
        for prefix, example_spec in all_specs.items():
            ranked_logger.info(
                f"Prevalidating design specification for example: {prefix}"
            )
            DesignInputSpecification.safe_init(**example_spec)

    return all_specs


def _reshape_trajectory(traj, align_structures: bool):
    traj = [traj[i] for i in range(len(traj))]  # make list of arrays
    max_frames = 100
    if len(traj) > max_frames:
        selected_indices = torch.linspace(0, len(traj) - 1, max_frames).long().tolist()
        traj = [traj[i] for i in selected_indices]
    if align_structures:
        # ... align the trajectories on the last prediction
        for step in range(len(traj) - 1):
            traj[step] = weighted_rigid_align(
                X_L=traj[-1][None],
                X_gt_L=traj[step][None],
            ).squeeze(0)
    traj = traj[::-1]  # reverse to go from noised -> denoised

    traj = torch.stack(traj).cpu().numpy()
    return traj
