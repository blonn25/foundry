"""Reuse the original engine's entire preprocessing/output contract."""
from omegaconf import DictConfig, ListConfig, open_dict
from rfd3_system.engine import RFD3InferenceEngine
from .policies import profile, validate


class Engine(RFD3InferenceEngine):
    def __init__(self, *, variation="rfd3_mean_5050", experiment=None, **kwargs):
        self.experiment = profile(variation) | dict(experiment or {})
        validate(self.experiment)
        self.experiment["debug"] = bool(kwargs.get("dump_trajectories", False))
        super().__init__(**kwargs)

    def _override_checkpoint_config(self, cfg):
        cfg = super()._override_checkpoint_config(cfg)
        def rewrite(node):
            if isinstance(node, DictConfig):
                target = node.get("_target_", "")
                if target == "rfd3_system.model.RFD3.RFD3":
                    with open_dict(node):
                        node["_target_"] = "rfd3_variants.model.Network"
                        node["experiment"] = self.experiment
                elif target == "rfd3_system.model.RFD3_diffusion_module.RFD3DiffusionModule":
                    node["_target_"] = "rfd3_variants.model.DiffusionModule"
                for value in node.values():
                    rewrite(value)
            elif isinstance(node, ListConfig):
                for value in node:
                    rewrite(value)
        rewrite(cfg)
        return cfg
