"""Local adapters preserve weights/forward values while enabling sequence VJPs."""
import torch
from rfd3_system.model.RFD3 import RFD3
from rfd3_system.model.RFD3_diffusion_module import RFD3DiffusionModule
from rfd3_system.model.layers.blocks import CompactStreamingDecoder


class DifferentiableDecoder(CompactStreamingDecoder):
    def forward(self, A_I, S_I, Z_II, Q_L, C_L, P_LL, tok_idx, indices,
                f=None, chunked_pairwise_embedder=None, initializer_outputs=None):
        if not torch.is_grad_enabled():
            return super().forward(A_I, S_I, Z_II, Q_L, C_L, P_LL, tok_idx,
                                   indices, f, chunked_pairwise_embedder, initializer_outputs)
        for i in range(self.n_blocks):
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)
            Q_L = self.atom_transformer[i](Q_L, C_L, P_LL, indices=indices, f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs)
        # This is the sole decoder difference: retain the continuous graph.
        return self.downcast(Q_L, A_I, S_I, tok_idx=tok_idx), Q_L, {}


class DiffusionModule(RFD3DiffusionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Change only this newly constructed instance's dispatch. No weights,
        # initialization draws, state_dict keys, or upstream classes are changed.
        self.decoder.__class__ = DifferentiableDecoder

    def forward_with_recycle(self, n_recycle, **kwargs):
        if not torch.is_grad_enabled():
            return super().forward_with_recycle(n_recycle, **kwargs)
        n = self.n_recycle if n_recycle is None else n_recycle
        if n < 1:
            raise ValueError("at least one denoiser pass is required")
        recycled = {}
        for _ in range(n):
            torch.clear_autocast_cache()
            recycled = self.process_(D_II_self=recycled.get("D_II_self"),
                X_L_self=recycled.get("X_L"), **kwargs)
        return recycled


class Network(RFD3):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        from .sampler import ExperimentalSampler
        from omegaconf import OmegaConf
        old = self.inference_sampler.sampler
        sampler = ExperimentalSampler.__new__(ExperimentalSampler)
        sampler.__dict__.update(old.__dict__)
        sampler.experiment = OmegaConf.to_container(experiment, resolve=True) if OmegaConf.is_config(experiment) else experiment
        self.inference_sampler.sampler = sampler
        # Only input-coordinate gradients are needed; never accumulate weights.
        self.requires_grad_(False)
