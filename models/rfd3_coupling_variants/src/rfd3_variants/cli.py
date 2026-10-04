"""Named model entry points share this Hydra-compatible command dispatcher."""
import sys
from pathlib import Path


def main(variation="rfd3_mean_5050"):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import rfd3_system
    from .engine import Engine
    config_dir = Path(rfd3_system.__file__).resolve().parents[2] / "configs"
    argv = [arg for arg in sys.argv[1:] if arg != "design"]
    defaults = ["inference_engine=rfdiffusion3", "coupling_mode=superdiff_shared_chain",
        "inference_sampler.kind=superdiff_shared_chain", "inference_sampler.proxy_norm_weight=0.5",
        "inference_sampler.proxy_kappa_min=0", "inference_sampler.proxy_kappa_max=1",
        "inference_sampler.step_scale=3", "inference_sampler.gamma_0=0.2",
        "inference_sampler.n_recycle=2", "diffusion_batch_size=1", "merged_output_policy=none"]
    specified = {s.split("=",1)[0].lstrip("+") for s in argv}
    defaults = [s for s in defaults if s.split("=",1)[0] not in specified]
    with initialize_config_dir(config_dir=str(config_dir),version_base="1.3"):
        cfg = compose(config_name="inference", overrides=defaults+argv)
    params = OmegaConf.to_container(cfg,resolve=True)
    run = {k:params.pop(k) for k in ("inputs","out_dir","n_batches")}
    params.pop("_target_",None)
    Engine(variation=variation,**params).run(**run)


if __name__ == "__main__":
    main()
