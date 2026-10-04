"""Real checkpoint parity and gradient contracts, run only inside GPU SLURM."""
import argparse
import json
from pathlib import Path
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from rfd3_system.engine import RFD3InferenceEngine
from rfd3_variants.sampler import ExperimentalSampler,canonical_indices,select_a
from rfd3_variants.policies import profile,js_energy
from rfd3_variants.model import DiffusionModule,DifferentiableDecoder


def validate(out,seed,full=False):
    import rfd3_system
    root=Path(rfd3_system.__file__).resolve().parents[2]/"configs"
    lengths=(90,80,100) if full else (24,20,28)
    a,b,c=lengths
    overrides=["inference_engine=rfdiffusion3","ckpt_path=/weights/rfd3_latest.ckpt",
       f"out_dir={out}","inputs=null",f"seed={seed}","diffusion_batch_size=1",
       "coupling_mode=superdiff_shared_chain","inference_sampler.kind=superdiff_shared_chain",
       "inference_sampler.proxy_norm_weight=0.5","inference_sampler.step_scale=3",
       "inference_sampler.gamma_0=0.2","inference_sampler.n_recycle=2",
       "inference_sampler.num_timesteps="+("200" if full else "12"),"merged_output_policy=none",
       f"+specification.contig='{a},/0,{b},/0,{c}'",f"+specification.length={a+b+c}",
       f"+track_1_specification.contig='A1-{a},/0,B1-{b}'",f"+track_1_specification.length={a+b}",
       f"+track_2_specification.contig='A1-{a},/0,C1-{c}'",f"+track_2_specification.length={a+c}"]
    for i in (1,2):
        overrides += [f"+track_{i}_specification.select_fixed_atoms=false",f"+track_{i}_specification.select_unfixed_sequence=true",f"+track_{i}_specification.is_non_loopy=true"]
    with initialize_config_dir(config_dir=str(root),version_base="1.3"):
        cfg=OmegaConf.to_container(compose(config_name="inference",overrides=overrides),resolve=True)
    run={k:cfg.pop(k) for k in ("inputs","out_dir","n_batches")}
    cfg.pop("_target_",None)
    engine=RFD3InferenceEngine(**cfg)
    engine.initialize()
    model=engine._get_forward_coupled_model(engine.trainer.state["model"])
    model.requires_grad_(False)
    original=model.inference_sampler.sampler
    native=original.sample_coupled_superdiff_proxy
    checks={}
    def compare(one,two):
        differences=[float((one[key]["X_L"]-two[key]["X_L"]).abs().max()) for key in ("track_1","track_2")]
        print("PARITY maximum absolute coordinate differences",differences,flush=True)
        return all(v==0 for v in differences)
    def wrapped(**kw):
        cpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state()
        reference=native(**kw)
        ending=torch.cuda.get_rng_state()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
        checks["native_repeat_exact"]=compare(reference,native(**kw))
        experimental=ExperimentalSampler.__new__(ExperimentalSampler)
        experimental.__dict__.update({k:v for k,v in original.__dict__.items() if not callable(v)})
        def sample(method,**config):
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
            experimental.experiment=profile(method)|config
            return experimental.sample_coupled_superdiff_proxy(**kw)
        mean=sample("mean_5050")
        checks["exact_native_state_parity"]=compare(reference,mean)
        checks["exact_native_rng_parity"]=torch.equal(ending,torch.cuda.get_rng_state())
        out.mkdir(parents=True,exist_ok=True)
        (out/"parity.json").write_text(json.dumps(checks,indent=2))
        if full:
            assert all(checks.values()),checks
            return reference
        checks["alpha_zero"]=compare(mean,sample("residual_consensus",alpha=0.))
        uncoupled=sample("uncoupled")
        checks["alpha_one"]=compare(uncoupled,sample("residual_consensus",alpha=1.))
        checks["release_zero"]=compare(uncoupled,sample("late_uncoupling",release_fraction=0.))
        checks["release_one"]=compare(mean,sample("late_uncoupling",release_fraction=1.))
        checks["noise_one"]=compare(uncoupled,sample("correlated_noise"))
        for method in ("soft_guidance","coarse_coupling","sequence_coupling"):
            checks["zero_"+method]=compare(uncoupled,sample(method))
        module=kw["diffusion_module"]
        module.__class__=DiffusionModule
        module.decoder.__class__=DifferentiableDecoder
        # Small real differentiable forward at an intermediate, native noise level.
        t=reference["track_1"]["t_hats"][6]
        x=reference["track_1"]["X_noisy_L_traj"][6]*torch.sqrt(t*t+256)/16
        track=kw["track_1"]
        args=dict(t_hat=t,D=1,f=track["f"],diffusion_module=module,initializer_outputs=track["initializer_outputs"],step_num=6)
        plain=original._denoise_once(X_noisy_L=x,**args)
        aa=canonical_indices(x.device)
        _,_,tokens=select_a(track["f"],kw["shared_update_atom_indices_1"])
        with torch.enable_grad():
            leaf=x.detach().requires_grad_(True)
            output=original._denoise_once(X_noisy_L=leaf,**args)
            torch.testing.assert_close(output["X_L"],plain["X_L"],rtol=.01,atol=.01)
            logits=output["sequence_logits_I"][:,tokens][:,:,aa]
            objective=js_energy(logits,torch.flip(plain["sequence_logits_I"][:,tokens][:,:,aa],[-1]))
            grad,=torch.autograd.grad(objective,leaf)
        checks["sequence_gradient_finite_nonzero"]=bool(torch.isfinite(grad).all() and grad.norm()>0)
        checks["gradient_forward_equivalence"]=True
        guided=sample("sequence_coupling",strength=.1)
        checks["sequence_guided_rollout"]=bool(torch.isfinite(guided["track_1"]["X_L"]).all())
        soft=sample("soft_guidance",strength=1.)
        checks["clean_guided_rollout"]=bool(torch.isfinite(soft["track_1"]["X_L"]).all())
        assert all(checks.values()),checks
        return reference
    original.sample_coupled_superdiff_proxy=wrapped
    engine.run(**run)
    return checks


if __name__=="__main__":
    # Deterministic reductions distinguish implementation parity from CUDA
    # atomic accumulation order. The environment sets deterministic cuBLAS too.
    torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser();p.add_argument("out");a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    checks={}
    for seed,full in [(90021,False),(90022,True)]:
        checks[str(seed)]=validate(out/f"seed_{seed}",seed,full)
        (out/"checks.json").write_text(json.dumps(checks,indent=2))
    (out/"_COMPLETE.json").write_text(json.dumps(checks,indent=2))
