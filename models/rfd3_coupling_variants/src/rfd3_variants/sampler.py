"""One EDM loop for all policies, retaining the original draw/update order.

No original model code is patched. Per-step corrections act on predictions or
directions; initialization/churn are explicit and independent of policy RNG.
"""
from __future__ import annotations
import hashlib
import time
import torch
from rfd3_system.model.inference_sampler import SampleDiffusionWithSuperDiffSharedChainProxy
from .policies import (schedule, released, residual, correlated, rmsd,
                       distance_energy, js_energy, cap_correction, validate)


def digest(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def canonical_indices(device):
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    # Match Caliby's allowed alphabet. Never assume DISCO's vocabulary order.
    names = "ALA ARG ASN ASP GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
    return torch.as_tensor(AF3SequenceEncoding().encode(names), device=device).long()


def select_a(f, indices):
    ca = indices[f["is_ca"][indices].bool()]
    token = f["atom_to_token_map"][ca].long()
    fixed_seq = f.get("is_motif_atom_with_fixed_seq", torch.zeros_like(f["is_ca"]))
    return ca, token, token[~fixed_seq[ca].bool()]


def structural_gradient(clean, feats, ca, block):
    """Gradient in residue-translation coordinates, lifted to movable A atoms."""
    with torch.enable_grad():
        leaves = [x[:, c].detach().float().requires_grad_(True) for x, c in zip(clean, ca)]
        # Token gaps mark motif exclusions/chain breaks. Do not pool across them.
        tokens = [f["atom_to_token_map"][c].long() for f, c in zip(feats, ca)]
        breaks = [0] + [i for i in range(1, len(tokens[0])) if
                       any(int(t[i]-t[i-1]) != 1 for t in tokens)] + [len(tokens[0])]
        pooled = []
        for leaf in leaves:
            pooled.append(torch.stack([leaf[:, j:min(j+block,end)].mean(1)
                for start,end in zip(breaks, breaks[1:]) for j in range(start,end,block)], 1))
        energy = distance_energy(*pooled)
        grads = torch.autograd.grad(energy, leaves)
    lifted = []
    for x, f, c, g in zip(clean, feats, ca, grads):
        # Gather residue translations once. Besides avoiding many small GPU
        # launches, this avoids a deterministic-CUDA boolean-index broadcast bug.
        ca_tokens,order=torch.sort(f["atom_to_token_map"][c].long())
        atom_tokens=f["atom_to_token_map"].long()
        positions=torch.searchsorted(ca_tokens,atom_tokens).clamp(max=len(ca_tokens)-1)
        mask=(ca_tokens[positions]==atom_tokens) & ~f["is_motif_atom_with_fixed_coord"].bool()
        lifted.append(g[:,order[positions]].to(x.dtype)*mask[None,:,None])
    return lifted, float(energy.detach())


class ExperimentalSampler(SampleDiffusionWithSuperDiffSharedChainProxy):
    def sample_coupled_superdiff_proxy(self, *, track_1, track_2,
            shared_update_atom_indices_1, shared_update_atom_indices_2,
            diffusion_module, diffusion_batch_size, coupling_metadata,
            shared_kappa_atom_indices_1=None, shared_kappa_atom_indices_2=None):
        cfg = dict(self.experiment)
        validate(cfg)
        self._validate_proxy_sampler_settings()
        if diffusion_batch_size != 1:
            raise ValueError("experiments use one independent paired seed per sample")
        if self.proxy_norm_weight != 0.5:
            raise ValueError("experimental foundation requires proxy_norm_weight=0.5")
        tracks = [track_1, track_2]
        feats = [t["f"] for t in tracks]
        device = tracks[0]["coord_atom_lvl_to_be_noised"].device
        idx = [v.to(device=device, dtype=torch.long) for v in
               (shared_update_atom_indices_1, shared_update_atom_indices_2)]
        if idx[0].numel() == 0 or idx[0].shape != idx[1].shape:
            raise ValueError("nonempty matching A atom maps required")
        fixed = [f["is_motif_atom_with_fixed_coord"].bool() for f in feats]
        for i in range(2):
            if fixed[i][idx[i]].any():
                raise ValueError("fixed atoms must not be coupled")
        selections = [select_a(f, ix) for f, ix in zip(feats, idx)]
        ca = [s[0] for s in selections]
        seq = [s[2] for s in selections]
        if ca[0].shape != ca[1].shape or seq[0].shape != seq[1].shape:
            raise ValueError("mapped A C-alpha/sequence selections disagree")
        if len(ca[0]) < 2:
            raise ValueError("at least two movable A residues required")
        sigmas = [self._construct_inference_noise_schedule(device, f.get("partial_t")) for f in feats]
        if not torch.equal(*sigmas):
            raise ValueError("identical noise schedules required")
        sigmas = sigmas[0]
        n = len(sigmas)-1
        if n < 1:
            raise ValueError("at least one denoising update required")
        churn_steps = sum(float(s) > self.gamma_min and self.gamma_0 > 0 for s in sigmas[1:])
        x = [self._get_initial_structure(c0=sigmas[0], D=1,
              L=f["ref_element"].shape[0], coord_atom_lvl_to_be_noised=t["coord_atom_lvl_to_be_noised"].clone(),
              is_motif_atom_with_fixed_coord=fix) for t,f,fix in zip(tracks,feats,fixed)]
        if cfg["shared_initialization"]:
            x[1][:,idx[1]] = x[0][:,idx[0]]
        method = cfg["method"]
        hard_method = method in {"mean_5050", "population_diversity", "late_uncoupling"}
        if hard_method and (not cfg["shared_initialization"] or any(schedule(cfg["noise_correlation"],p)!=1 for p in [0.,1.])):
            raise ValueError("hard-consensus profiles require shared initialization/churn")
        trajectory = [dict(X_noisy_L_traj=[], X_denoised_L_traj=[], t_hats=[], sequence_entropy_traj=[]) for _ in range(2)]
        trace = []
        initial_hashes = [digest(v[:,ix]) for v,ix in zip(x,idx)]
        def state(k, sigma):
            return dict(completed_steps=k, completed_fraction=k/n, sigma=float(sigma),
                ca_rmsd=float(rmsd(x[0][:,ca[0]],x[1][:,ca[1]])),
                ca_frame_rmsd=float(rmsd(x[0][:,ca[0]],x[1][:,ca[1]],False)))
        trace.append(state(0,sigmas[0]))
        aa = canonical_indices(device)
        started = time.monotonic()
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        def predict(i, coords, sigma, step):
            return self._denoise_once(X_noisy_L=coords, t_hat=sigma, D=1,
                f=feats[i], diffusion_module=diffusion_module,
                initializer_outputs=tracks[i]["initializer_outputs"], step_num=step)
        for step, (previous, following) in enumerate(zip(sigmas, sigmas[1:])):
            p = step / max(n-1,1)
            noise_p = min(1., step/max(churn_steps-1,1))
            rho = schedule(cfg["noise_correlation"], noise_p)
            alpha = schedule(cfg["alpha"],p)
            active = hard_method and (method != "late_uncoupling" or not released(step,n,cfg["release_fraction"]))
            # This explicit alpha-zero path preserves original floating-point order.
            active = active or (method == "residual_consensus" and alpha == 0 and rho == 1)
            gamma = self.gamma_0 if following > self.gamma_min else 0
            sigma = previous*(1+gamma)
            h = self.step_scale*(following-sigma)
            eps_scale = self.noise_scale*torch.sqrt(sigma.square()-previous.square())
            # Always consume the same three draws, even after churn turns off.
            eps = [eps_scale*torch.normal(0.,1.,size=v.shape,device=device) for v in x]
            shared = eps_scale*torch.normal(0.,1.,size=x[0][:,idx[0]].shape,device=device)
            mapped = correlated(shared,eps[0][:,idx[0]],eps[1][:,idx[1]],rho)
            for i in range(2):
                eps[i][:,fixed[i]] = 0
                eps[i][:,idx[i]] = mapped[i]
            if active:
                x[1][:,idx[1]] = x[0][:,idx[0]]
            noisy = [v+e for v,e in zip(x,eps)]
            with torch.no_grad():
                outs = [predict(i,noisy[i],sigma,step) for i in range(2)]
            clean = [o["X_L"] for o in outs]
            native_delta = [(v-c)/sigma for v,c in zip(noisy,clean)]
            native_update = [h*d for d in native_delta]
            selected = [o["sequence_logits_I"][:,s][:,:,aa] for o,s in zip(outs,seq)]
            temperature = schedule(cfg["temperature"],p)
            energy_js = float(js_energy(*selected,temperature)) if seq[0].numel() else 0.
            entropy = [float(-(z.float().softmax(-1)*z.float().log_softmax(-1)).sum(-1).mean()) if z.numel() else None for z in selected]
            gradients = None
            energy = None
            strength = cfg["strength"]*schedule(cfg["guidance_schedule"],p)
            needs_gradient = (strength > 0 or cfg["calibration"]) and method in {"soft_guidance","coarse_coupling","sequence_coupling"}
            if needs_gradient:
                if method == "sequence_coupling":
                    if not seq[0].numel():
                        raise ValueError("sequence guidance requires designable shared residues")
                    with torch.enable_grad():
                        logits = [z.detach().float().requires_grad_(True) for z in selected]
                        objective = js_energy(*logits,temperature)
                        cotangents = torch.autograd.grad(objective,logits)
                    gradients = []
                    for i in range(2):
                        with torch.enable_grad():
                            leaf = noisy[i].detach().requires_grad_(True)
                            prediction = predict(i,leaf,sigma,step)
                            logits_i = prediction["sequence_logits_I"][:,seq[i]][:,:,aa]
                            # Recompute every continuous cycle; other track enters
                            # only via its exact output cotangent, not a frozen trunk.
                            gradient, = torch.autograd.grad(logits_i,leaf,
                                grad_outputs=cotangents[i].to(logits_i.dtype))
                        gradient = gradient.clone()
                        gradient[:,fixed[i]] = 0
                        gradients.append(gradient.detach()*sigma.square())
                    energy = energy_js
                else:
                    gradients, energy = structural_gradient(clean,feats,ca,
                        cfg["block"] if method == "coarse_coupling" else 1)
            unit_ratios, raw_ratios, cap = [None,None], [0.,0.], 1.
            corrected = clean
            masks = [~f for f in fixed] if method == "sequence_coupling" else [torch.isin(torch.arange(len(f),device=device),ix) for f,ix in zip(fixed,idx)]
            if gradients is not None:
                unit_update = [h*g/sigma for g in gradients]
                unit_ratios = [float(c[:,m].float().norm()/u[:,m].float().norm().clamp_min(1e-12)) for c,u,m in zip(unit_update,native_update,masks)]
                if strength > 0:
                    actual, raw_ratios, cap = cap_correction([strength*c for c in unit_update],native_update,masks,cfg["correction_cap"])
                    corrected = [c-strength*cap*g for c,g in zip(clean,gradients)]
            delta = [(v-c)/sigma for v,c in zip(noisy,corrected)]
            next_x = [v+h*d for v,d in zip(noisy,delta)]
            if active:
                mix = 0.5*delta[0][:,idx[0]] + 0.5*delta[1][:,idx[1]]
                common = noisy[0][:,idx[0]] + h*mix
                for i in range(2):
                    next_x[i][:,idx[i]] = common
            elif method == "residual_consensus":
                mixed = residual(delta[0][:,idx[0]],delta[1][:,idx[1]],alpha)
                for i in range(2):
                    next_x[i][:,idx[i]] = noisy[i][:,idx[i]] + h*mixed[i]
            for i in range(2):
                if not torch.isfinite(next_x[i]).all():
                    raise FloatingPointError(f"nonfinite state at update {step}, track {i}")
                if not torch.equal(next_x[i][:,fixed[i]],x[i][:,fixed[i]]):
                    raise AssertionError("fixed coordinates changed")
            total_correction_ratios=[float((v-(z+u))[:,ix].float().norm()/u[:,ix].float().norm().clamp_min(1e-12))
                for v,z,u,ix in zip(next_x,noisy,native_update,idx)]
            vectors=[d[:,ix].float().flatten() for d,ix in zip(native_delta,idx)]
            direction_cosine=float(torch.nn.functional.cosine_similarity(vectors[0],vectors[1],dim=0))
            x = [v.detach() for v in next_x]
            row = state(step+1,following) | dict(progress=p,sigma_hat=float(sigma),
                structural_coupling=active,alpha=alpha,noise_correlation=rho,
                churn_active=bool(gamma),churn_A_hashes=[digest(e[:,ix]) for e,ix in zip(eps,idx)],
                base_shared_noise_hash=digest(shared),
                js=energy_js,sequence_entropy=entropy,temperature=temperature,
                strength=strength,energy=energy,unit_correction_ratios=unit_ratios,
                raw_correction_ratios=raw_ratios,cap_factor=cap,
                total_A_correction_ratios=total_correction_ratios,native_A_direction_cosine=direction_cosine,
                native_update_norms=[float(u[:,m].float().norm()) for u,m in zip(native_update,masks)],
                clean_ca_rmsd=float(rmsd(clean[0][:,ca[0]],clean[1][:,ca[1]])),
                corrected_clean_ca_rmsd=float(rmsd(corrected[0][:,ca[0]],corrected[1][:,ca[1]])))
            trace.append(row)
            for i in range(2):
                trajectory[i]["t_hats"].append(sigma.detach().cpu())
                probs = outs[i]["sequence_logits_I"].softmax(-1).cpu()
                trajectory[i]["sequence_entropy_traj"].append(-(probs*(probs+1e-10).log()).sum(-1))
                if cfg["debug"]:
                    trajectory[i]["X_noisy_L_traj"].append((self.sigma_data*noisy[i]/torch.sqrt(sigma**2+self.sigma_data**2)).detach().cpu())
                    trajectory[i]["X_denoised_L_traj"].append(corrected[i].detach().cpu())
            if step % max(1,n//10) == 0 or step == n-1:
                print(f"{method}: {step+1}/{n}, A RMSD={row['ca_rmsd']:.4f}, JS={energy_js:.5f}",flush=True)
        outputs = {}
        for i in range(2):
            outputs[f"track_{i+1}"] = trajectory[i] | dict(X_L=x[i],
                sequence_logits_I=outs[i]["sequence_logits_I"],sequence_indices_I=outs[i]["sequence_indices_I"])
        consensus = (0.5*selected[0].float()+0.5*selected[1].float()).softmax(-1)
        logs=[z.float().log_softmax(-1) for z in selected]
        log_mean=torch.logaddexp(*logs)-torch.log(torch.tensor(2.,device=device))
        conflict=.5*sum((z.exp()*(z-log_mean)).sum(-1) for z in logs)
        outputs["coupling_metadata"] = coupling_metadata | dict(
            implementation="isolated coupling policy; original EDM direction and noise schedule",
            experiment=cfg,initial_A_hashes=initial_hashes,trace=trace,
            elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            shared_sequence_readout=dict(token_indices=[s.cpu().tolist() for s in seq],
                alphabet_indices=aa.cpu().tolist(),probabilities=consensus.cpu().tolist(),
                per_position_js=conflict.cpu().tolist(),native_argmax_agreement=float((selected[0].argmax(-1)==selected[1].argmax(-1)).float().mean()) if selected[0].numel() else None),
            diagnostics={},superdiff_exact=False)
        return outputs
