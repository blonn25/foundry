import copy
import unittest
import torch
from rfd3_variants.policies import *
from rfd3_variants.conditions import conditions
from rfd3_variants.sampler import ExperimentalSampler, structural_gradient
from rfd3_system.model.inference_sampler import SampleDiffusionWithSuperDiffSharedChainProxy as Original


class Toy(torch.nn.Module):
    def forward(self,X_noisy_L,t,f,**kwargs):
        clean=0.25*X_noisy_L+f["shift"]
        fixed=f["is_motif_atom_with_fixed_coord"]
        clean[:,fixed]=X_noisy_L[:,fixed]
        weights=torch.arange(32,device=X_noisy_L.device).float()/32
        logits=torch.sin(X_noisy_L.sum(-1,keepdim=True)*weights)
        return dict(X_L=clean,sequence_logits_I=logits,sequence_indices_I=logits.argmax(-1))


def fixture():
    def track(n,shift):
        f=dict(ref_element=torch.ones(n),is_motif_atom_with_fixed_coord=torch.zeros(n,dtype=torch.bool),
               is_motif_atom_with_fixed_seq=torch.zeros(n,dtype=torch.bool),
               is_ca=torch.ones(n,dtype=torch.bool),atom_to_token_map=torch.arange(n),shift=shift)
        f["is_motif_atom_with_fixed_coord"][-1]=True
        return dict(f=f,initializer_outputs={},coord_atom_lvl_to_be_noised=torch.zeros(1,n,3))
    return dict(track_1=track(6,.2),track_2=track(7,.4),
        shared_update_atom_indices_1=torch.tensor([0,1,2]),shared_update_atom_indices_2=torch.tensor([0,1,2]),
        diffusion_module=Toy(),diffusion_batch_size=1,coupling_metadata={})


def run(method="mean_5050",old=False,**overrides):
    cls=Original if old else ExperimentalSampler
    obj=cls(num_timesteps=6,sigma_data=1,s_max=2,s_min=.1,p=1,
            gamma_min=0,step_scale=1,proxy_norm_weight=.5)
    if not old:
        obj.experiment=profile(method)|overrides
    torch.manual_seed(13)
    with torch.no_grad():
        result=obj.sample_coupled_superdiff_proxy(**fixture())
    return result,torch.random.get_rng_state()


class Operators(unittest.TestCase):
    def test_residue_lift_preserves_partners_and_fixed_atoms(self):
        torch.manual_seed(70)
        clean=[torch.randn(1,8,3),torch.randn(1,8,3)]
        f=dict(atom_to_token_map=torch.tensor([0,0,1,1,2,2,3,3]),
            is_motif_atom_with_fixed_coord=torch.tensor([False]*5+[True,False,False]))
        ca=[torch.tensor([4,0,2])]*2
        gradients,e=structural_gradient(clean,[f,f],ca,1)
        for g in gradients:
            self.assertTrue(torch.equal(g[:,0],g[:,1]))
            self.assertTrue(torch.equal(g[:,2],g[:,3]))
            self.assertEqual(float(g[:,5:].abs().sum()),0.)
        corrected=[x-.1*g for x,g in zip(clean,gradients)]
        self.assertLess(float(distance_energy(corrected[0][:,ca[0]],corrected[1][:,ca[1]])),e)

    def test_matrix(self):
        rows=conditions()
        self.assertEqual(len(rows),37)
        self.assertEqual(50*4*2*len(rows),14800)

    def test_schedules(self):
        for kind in ("linear","cosine"):
            s=dict(start=.1,end=1,kind=kind)
            self.assertEqual(schedule(s,0),.1)
            self.assertEqual(schedule(s,1),1)
            self.assertAlmostEqual(schedule(s,.5),.55)
        self.assertEqual([released(i,5,.7) for i in range(5)],[False]*3+[True]*2)

    def test_residual_endpoints(self):
        a,b=torch.randn(2,7,3),torch.randn(2,7,3)
        u,v=residual(a,b,0)
        self.assertTrue(torch.equal(u,v))
        u,v=residual(a,b,1)
        self.assertTrue(torch.equal(u,a) and torch.equal(v,b))

    def test_noise_statistics(self):
        torch.manual_seed(17)
        s,a,b=torch.randn(3,100000)
        for rho in (0,.5,.75,1):
            u,v=correlated(s,a,b,rho)
            self.assertAlmostEqual(float(u.var()),1,delta=.025)
            self.assertAlmostEqual(float(torch.corrcoef(torch.stack([u,v]))[0,1]),rho,delta=.02)
            if rho==1:
                self.assertTrue(torch.equal(u,v) and torch.equal(u,s))

    def test_energy_rigid_invariance_and_gradient(self):
        torch.manual_seed(42)
        a,b=torch.randn(1,12,3),torch.randn(1,12,3)
        rotation=torch.linalg.qr(torch.randn(3,3)).Q
        for block in (1,3,5):
            self.assertAlmostEqual(float(distance_energy(a,b,block)),float(distance_energy(a@rotation+7,b,block)),places=6)
        a.requires_grad_()
        e=distance_energy(a,b)
        g,=torch.autograd.grad(e,a)
        self.assertLess(float(distance_energy(a-.1*g,b)),float(e))
        direction=torch.randn_like(a)
        fd=(distance_energy(a+.001*direction,b)-distance_energy(a-.001*direction,b))/.002
        self.assertAlmostEqual(float(fd),float((g*direction).sum()),delta=1e-5)

    def test_js_gradient(self):
        a,b=torch.randn(1,4,19,requires_grad=True),torch.randn(1,4,19,requires_grad=True)
        self.assertAlmostEqual(float(js_energy(a,a)),0.,places=6)
        e=js_energy(a,b)
        ga,gb=torch.autograd.grad(e,(a,b))
        self.assertLess(float(js_energy(a-ga,b-gb)),float(e))

    def test_rmsd_alignment(self):
        a=torch.randn(1,15,3)
        self.assertLess(float(rmsd(a,a+9)),1e-5)
        self.assertGreater(float(rmsd(a,a+9,False)),10)

    def test_sequence_correction_identity(self):
        x,c,g=torch.randn(3,4,3)
        sigma,h,lam=4.,-2.,.3
        direct=x+h*(x-(c-lam*sigma**2*g))/sigma
        score=x+h*((x-c)/sigma+lam*sigma*g)
        torch.testing.assert_close(direct,score)

    def test_joint_cap(self):
        u=[torch.ones(1,4,3)]*2
        c=[u[0]*2,u[0]*4]
        corrected,r,factor=cap_correction(c,u,[torch.ones(4,dtype=torch.bool)]*2)
        self.assertEqual(factor,.125)
        self.assertLessEqual(float(corrected[1].norm()/u[1].norm()),.5)


class Loops(unittest.TestCase):
    def test_exact_native_parity(self):
        expected,rng=run(old=True)
        actual,new_rng=run(debug=True)
        self.assertTrue(torch.equal(rng,new_rng))
        for key in ("track_1","track_2"):
            self.assertTrue(torch.equal(expected[key]["X_L"],actual[key]["X_L"]))
            for a,b in zip(expected[key]["X_denoised_L_traj"],actual[key]["X_denoised_L_traj"]):
                self.assertTrue(torch.equal(a,b))

    def test_residual_endpoint_rollouts(self):
        for alpha,method in [(0,"mean_5050"),(1,"uncoupled")]:
            a,_=run("residual_consensus",alpha=alpha)
            b,_=run(method)
            for track in ("track_1","track_2"):
                self.assertTrue(torch.equal(a[track]["X_L"],b[track]["X_L"]))

    def test_release_and_noise(self):
        a,_=run("late_uncoupling",release_fraction=.4)
        rows=a["coupling_metadata"]["trace"]
        self.assertEqual(len(rows),6)
        self.assertEqual([r["structural_coupling"] for r in rows[1:]],[True,True,False,False,False])
        for r in rows[1:]:
            self.assertEqual(*r["churn_A_hashes"])
        self.assertEqual(rows[2]["ca_frame_rmsd"],0)
        self.assertGreater(rows[3]["ca_frame_rmsd"],0)

    def test_zero_guidance_and_partner_identity(self):
        base,_=run("uncoupled")
        for method in ("soft_guidance","coarse_coupling","sequence_coupling"):
            result,_=run(method)
            for key in ("track_1","track_2"):
                self.assertTrue(torch.equal(base[key]["X_L"],result[key]["X_L"]))
        hard,_=run("mean_5050")
        for key in ("track_1","track_2"):
            # Toy partners do not depend on A; all partner randomness is unchanged.
            self.assertTrue(torch.equal(hard[key]["X_L"][:,3:],base[key]["X_L"][:,3:]))

    def test_guided_paths_finite(self):
        for method in ("soft_guidance","sequence_coupling"):
            result,_=run(method,strength=.1)
            self.assertTrue(torch.isfinite(result["track_1"]["X_L"]).all())
            self.assertTrue(any(r["unit_correction_ratios"][0]>0 for r in result["coupling_metadata"]["trace"][1:]))


if __name__=="__main__":
    unittest.main()
