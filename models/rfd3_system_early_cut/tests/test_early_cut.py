"""Container-native tests: python -m unittest discover -s ... -p test_early_cut.py."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from biotite.structure import AtomArray

from rfd3_system.model.inference_sampler import (
    SampleDiffusionWithSuperDiffSharedChainProxy as OriginalSampler,
)
from rfd3_system_early_cut.engine import _to_jsonable
from rfd3_system_early_cut.model.inference_sampler import (
    ConditionalDiffusionSampler,
    SampleDiffusionWithSuperDiffSharedChainProxy as EarlyCutSampler,
)
from rfd3_system_early_cut.system.chains import build_shared_ca_atom_map
from rfd3_system_early_cut.system.early_cut import resolve_cutoff, state_ca_rmsd


class LinearDenoiser:
    def _denoise_once(self, *, X_noisy_L, f, **kwargs):
        prediction = X_noisy_L * f["scale"]
        fixed = f["is_motif_atom_with_fixed_coord"]
        prediction[:, fixed] = X_noisy_L[:, fixed]
        self.calls.append((X_noisy_L.clone(), prediction.clone()))
        return {"X_L": prediction}


class NewSampler(LinearDenoiser, EarlyCutSampler):
    pass


class OldSampler(LinearDenoiser, OriginalSampler):
    pass


def track(n, scale, fixed_offset=0):
    coords = torch.zeros(n, 3)
    coords[-1] = fixed_offset
    fixed = torch.zeros(n, dtype=torch.bool)
    fixed[-1] = True
    return dict(f={"ref_element": torch.zeros(n), "is_motif_atom_with_fixed_coord": fixed,
                   "scale": scale}, initializer_outputs={}, coord_atom_lvl_to_be_noised=coords)


def run(sampler_class=NewSampler, *, partial=None, identical=False, **settings):
    sampler = sampler_class(num_timesteps=6, sigma_data=1, s_max=2, s_min=0.1, p=1,
                            gamma_min=0, step_scale=1, **settings)
    sampler.calls = []
    t1, t2 = track(5, 0.2), track(7, 0.2 if identical else 0.6, 1)
    if partial is not None:
        t1["f"]["partial_t"] = t2["f"]["partial_t"] = torch.tensor(partial)
    args = dict(track_1=t1, track_2=t2, diffusion_module=None, diffusion_batch_size=2,
                shared_update_atom_indices_1=torch.tensor([0, 2, 1]),
                shared_update_atom_indices_2=torch.tensor([3, 0, 2]), coupling_metadata={})
    if sampler_class is NewSampler:
        args["shared_ca_atom_indices"] = {
            "all_1": torch.tensor([0, 1, 4]), "all_2": torch.tensor([3, 2, 6]),
            "movable_1": torch.tensor([0, 1]), "movable_2": torch.tensor([3, 2]),
        }
    torch.manual_seed(123)
    with torch.no_grad():
        result = sampler.sample_coupled_superdiff_proxy(**args)
    return result, sampler, torch.random.get_rng_state()


class EarlyCutTests(unittest.TestCase):
    def assert_same_tracks(self, one, two):
        for key in ("track_1", "track_2"):
            self.assertTrue(torch.equal(one[key]["X_L"], two[key]["X_L"]))
            for field in ("X_noisy_L_traj", "X_denoised_L_traj"):
                for a, b in zip(one[key][field], two[key][field]):
                    self.assertTrue(torch.equal(a, b))

    def test_exact_original_parity_and_rng(self):
        old, _, old_rng = run(OldSampler)
        new, _, new_rng = run()
        self.assert_same_tracks(old, new)
        self.assertTrue(torch.equal(old_rng, new_rng))
        for key, values in old["coupling_metadata"]["diagnostics"].items():
            for before, after in zip(values, new["coupling_metadata"]["diagnostics"][key]):
                self.assertTrue(torch.equal(before, after))
        for settings in ({"coupling_cut_fraction": 1}, {"coupling_cut_sigma": 0}):
            alternate, _, _ = run(**settings)
            self.assert_same_tracks(new, alternate)

    def test_release_actual_state_and_null_proxy(self):
        result, sampler, _ = run(coupling_cut_fraction=0.4)
        meta = result["coupling_metadata"]
        rmsd = torch.stack(meta["shared_chain_state"]["movable_ca_rmsd"])
        self.assertEqual(rmsd.shape, (6, 2))
        self.assertTrue(torch.equal(rmsd[:3], torch.zeros(3, 2)))
        self.assertTrue(torch.all(rmsd[3:] > 0))
        self.assertEqual(meta["diagnostics"]["coupling_active"], [True, True, False, False, False])
        self.assertEqual(meta["diagnostics"]["kappa"][2:], [[None, None]] * 3)
        self.assertTrue(torch.all(meta["shared_chain_state"]["all_ca_rmsd"][0] > 0))
        # Context-specific denoiser predictions already differ in the coupled phase.
        self.assertFalse(torch.equal(sampler.calls[0][1][:, [0, 1]], sampler.calls[1][1][:, [3, 2]]))
        final = state_ca_rmsd(result["track_1"]["X_L"], result["track_2"]["X_L"], [0, 1], [3, 2])
        self.assertTrue(torch.equal(final, rmsd[-1]))
        json.dumps(_to_jsonable(meta), allow_nan=False)

    def test_independent_update_and_no_recoupling(self):
        result, sampler, _ = run(coupling_cut_fraction=0)
        schedule = sampler._construct_inference_noise_schedule(torch.device("cpu"))
        for j, name in enumerate(("track_1", "track_2")):
            noisy, predicted = sampler.calls[-2 + j]
            t_hat = result[name]["t_hats"][-1]
            expected = noisy + (schedule[-1] - t_hat) * ((noisy - predicted) / t_hat)
            self.assertTrue(torch.equal(expected, result[name]["X_L"]))
        rmsd = torch.stack(result["coupling_metadata"]["shared_chain_state"]["movable_ca_rmsd"])
        self.assertTrue(torch.all(rmsd[0] == 0))
        self.assertTrue(torch.all(rmsd[1:] > 0))
        with patch("rfd3_system_early_cut.model.inference_sampler.solve_two_track_proxy_kappa",
                   side_effect=AssertionError("solver must not run after release")):
            run(coupling_cut_fraction=0)

    def test_shared_noise_and_fixed_motifs(self):
        for churn in (0.0, 0.6):
            baseline, _, rng = run(gamma_0=churn)
            result, sampler, cut_rng = run(coupling_cut_fraction=0.4, gamma_0=churn)
            self.assertTrue(torch.equal(rng, cut_rng))
            diag = result["coupling_metadata"]["diagnostics"]
            self.assertTrue(torch.all(torch.stack(diag["shared_noise_max_abs_difference"]) == 0))
            # Unequal track sizes and reordered maps; fixed coordinates never receive noise.
            for left, right in zip(sampler.calls[::2], sampler.calls[1::2]):
                self.assertTrue(torch.all(left[0][:, -1] == 0))
                self.assertTrue(torch.all(right[0][:, -1] == 1))
            # Identical atomwise denoisers remain identical even after immediate release.
            same, _, _ = run(coupling_cut_fraction=0, gamma_0=churn, identical=True)
            self.assertTrue(torch.all(torch.stack(same["coupling_metadata"]["shared_chain_state"]["movable_ca_rmsd"]) == 0))

    def test_fraction_sigma_equivalence_prechurn_and_partial(self):
        for partial in (None, 1.3):
            result, sampler, _ = run(coupling_cut_fraction=0.4, partial=partial)
            cutoff = result["coupling_metadata"]["cutoff"]
            equivalent, _, _ = run(coupling_cut_sigma=cutoff["release_pre_churn_sigma"], partial=partial)
            self.assert_same_tracks(result, equivalent)
            self.assertGreater(cutoff["release_t_hat"], cutoff["release_pre_churn_sigma"])
        self.assertEqual(resolve_cutoff([8, 6, 4, 2], sigma=6)["coupled_update_count"], 1)
        self.assertIsNone(resolve_cutoff([8, 6, 4, 2], sigma=2)["first_independent_update_index"])
        self.assertEqual(resolve_cutoff([8, 6, 4, 2], sigma=9)["coupled_update_count"], 0)

    def test_invalid_controls_and_sampler_kind(self):
        for settings in ({"coupling_cut_fraction": -0.1}, {"coupling_cut_fraction": 1.1},
                         {"coupling_cut_fraction": float("nan")}, {"coupling_cut_sigma": -1},
                         {"coupling_cut_sigma": float("inf")},
                         {"coupling_cut_fraction": 0.5, "coupling_cut_sigma": 2}):
            with self.assertRaises(ValueError):
                NewSampler(**settings)
        with self.assertRaises(ValueError):
            ConditionalDiffusionSampler(kind="default", coupling_cut_fraction=0)
        with self.assertRaises(ValueError):
            run(partial=0.001)

    def test_ca_metric_ignores_sidechains_and_does_not_align(self):
        x = torch.zeros(2, 4, 3)
        y = x.clone()
        y[:, 2:] = 100
        self.assertTrue(torch.all(state_ca_rmsd(x, y, [0, 1], [0, 1]) == 0))
        y[:, :2, 0] = 3
        self.assertTrue(torch.all(state_ca_rmsd(x, y, [0, 1], [0, 1]) == 3))

    def test_fixed_sep_ser_ca_mapping(self):
        def atoms(names, residues, resnames, source):
            a = AtomArray(len(names))
            a.chain_id[:] = "A"
            a.atom_name = np.array(names)
            a.res_id = np.array(residues)
            a.res_name = np.array(resnames)
            a.set_annotation("src_component", np.array(source))
            return a
        a = atoms(["N", "CA", "C", "CA", "P"], [1, 1, 1, 90, 90],
                  ["GLY"]*3 + ["SEP"]*2, [""]*3 + ["A240"]*2)
        b = atoms(["N", "CA", "C", "OG", "CA"], [1, 1, 1, 100, 100],
                  ["GLY"]*3 + ["SER"]*2, [""]*3 + ["A240"]*2)
        mask = np.array([False]*3 + [True]*2)
        mapping = build_shared_ca_atom_map(a, b, "A", mask, mask)
        self.assertEqual(mapping["all_1"].tolist(), [1, 3])
        self.assertEqual(mapping["all_2"].tolist(), [1, 4])
        self.assertEqual(mapping["movable_1"].tolist(), [1])
        b.atom_name[4] = "CB"
        with self.assertRaises(ValueError):
            build_shared_ca_atom_map(a, b, "A", mask, mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
