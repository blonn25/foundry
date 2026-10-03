"""Scientific contracts for trajectory recording and paired structural metrics."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_early_cut import run
from rfd3_system_early_cut.experiments.substrate_common import (
    StateRecorder, jaccard, rmsds, secondary_structure, topology,
)


class SubstrateSweepTests(unittest.TestCase):
    def test_observer_cannot_mutate_sampler_and_does_not_change_rng(self):
        reference, _, rng = run(coupling_cut_fraction=0.4)
        events = []
        def observer(event):
            events.append(event["completed_updates"])
            for value in event.values():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, "cpu")
                    value.fill_(999)
        actual, _, observed_rng = run(coupling_cut_fraction=0.4, observer=observer)
        self.assertEqual(events, list(range(6)))
        self.assertTrue(torch.equal(rng, observed_rng))
        for name in ("track_1", "track_2"):
            self.assertTrue(torch.equal(reference[name]["X_L"], actual[name]["X_L"]))

    def test_shared_noise_matches_across_release_points(self):
        records = []
        for cut in (0, 0.4, 1):
            recorder = StateRecorder()
            run(coupling_cut_fraction=cut, observer=recorder)
            self.assertEqual(recorder.audit()["state_count"], 6)
            self.assertEqual(recorder.max_fixed_drift, 0)
            records.append((recorder.initial_hash, recorder.noise_hashes))
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[0], records[2])

    def test_kabsch_translation_rotation_and_reflection(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3.]])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        y = x @ rotation + 5
        raw, aligned = rmsds(np.stack([x, x]), np.stack([x, y]))
        self.assertEqual(raw[0], 0)
        self.assertGreater(raw[1], 1)
        self.assertTrue(np.all(aligned < 1e-12))
        self.assertGreater(rmsds(x, x * [-1, 1, 1])[1], .1)

    def test_secondary_geometry_and_missing_denominators(self):
        t = np.arange(30)
        helix = np.column_stack([2.3*np.cos(t*np.deg2rad(100)),
                                 2.3*np.sin(t*np.deg2rad(100)), 1.5*t])
        self.assertGreater(np.mean(secondary_structure(helix) == 'a'), .7)
        metrics = topology(helix, helix)
        self.assertEqual(metrics['helix_jaccard'], 1)
        self.assertEqual(metrics['secondary_agreement'], 1)
        self.assertIsNone(metrics['strand_jaccard'])
        self.assertIsNone(jaccard([False], [False]))
        self.assertEqual(jaccard([True, False], [True, True]), .5)
        with patch('rfd3_system_early_cut.experiments.substrate_common.secondary_structure',
                   return_value=np.array(['c']*30)):
            self.assertIsNone(topology(helix, helix)['structured_agreement'])

    def test_cif_final_state_and_fixed_ligand_validation(self):
        import tempfile
        from biotite.structure import AtomArray, concatenate
        from biotite.structure.io.pdbx import CIFFile, set_structure
        from rfd3_system_early_cut.experiments.substrate_analysis import structure_metrics
        t = np.arange(120)
        coords = np.column_stack([2.3*np.cos(t*np.deg2rad(100)),
                                  2.3*np.sin(t*np.deg2rad(100)), 1.5*t]).astype(np.float32)
        protein = AtomArray(120)
        protein.coord = coords
        protein.atom_name[:] = 'CA'
        protein.element[:] = 'C'
        protein.chain_id[:] = 'A'
        protein.res_id = t+1
        protein.res_name[:] = 'ALA'
        ligand = AtomArray(2)
        ligand.coord = np.array([[30, 0, 0], [31, 0, 0]], dtype=np.float32)
        ligand.atom_name = np.array(['C1', 'O1'])
        ligand.element = np.array(['C', 'O'])
        ligand.chain_id[:] = 'L'
        ligand.res_name[:] = 'LIG'
        ligand.res_id[:] = 1
        ligand.hetero[:] = True
        config = {'origin': [0, 0, 0], 'ligands': {'acetate': {'coordinates': {
            str(a.atom_name): a.coord.tolist() for a in ligand}}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.cif'
            cif = CIFFile()
            set_structure(cif, concatenate([protein, ligand]))
            cif.write(path)
            result = structure_metrics(path, coords, config, 'acetate')
            self.assertEqual(result['sequence'], 'A'*120)
            self.assertEqual(result['ca_break_count'], 0)
            self.assertEqual(result['ligand_contact_atom_pairs'], 0)
            self.assertEqual(result['ligand_fixed_max_coordinate_error'], 0)
            with self.assertRaisesRegex(ValueError, 'recorded final state'):
                structure_metrics(path, coords+1, config, 'acetate')
            config['origin'] = [1, 0, 0]
            with self.assertRaisesRegex(ValueError, 'fixed pose'):
                structure_metrics(path, coords, config, 'acetate')
            free = structure_metrics(path, coords, config, 'acetate', fixed_ligand=False)
            self.assertEqual(free['ligand_max_coordinate_displacement'], 1)
            self.assertNotIn('ligand_fixed_max_coordinate_error', free)

    def test_summary_preserves_missing_values_and_valid_counts(self):
        from rfd3_system_early_cut.experiments.substrate_analysis import stats
        self.assertEqual(stats([None, None]), {'n': 0, 'mean': None, 'sd': None})
        self.assertEqual(stats([None, 1]), {'n': 1, 'mean': 1., 'sd': None})
        observed = stats([1., None, 3.])
        self.assertEqual(observed['n'], 2)
        self.assertEqual(observed['mean'], 2.)
        self.assertAlmostEqual(observed['sd'], 2**.5)
        with self.assertRaises(ValueError):
            stats([float('nan')])

    def test_exact_manifest_size_and_endpoints(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/substrate_sweep/config.json'
        config = json.loads(path.read_text())
        rows = [(seed, p) for seed in config['seeds'] for p in config['coupled_percentages']]
        self.assertEqual(len(set(rows)), 210)
        self.assertEqual(config['coupled_percentages'], list(range(100, -1, -5)))
        self.assertEqual(config['sampler']['proxy_norm_weight'], .5)
        self.assertEqual(config['protein_length'], 120)
        self.assertEqual([p*199//100 for p in (0, 20, 40, 60, 80, 100)], [0, 39, 79, 119, 159, 199])


if __name__ == '__main__':
    unittest.main()
