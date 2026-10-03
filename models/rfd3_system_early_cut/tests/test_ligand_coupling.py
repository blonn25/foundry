"""Explicit ligand mapping, release semantics and original RNG preservation."""
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from biotite.structure import AtomArray, BondList

from test_early_cut import run
from rfd3_system_early_cut.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3_system_early_cut.system.ligand_coupling import normalize_ligand_pairs, resolve_ligand_pairs

MAP = dict(mapped_1=[3], mapped_2=[1], all_1=[3], all_2=[1, 4])


def fixtures():
    sources, examples = [], []
    for track, names in enumerate((['C1', 'O1'], ['CX', 'OX', 'C2'])):
        a = AtomArray(len(names))
        a.chain_id[:] = 'L'
        a.res_id[:] = 1
        a.res_name[:] = 'AC' if track == 0 else 'BU'
        a.atom_name = np.array(names)
        a.element = np.array(['C', 'O'] + (['C'] if track else []))
        a.bonds = BondList(len(a), np.array([[0, 1, 2]] + ([[0, 2, 1]] if track else [])))
        sources.append(a)
        prepared = a.copy()
        prepared.chain_id[:] = 'B'  # Source selectors must survive native chain compaction.
        prepared.set_annotation('gt_atom_name', np.array(names))
        prepared.set_annotation('is_ligand', np.ones(len(a), dtype=bool))
        prepared.set_annotation('is_protein', np.zeros(len(a), dtype=bool))
        examples.append(dict(atom_array=prepared, feats=dict(is_motif_atom_with_fixed_coord=np.zeros(len(a), dtype=bool),
                         is_motif_atom_with_fixed_seq=np.ones(len(a), dtype=bool))))
    pairs = [dict(track_1=dict(chain='L', residue=1, atom=a), track_2=dict(chain='L', residue=1, atom=b))
             for a, b in zip(['C1', 'O1'], ['CX', 'OX'])]
    return pairs, sources, examples


class LigandCouplingTests(unittest.TestCase):
    def test_explicit_mapping_different_names_and_substituents(self):
        pairs, sources, examples = fixtures()
        indices, meta = resolve_ligand_pairs(pairs, *sources, *examples)
        self.assertEqual(indices['mapped_1'], [0, 1])
        self.assertEqual(indices['mapped_2'], [0, 1])
        self.assertEqual(meta['boundary_bonds']['track_1'], [])
        self.assertEqual(meta['boundary_bonds']['track_2'][0]['outside_atom'], 'C2')
        reverse = [pairs[1], pairs[0]]
        self.assertEqual(resolve_ligand_pairs(reverse, *sources, *examples)[0]['mapped_1'], [1, 0])

    def test_invalid_selectors_and_chemistry_rejected(self):
        pairs, sources, examples = fixtures()
        for bad in ('L1', [{}], [pairs[0], pairs[0]], [{'track_1': pairs[0]['track_1']} ]):
            with self.assertRaises(ValueError):
                normalize_ligand_pairs(bad)
        for change in ('missing', 'element', 'bond', 'fixed', 'chemistry', 'protein'):
            pairs, sources, examples = fixtures()
            if change == 'missing': pairs[0]['track_1']['atom'] = 'MISSING'
            if change == 'element': sources[1].element[0] = 'N'
            if change == 'bond': sources[1].bonds = BondList(3, np.array([[0, 1, 1], [0, 2, 1]]))
            if change == 'fixed': examples[0]['feats']['is_motif_atom_with_fixed_coord'][0] = True
            if change == 'chemistry': examples[0]['feats']['is_motif_atom_with_fixed_seq'][0] = False
            if change == 'protein': examples[0]['atom_array'].is_ligand[0] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve_ligand_pairs(pairs, *sources, *examples)

    def test_ligand_release_shared_noise_and_protein_rng_parity(self):
        baseline_noise = None
        for cut in (0, .4, 1):
            events = []
            result, _, rng = run(ligand_map=MAP, coupling_cut_fraction=cut, proxy_norm_weight=.5,
                                  observer=events.append)
            old, _, old_rng = run(coupling_cut_fraction=cut, proxy_norm_weight=.5)
            self.assertTrue(torch.equal(rng, old_rng))
            for track, selection in [('track_1', [0, 2, 1]), ('track_2', [3, 0, 2])]:
                self.assertTrue(torch.equal(result[track]['X_L'][:, selection], old[track]['X_L'][:, selection]))
            states = torch.stack(result['coupling_metadata']['shared_ligand_state']['mapped_atom_rmsd'])
            k = int(cut*5)
            self.assertTrue(torch.all(states[:k+1] == 0))
            if k < 5:
                self.assertTrue(torch.all(states[k+1:] > 0))
            noise = [e['mapped_ligand_noise_1'] for e in events[1:]]
            for event in events[1:]:
                self.assertTrue(torch.equal(event['mapped_ligand_noise_1'], event['mapped_ligand_noise_2']))
            if baseline_noise is None:
                baseline_noise = noise
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline_noise, noise)))
            # Unique ligand atom does not join coordinate coupling.
            self.assertFalse(torch.equal(result['track_2']['X_L'][:, 4], result['track_2']['X_L'][:, 1]))

    def test_ligand_sigma_equivalence_and_no_solver_after_release(self):
        result, _, _ = run(ligand_map=MAP, coupling_cut_fraction=.4)
        sigma = result['coupling_metadata']['cutoff']['release_pre_churn_sigma']
        same, _, _ = run(ligand_map=MAP, coupling_cut_sigma=sigma)
        for track in ('track_1', 'track_2'):
            self.assertTrue(torch.equal(result[track]['X_L'], same[track]['X_L']))
        with patch('rfd3_system_early_cut.model.inference_sampler.solve_two_track_proxy_kappa',
                   side_effect=AssertionError('No proxy after release')):
            run(ligand_map=MAP, coupling_cut_fraction=0)

    def test_repeated_inference_calls_advance_ligand_noise(self):
        first, second = [], []
        run(ligand_map=MAP, coupling_cut_fraction=.4, observer=first.append)
        # Emulate a subsequent engine batch without resetting its global RNG.
        with patch('torch.manual_seed'):
            run(ligand_map=MAP, coupling_cut_fraction=.4, observer=second.append)
        self.assertFalse(torch.equal(first[1]['mapped_ligand_noise_1'], second[1]['mapped_ligand_noise_1']))
        for event in second[1:]:
            self.assertTrue(torch.equal(event['mapped_ligand_noise_1'], event['mapped_ligand_noise_2']))

    def test_observer_copies_do_not_mutate_ligands(self):
        reference, _, rng = run(ligand_map=MAP, coupling_cut_fraction=.4)
        def mutate(event):
            for key, value in event.items():
                if isinstance(value, torch.Tensor): value.fill_(123)
        actual, _, same_rng = run(ligand_map=MAP, coupling_cut_fraction=.4, observer=mutate)
        self.assertTrue(torch.equal(rng, same_rng))
        for track in ('track_1', 'track_2'):
            self.assertTrue(torch.equal(reference[track]['X_L'], actual[track]['X_L']))

    def test_public_hydra_input_and_mode_guard(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base='1.3'):
            cfg = compose(config_name='inference', overrides=['inputs=null', 'out_dir=/project/outputs/unused',
                'coupling_mode=superdiff_shared_chain', 'inference_sampler.kind=superdiff_shared_chain',
                'coupled_ligand_atom_pairs=[{track_1:{chain:L,residue:1,atom:C1},track_2:{chain:L,residue:1,atom:CX}}]'])
        options = {k:v for k,v in OmegaConf.to_container(cfg, resolve=True).items()
                   if k not in {'_target_', 'inputs', 'out_dir', 'n_batches'}}
        engine = RFD3InferenceEngine(**asdict(RFD3InferenceConfig(**options)))
        self.assertEqual(engine.coupled_ligand_atom_pairs[0]['track_2']['atom'], 'CX')
        options['coupling_mode'] = None
        with self.assertRaisesRegex(ValueError, 'Ligand coupling requires'):
            RFD3InferenceEngine(**asdict(RFD3InferenceConfig(**options)))


if __name__ == '__main__':
    unittest.main()
