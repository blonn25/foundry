"""Known rigid transforms, internal distortion, and a complete synthetic sweep."""
import csv
import gzip
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np

from rfd3_system_early_cut.experiments.ligand_codiffusion_analysis import (
    analyze, ligand_geometry, protein_fit_ligand_rmsd,
)
from rfd3_system_early_cut.experiments.substrate_common import input_information, read_cif, read_json, write_json


class LigandAnalysisTests(unittest.TestCase):
    def test_pose_separated_from_internal_geometry(self):
        x = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        a = x+8
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        y, b = x@rotation+5, a@rotation+5
        self.assertLess(protein_fit_ligand_rmsd(x, y, a, b), 1e-12)
        self.assertAlmostEqual(protein_fit_ligand_rmsd(x, y, a, b+[2, 0, 0]), 2)
        result = ligand_geometry(b, a, [[0, 1, 1], [0, 2, 1]])
        self.assertLess(result['ligand_internal_reference_rmsd'], 1e-12)
        self.assertEqual(result['ligand_bond_length_rmse'], 0)
        distorted = a.copy()
        distorted[1] += [1, 0, 0]
        result = ligand_geometry(distorted, a, [[0, 1, 1], [0, 2, 1]])
        self.assertAlmostEqual(result['ligand_bond_length_rmse'], np.sqrt(.5))
        self.assertEqual(result['ligand_bond_length_max_deviation'], 1)
        self.assertGreater(result['ligand_internal_reference_rmsd'], 0)

    def test_batched_protein_fit_and_reflection(self):
        x = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        value = protein_fit_ligand_rmsd(np.stack([x, x]), np.stack([x+5, x]),
                                      np.stack([x+2, x+2]), np.stack([x+7, x+3]))
        np.testing.assert_allclose(value, [0, np.sqrt(3)], atol=1e-12)
        self.assertGreater(protein_fit_ligand_rmsd(x, x*[-1, 1, 1], x, x*[-1, 1, 1]), .1)


def synthetic_sweep(project, destination):
    """Create known 3 Å CA / 3.5 Å ligand translations after each release."""
    from biotite.structure import AtomArray, concatenate
    from biotite.structure.io.pdbx import CIFFile, set_structure
    destination.mkdir(parents=True, exist_ok=False)
    template = read_json(Path(__file__).resolve().parents[1]/'experiments/ligand_codiffusion_sweep/config.json')
    provenance = input_information(project, template)
    # The real molecule inputs are included so custom CIF bond reading is exercised.
    config = template | {key: provenance[key] for key in ('ligands', 'origin')}
    config.update(seeds=[101], foundry_revision='synthetic-validation', input_dir='inputs', fixed_baseline='fixed')
    shutil.copytree(project/template['input_dir'], destination/'inputs')
    roots = [destination/'free', destination/'fixed']
    for root in roots:
        root.mkdir()
        write_json(root/'resolved_config.json', config)
        (root/'seed_101_synthetic').mkdir()
    t = np.arange(120)
    protein = AtomArray(120)
    protein.coord = np.column_stack([2.3*np.cos(t*np.deg2rad(100)), 2.3*np.sin(t*np.deg2rad(100)), 1.5*t])
    protein.atom_name[:] = 'CA'
    protein.element[:] = 'C'
    protein.chain_id[:] = 'A'
    protein.res_id = t+1
    protein.res_name[:] = 'ALA'
    sigma = np.geomspace(100, .01, 200)
    ligand_inputs = [read_cif(destination/'inputs'/f'{name}.cif') for name in config['track_ligands']]
    names = {str(i): list(a.atom_name) for i, a in enumerate(ligand_inputs, 1)}
    mapped_idx = [[list(a.atom_name).index(p[f'track_{i}']['atom']) for p in config['coupled_ligand_atom_pairs']]
                  for i, a in enumerate(ligand_inputs, 1)]
    for percent in config['coupled_percentages']:
        k = percent*199//100
        dirs = [r/'seed_101_synthetic'/f'coupled_{percent:03d}' for r in roots]
        for directory in dirs:
            directory.mkdir()
        ca = np.broadcast_to(protein.coord, (200, 2, 120, 3)).copy()
        for directory in dirs:
            np.savez_compressed(directory/'ca_states.npz', ca=ca, sigma=sigma, completed_updates=np.arange(200))
        ca[k+1:, 1] += [3, 0, 0]
        np.savez_compressed(dirs[0]/'ca_states.npz', ca=ca, sigma=sigma, completed_updates=np.arange(200))
        ligand_states = [np.broadcast_to(a.coord, (200, len(a), 3)).copy() for a in ligand_inputs]
        ligand_states[1][k+1:] += [3.5, 0, 0]
        mapped = np.stack([a[:, index] for a, index in zip(ligand_states, mapped_idx)], axis=1)
        np.savez_compressed(dirs[0]/'ligand_states.npz', mapped=mapped, track_1=ligand_states[0], track_2=ligand_states[1])
        write_json(dirs[0]/'ligand_atom_names.json', names)
        audit = dict(initial_shared_sha256='synthetic', churn_sha256=['synthetic']*199, state_count=200,
                     ligand_initial_sha256=['synthetic']*2, ligand_churn_sha256=[['synthetic']*2]*199)
        for directory in dirs:
            write_json(directory/'noise_audit.json', audit)
            (directory/'_COMPLETE').touch()
        write_json(dirs[0]/'pair.json', dict(seed=101, coupled_fraction=percent/100,
                   cutoff=dict(coupled_update_count=k, release_pre_churn_sigma=float(sigma[k]) if k < 199 else None)))
        for i, reference in enumerate(ligand_inputs):
            p, ligand = protein.copy(), reference.copy()
            p.coord, ligand.coord = ca[-1, i], ligand_states[i][-1]
            cif = CIFFile()
            set_structure(cif, concatenate([p, ligand]))
            with gzip.open(dirs[0]/f'synthetic_track{i+1}.cif.gz', 'wt') as handle:
                cif.write(handle)
    for root in roots:
        (root/'seed_101_synthetic/_COMPLETE').touch()
    output = destination/'analysis'
    analyze(destination, roots[0], output)
    with (output/'pairs.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    for row in rows:
        released = float(row['coupled_fraction']) < 1
        np.testing.assert_allclose(float(row['raw_ca_rmsd']), 3 if released else 0, atol=1e-5)
        np.testing.assert_allclose(float(row['delta_raw_ca_rmsd']), 3 if released else 0, atol=1e-5)
        np.testing.assert_allclose(float(row['aligned_ca_rmsd']), 0, atol=1e-5)
        np.testing.assert_allclose(float(row['raw_ligand_rmsd']), 3.5 if released else 0, atol=1e-5)
        np.testing.assert_allclose(float(row['protein_fit_ligand_rmsd']), .5 if released else 0, atol=1e-5)
    print('Synthetic 21-condition analysis passed: rigid pose, internal geometry, baseline pairing, CIF/state mapping.')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fixture-output':
        synthetic_sweep(Path('/project'), Path(sys.argv[2]))
    else:
        unittest.main()
