"""Collect every completed pair without selection and measure final topology."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .substrate_common import (
    read_json, read_cif, require, rmsds, topology, write_csv, write_json,
)

AA = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
              'ARNDCQEGHILKMFPSTWYV'))
RADII = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8}
PAIR_METRICS = ['raw_ca_rmsd', 'aligned_ca_rmsd', 'sequence_identity',
                'secondary_agreement', 'structured_agreement', 'helix_jaccard',
                'strand_jaccard', 'contact_jaccard',
                'ac_helix_fraction', 'bu_helix_fraction', 'ac_strand_fraction',
                'bu_strand_fraction', 'ac_coil_fraction', 'bu_coil_fraction']


def stats(values):
    array = np.array([v for v in values if v is not None], dtype=float)
    require(np.isfinite(array).all(), 'Nonfinite measured metric')
    return dict(n=len(array), mean=float(array.mean()) if len(array) else None,
                sd=float(array.std(ddof=1)) if len(array) > 1 else None)


def structure_metrics(path, expected_ca, config, ligand, *, fixed_ligand=True):
    atoms = read_cif(path)
    protein = atoms.chain_id == 'A'
    ca = atoms[protein & (atoms.atom_name == 'CA')]
    require(len(ca) == 120 and np.array_equal(ca.res_id, np.arange(1, 121)), 'Output CA correspondence lost')
    require(np.allclose(ca.coord, expected_ca, atol=.001, rtol=0), 'CIF differs from recorded final state')
    lig = atoms[~protein]
    expected = config['ligands'][ligand]['coordinates']
    require(set(lig.atom_name) == set(expected) and len(lig) == len(expected), 'Output ligand atoms changed')
    reference = np.array([expected[str(name)] for name in lig.atom_name]) - config['origin']
    fixed_drift = float(np.max(np.abs(lig.coord-reference)))
    if fixed_ligand:
        require(fixed_drift < .001, 'Output ligand differs from prepared fixed pose')
    require(np.isfinite(atoms.coord).all(), 'Nonfinite output atoms')
    # Unknown native identities remain in the index; flag them instead of filtering a pair.
    sequence = ''.join(AA.get(str(name), 'X') for name in ca.res_name)
    heavy = atoms[protein & ~np.isin(atoms.element, ['H', 'D']) & (atoms.atom_name != 'VX')]
    distance = np.linalg.norm(heavy.coord[:, None] - lig.coord[None, :], axis=-1)
    radii_p = np.array([RADII.get(str(e), 1.7) for e in heavy.element])
    radii_l = np.array([RADII[str(e)] for e in lig.element])
    contacts = distance < 4.0
    clashes = radii_p[:, None] + radii_l[None, :] - distance > .4
    adjacent = np.linalg.norm(np.diff(ca.coord, axis=0), axis=-1)
    result = dict(sequence=sequence, unknown_residues=sequence.count('X'),
                ligand_minimum_heavy_distance=float(distance.min()),
                ligand_contact_atom_pairs=int(contacts.sum()),
                ligand_contact_residues=len(set(heavy.res_id[contacts.any(axis=1)])),
                ligand_clash_atom_pairs=int(clashes.sum()),
                ligand_fixed_max_coordinate_error=fixed_drift,
                ca_break_count=int(np.sum(np.abs(adjacent-3.8) > .75)),
                ca_adjacent_max_distance=float(adjacent.max()))
    if not fixed_ligand:
        result['ligand_max_coordinate_displacement'] = result.pop('ligand_fixed_max_coordinate_error')
    return result


def analyze(root, output, seeds=None):
    config = read_json(root / 'resolved_config.json')
    seeds = config['seeds'] if seeds is None else seeds
    require(set(seeds) <= set(config['seeds']), 'Unexpected analysis seed')
    directories = {}
    for seed in seeds:
        complete = [p for p in root.glob(f'seed_{seed}_*') if (p/'_COMPLETE').exists()]
        require(len(complete) == 1, f'Seed {seed}: expected exactly one completed attempt, found {len(complete)}')
        directories[seed] = complete[0]
    output.mkdir(parents=True, exist_ok=False)
    pairs, structures, trajectories, noise = [], [], [], []
    grouped = defaultdict(list)
    trajectory_groups = defaultdict(list)
    fasta = []
    for seed, directory in directories.items():
        reference_noise = None
        for percent in config['coupled_percentages']:
            pair_dir = directory / f'coupled_{percent:03d}'
            require((pair_dir / '_COMPLETE').exists(), 'Incomplete condition')
            metadata = read_json(pair_dir / 'pair.json')
            require(metadata['seed'] == seed and metadata['coupled_fraction'] == percent/100, 'Pair metadata mismatch')
            audit = read_json(pair_dir / 'noise_audit.json')
            fingerprints = (audit['initial_shared_sha256'], audit['churn_sha256'])
            if reference_noise is None:
                reference_noise = fingerprints
            require(fingerprints == reference_noise, 'Across-condition noise mismatch')
            require(audit['state_count'] == 200 and len(audit['churn_sha256']) == 199, 'Incomplete noise audit')
            with np.load(pair_dir / 'ca_states.npz') as archive:
                states, sigma = archive['ca'], archive['sigma']
                require(np.array_equal(archive['completed_updates'], np.arange(200)), 'State indices changed')
            require(states.shape == (200, 2, 120, 3) and np.isfinite(states).all(), 'Invalid recorded CA states')
            require(np.all(np.diff(sigma) < 0) and sigma[-1] > 0, 'Expected positive descending schedule')
            raw, aligned = rmsds(states[:, 0], states[:, 1])
            k = metadata['cutoff']['coupled_update_count']
            require(k == percent*199//100 and np.all(raw[:k+1] == 0), 'Incorrect coupling boundary')
            info = dict(seed=seed, condition=f'coupled_{percent:03d}', coupled_fraction=percent/100,
                        coupled_updates=k, release_sigma=metadata['cutoff']['release_pre_churn_sigma'])
            row = dict(info, raw_ca_rmsd=float(raw[-1]), aligned_ca_rmsd=float(aligned[-1]),
                       **topology(states[-1, 0], states[-1, 1]))
            sequences = []
            for track, ligand in enumerate(config['track_ligands'], 1):
                matches = list(pair_dir.glob(f'*track{track}*.cif.gz'))
                require(len(matches) == 1, 'Missing or ambiguous output CIF')
                metrics = structure_metrics(matches[0], states[-1, track-1], config, ligand)
                sequences.append(metrics['sequence'])
                label = f'seed{seed}_coupled{percent:03d}_track{track}_{ligand}'
                fasta.append(f'>{label}\n{metrics["sequence"]}\n')
                structures.append(dict(info, track=track, ligand=ligand,
                                       cif=str(matches[0].relative_to(root)), **metrics))
            row['sequence_identity'] = float(np.mean(np.array(list(sequences[0])) == np.array(list(sequences[1]))))
            row['pair_directory'] = str(pair_dir.relative_to(root))
            pairs.append(row)
            grouped[percent].append(row)
            noise.append(dict(info, initial_sha256=audit['initial_shared_sha256'],
                              maximum_fixed_coordinate_drift=audit['maximum_fixed_coordinate_drift']))
            for step in range(200):
                tr = dict(info, completed_updates=step, denoising_fraction=step/199, sigma=float(sigma[step]),
                          raw_ca_rmsd=float(raw[step]), aligned_ca_rmsd=float(aligned[step]))
                trajectories.append(tr)
                trajectory_groups[(percent, step)].append(tr)
    require(len(pairs) == 21*len(seeds) and len(structures) == 42*len(seeds), 'Wrong output count')
    require(len({row['initial_sha256'] for row in noise}) == len(seeds), 'Distinct seeds share an initialization')
    final_summary = []
    for percent, rows in sorted(grouped.items()):
        require(len({r['release_sigma'] for r in rows}) == 1, 'Release schedule differs across seeds')
        result = {k: rows[0][k] for k in ('coupled_fraction', 'coupled_updates', 'release_sigma')}
        for metric in PAIR_METRICS:
            result.update({f'{metric}_{name}': value for name, value in stats(r[metric] for r in rows).items()})
        final_summary.append(result)
    trajectory_summary = []
    for (percent, step), rows in sorted(trajectory_groups.items()):
        require(len({r['sigma'] for r in rows}) == 1, 'Noise schedule differs across seeds')
        result = {k: rows[0][k] for k in ('coupled_fraction', 'completed_updates', 'denoising_fraction', 'sigma')}
        for metric in ('raw_ca_rmsd', 'aligned_ca_rmsd'):
            result.update({f'{metric}_{name}': value for name, value in stats(r[metric] for r in rows).items()})
        trajectory_summary.append(result)
    for name, rows in [('pairs', pairs), ('designs', structures), ('trajectory', trajectories),
                       ('final_summary', final_summary), ('trajectory_summary', trajectory_summary), ('noise_audit', noise)]:
        write_csv(output / f'{name}.csv', rows)
    (output / 'sequences.fasta').write_text(''.join(fasta))
    write_json(output / 'summary.json', dict(seeds=seeds, pair_count=len(pairs), structure_count=len(structures),
               state_count_per_pair=200, noise_matches_across_conditions=True,
               maximum_fixed_coordinate_drift=max(r['maximum_fixed_coordinate_drift'] for r in noise),
               pairs_all_retained=True, foundry_generation_revision=config['foundry_revision'],
               secondary_structure_method='Biotite annotate_sse; CA geometry (a helix, b strand, c coil)',
               contact_definition='CA distance <8 A, sequence separation >=6',
               clash_definition='Protein-ligand heavy-atom vdW overlap >0.4 A; C1.7 N1.55 O1.52 S/P1.8',
               chain_break_definition='Adjacent CA distance differs from 3.8 A by >0.75 A',
               spread='sample SD across paired seeds; undefined denominators left empty with valid n'))
    (output / '_METRICS_COMPLETE').touch()
    print(f'Analyzed {len(pairs)} pairs / {len(structures)} structures in {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    analyze(args.root, args.output, args.seeds)


if __name__ == '__main__':
    main()
