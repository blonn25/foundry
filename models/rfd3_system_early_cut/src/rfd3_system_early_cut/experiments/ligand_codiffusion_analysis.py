"""Audit free-ligand trajectories and compare paired seeds with fixed ligands."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .substrate_analysis import PAIR_METRICS, stats, structure_metrics
from .substrate_common import read_cif, read_json, require, rmsds, topology, write_csv, write_json

LIGAND_METRICS = ['raw_ligand_rmsd', 'aligned_ligand_rmsd', 'protein_fit_ligand_rmsd']
COMPARISON_METRICS = ['raw_ca_rmsd', 'aligned_ca_rmsd', 'secondary_agreement',
                      'structured_agreement', 'helix_jaccard', 'strand_jaccard', 'contact_jaccard']
FINAL_METRICS = PAIR_METRICS + LIGAND_METRICS + [
    f'{prefix}_{metric}' for metric in COMPARISON_METRICS for prefix in ('fixed', 'delta')]
TRACE_METRICS = ['raw_ca_rmsd', 'aligned_ca_rmsd'] + LIGAND_METRICS + [
    f'{prefix}_{metric}' for metric in ('raw_ca_rmsd', 'aligned_ca_rmsd') for prefix in ('fixed', 'delta')]


def protein_fit_ligand_rmsd(protein_1, protein_2, ligand_1, ligand_2):
    """Fit track 1 CA onto track 2 CA, then measure ligand atoms without refitting."""
    x, y, a, b = [np.asarray(v, dtype=np.float64) for v in (protein_1, protein_2, ligand_1, ligand_2)]
    cx, cy = x.mean(axis=-2, keepdims=True), y.mean(axis=-2, keepdims=True)
    u, _, vh = np.linalg.svd(np.swapaxes(x-cx, -1, -2) @ (y-cy))
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    transformed = (a-cx) @ (u @ vh) + cy
    return np.sqrt(np.mean(np.sum((transformed-b)**2, axis=-1), axis=-1))


def ligand_geometry(coords, reference, bonds):
    """Internal distortion against the supplied conformer, independent of pose."""
    coords, reference = np.asarray(coords), np.asarray(reference)
    bonds = np.asarray(bonds, dtype=int)
    require(bonds.ndim == 2 and bonds.shape[1] >= 2 and len(bonds), 'Missing ligand bond graph')
    i, j = bonds[:, 0], bonds[:, 1]
    error = np.linalg.norm(coords[i]-coords[j], axis=-1) - np.linalg.norm(reference[i]-reference[j], axis=-1)
    return dict(ligand_internal_reference_rmsd=float(rmsds(coords, reference)[1]),
                ligand_bond_length_rmse=float(np.sqrt(np.mean(error**2))),
                ligand_bond_length_max_deviation=float(np.abs(error).max()))


def completed_seed(root, seed):
    matches = [p for p in root.glob(f'seed_{seed}_*') if (p/'_COMPLETE').exists()]
    require(len(matches) == 1, f'{root}, seed {seed}: expected exactly one complete attempt')
    return matches[0]


def summarize(groups, fields, metrics):
    result = []
    for _, rows in sorted(groups.items()):
        item = {key: rows[0][key] for key in fields}
        for key in fields:
            require(len({row[key] for row in rows}) == 1, f'Inconsistent summary axis {key}')
        for metric in metrics:
            item.update({f'{metric}_{key}': value for key, value in stats(row[metric] for row in rows).items()})
        result.append(item)
    return result


def analyze(project, root, output, seeds=None):
    config = read_json(root/'resolved_config.json')
    seeds = config['seeds'] if seeds is None else seeds
    require(len(set(seeds)) == len(seeds) and set(seeds) <= set(config['seeds']), 'Unexpected or repeated seed')
    baseline = project/config['fixed_baseline']
    fixed_config = read_json(baseline/'resolved_config.json')
    for key in ('seeds', 'coupled_percentages', 'sampler', 'protein_length', 'ligands', 'origin'):
        require(config[key] == fixed_config[key], f'Fixed-ligand baseline differs: {key}')
    directories = {seed: completed_seed(root, seed) for seed in seeds}
    fixed_dirs = {seed: completed_seed(baseline, seed) for seed in seeds}
    references = {name: read_cif(project/config['input_dir']/f'{name}.cif') for name in config['track_ligands']}
    output.mkdir(parents=True, exist_ok=False)
    pairs, designs, traces, noises, fasta = [], [], [], [], []
    grouped, trace_groups = defaultdict(list), defaultdict(list)
    for seed, directory in directories.items():
        reference_noise = None
        for percent in config['coupled_percentages']:
            condition = f'coupled_{percent:03d}'
            pair_dir, fixed_dir = directory/condition, fixed_dirs[seed]/condition
            require((pair_dir/'_COMPLETE').exists() and (fixed_dir/'_COMPLETE').exists(), 'Incomplete pair')
            meta, audit = read_json(pair_dir/'pair.json'), read_json(pair_dir/'noise_audit.json')
            require(meta['seed'] == seed and meta['coupled_fraction'] == percent/100, 'Pair metadata mismatch')
            fingerprint = [audit[key] for key in ('initial_shared_sha256', 'churn_sha256',
                           'ligand_initial_sha256', 'ligand_churn_sha256')]
            if reference_noise is None:
                reference_noise = fingerprint
            require(fingerprint == reference_noise, 'Noise differs across conditions')
            fixed_audit = read_json(fixed_dir/'noise_audit.json')
            require(all(audit[k] == fixed_audit[k] for k in ('initial_shared_sha256', 'churn_sha256')),
                    'Protein noise differs from fixed-ligand baseline')
            require(audit['state_count'] == 200 and len(audit['churn_sha256']) == 199 and
                    len(audit['ligand_churn_sha256']) == 199, 'Incomplete noise audit')
            with np.load(pair_dir/'ca_states.npz') as archive:
                ca, sigma = archive['ca'], archive['sigma']
                require(np.array_equal(archive['completed_updates'], np.arange(200)), 'State indices changed')
            with np.load(fixed_dir/'ca_states.npz') as archive:
                fixed_ca = archive['ca']
                require(np.array_equal(sigma, archive['sigma']), 'Baseline schedule differs')
            with np.load(pair_dir/'ligand_states.npz') as archive:
                ligand = archive['mapped']
                all_ligands = [archive[f'track_{t}'] for t in (1, 2)]
            require(ca.shape == fixed_ca.shape == (200, 2, 120, 3) and ligand.shape == (200, 2, 16, 3),
                    'Invalid state dimensions')
            require(all(np.isfinite(a).all() for a in [ca, fixed_ca, ligand, *all_ligands]), 'Nonfinite states')
            require(np.all(np.diff(sigma) < 0) and sigma[-1] > 0, 'Invalid schedule')
            raw, aligned = rmsds(ca[:, 0], ca[:, 1])
            lig_raw, lig_aligned = rmsds(ligand[:, 0], ligand[:, 1])
            lig_protein_fit = protein_fit_ligand_rmsd(ca[:, 0], ca[:, 1], ligand[:, 0], ligand[:, 1])
            fixed_raw, fixed_aligned = rmsds(fixed_ca[:, 0], fixed_ca[:, 1])
            k = meta['cutoff']['coupled_update_count']
            require(k == percent*199//100 and np.all(raw[:k+1] == 0) and np.all(lig_raw[:k+1] == 0),
                    'Protein or ligand diverged while coupled')
            info = dict(seed=seed, condition=condition, coupled_fraction=percent/100,
                        coupled_updates=k, release_sigma=meta['cutoff']['release_pre_churn_sigma'])
            row = dict(info, raw_ca_rmsd=float(raw[-1]), aligned_ca_rmsd=float(aligned[-1]),
                       raw_ligand_rmsd=float(lig_raw[-1]), aligned_ligand_rmsd=float(lig_aligned[-1]),
                       protein_fit_ligand_rmsd=float(lig_protein_fit[-1]), **topology(ca[-1, 0], ca[-1, 1]))
            fixed_metrics = dict(raw_ca_rmsd=float(fixed_raw[-1]), aligned_ca_rmsd=float(fixed_aligned[-1]),
                                 **topology(fixed_ca[-1, 0], fixed_ca[-1, 1]))
            for metric in COMPARISON_METRICS:
                row[f'fixed_{metric}'] = fixed_metrics[metric]
                row[f'delta_{metric}'] = (row[metric]-fixed_metrics[metric]
                    if row[metric] is not None and fixed_metrics[metric] is not None else None)
            sequences = []
            atom_names = read_json(pair_dir/'ligand_atom_names.json')
            for index, name in enumerate(config['track_ligands']):
                track = index+1
                matches = list(pair_dir.glob(f'*track{track}*.cif.gz'))
                require(len(matches) == 1, 'Missing or ambiguous final structure')
                metrics = structure_metrics(matches[0], ca[-1, index], config, name, fixed_ligand=False)
                atoms = read_cif(matches[0])
                final_ligand = atoms[atoms.chain_id != 'A']
                lookup = {str(a.atom_name): a.coord for a in final_ligand}
                final_coords = np.array([lookup[a] for a in atom_names[str(track)]])
                require(np.allclose(final_coords, all_ligands[index][-1], rtol=0, atol=.001),
                        'Final ligand differs from recorded state')
                mapped_final = np.array([lookup[pair[f'track_{track}']['atom']]
                                         for pair in config['coupled_ligand_atom_pairs']])
                require(np.allclose(mapped_final, ligand[-1, index], rtol=0, atol=.001), 'Final ligand mapping changed')
                reference = references[name]
                coords_in_reference_order = np.array([lookup[str(a)] for a in reference.atom_name])
                metrics.update(ligand_geometry(coords_in_reference_order, reference.coord, reference.bonds.as_array()))
                metrics['ligand_protein_centroid_distance'] = float(np.linalg.norm(final_coords.mean(0)-ca[-1, index].mean(0)))
                sequences.append(metrics['sequence'])
                designs.append(dict(info, track=track, ligand=name, cif=str(matches[0].relative_to(root)), **metrics))
                fasta.append(f'>seed{seed}_{condition}_track{track}_{name}\n{metrics["sequence"]}\n')
            row['sequence_identity'] = float(np.mean(np.array(list(sequences[0])) == np.array(list(sequences[1]))))
            row['pair_directory'] = str(pair_dir.relative_to(root))
            pairs.append(row)
            grouped[percent].append(row)
            noises.append(dict(info, initial_sha256=audit['initial_shared_sha256'],
                               protein_noise_matches_fixed_baseline=True, mapped_ligand_noise_identical=True))
            for step in range(200):
                item = dict(info, completed_updates=step, denoising_fraction=step/199, sigma=float(sigma[step]),
                            raw_ca_rmsd=float(raw[step]), aligned_ca_rmsd=float(aligned[step]),
                            raw_ligand_rmsd=float(lig_raw[step]), aligned_ligand_rmsd=float(lig_aligned[step]),
                            protein_fit_ligand_rmsd=float(lig_protein_fit[step]),
                            fixed_raw_ca_rmsd=float(fixed_raw[step]), fixed_aligned_ca_rmsd=float(fixed_aligned[step]),
                            delta_raw_ca_rmsd=float(raw[step]-fixed_raw[step]),
                            delta_aligned_ca_rmsd=float(aligned[step]-fixed_aligned[step]))
                traces.append(item)
                trace_groups[(percent, step)].append(item)
    require(len(pairs) == 21*len(seeds) and len(designs) == 42*len(seeds), 'Wrong output count')
    require(len({r['initial_sha256'] for r in noises}) == len(seeds), 'Distinct seeds share an initialization')
    finals = summarize(grouped, ['coupled_fraction', 'coupled_updates', 'release_sigma'], FINAL_METRICS)
    trace_summary = summarize(trace_groups, ['coupled_fraction', 'completed_updates', 'denoising_fraction', 'sigma'], TRACE_METRICS)
    for name, rows in [('pairs', pairs), ('designs', designs), ('trajectory', traces), ('noise_audit', noises),
                       ('final_summary', finals), ('trajectory_summary', trace_summary)]:
        write_csv(output/f'{name}.csv', rows)
    (output/'sequences.fasta').write_text(''.join(fasta))
    write_json(output/'summary.json', dict(seeds=seeds, pair_count=len(pairs), structure_count=len(designs),
        pairs_all_retained=True, state_count_per_pair=200, mapped_ligand_atom_count=16,
        noise_matches_across_conditions=True, protein_noise_matches_fixed_baseline=True,
        fixed_baseline=config['fixed_baseline'], foundry_generation_revision=config['foundry_revision'],
        secondary_structure_method='Biotite annotate_sse, CA geometry; no sequence or H-bond input',
        topology_definition='CA contact distance <8 A with sequence separation >=6; position-label agreements/Jaccards',
        geometry_definition='Protein-ligand vdW overlap >0.4 A; adjacent CA spacing outside 3.8 +/- 0.75 A',
        ligand_metrics='Explicit mapped atoms: common frame, ligand Kabsch, and protein CA fit without ligand refitting',
        ligand_reference_metrics='Whole-ligand Kabsch RMSD and bond-length deviations from input conformer; not ideal bond lengths',
        delta_definition='Co-diffusing minus fixed-ligand measurement for the same seed/fraction, then mean and sample SD',
        spread='Sample SD; undefined topology denominators omitted with valid n; no geometry filtering',
        interpretation='Changes both ligand mobility and mapped-atom coupling relative to fixed-ligand baseline'))
    (output/'_METRICS_COMPLETE').touch()
    print(f'Analyzed {len(pairs)} pairs / {len(designs)} structures in {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/project'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    analyze(args.project, args.root, args.output, args.seeds)


if __name__ == '__main__':
    main()
