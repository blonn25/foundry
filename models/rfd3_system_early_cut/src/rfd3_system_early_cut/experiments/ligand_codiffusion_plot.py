"""Protein/ligand dynamics and paired fixed-ligand comparisons; CSV-only plotting."""
import argparse
import csv
from pathlib import Path

import substrate_plot as base
import matplotlib.pyplot as plt
import numpy as np

LIGAND_LABELS = {
    'raw_ligand_rmsd': 'Common-frame mapped ligand RMSD (Å)',
    'aligned_ligand_rmsd': 'Ligand-fitted mapped RMSD (Å)',
    'protein_fit_ligand_rmsd': 'Mapped ligand RMSD after CA fit (Å)',
}


def trajectory_panels(rows, finals, output, sparse, sigma_axis, *, differences=False):
    labels = ({f'delta_{key}': f'Co-diffused − fixed: {label}' for key, label in base.LABELS.items()}
              if differences else LIGAND_LABELS)
    fig, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 4.8), layout='constrained')
    selected = sorted({row['coupled_fraction'] for row in rows} &
                      (base.SPARSE if sparse else {row['coupled_fraction'] for row in rows}))
    lookup = {r['coupled_fraction']: r for r in finals}
    field = 'sigma' if sigma_axis else 'denoising_fraction'
    for fraction in selected:
        subset = sorted([r for r in rows if r['coupled_fraction'] == fraction], key=lambda r: r['completed_updates'])
        x = np.array([r[field] for r in subset])
        for ax, (metric, label) in zip(axes, labels.items()):
            mean, sd = [np.array([r[f'{metric}_{stat}'] for r in subset]) for stat in ('mean', 'sd')]
            ax.plot(x, mean, color=base.CMAP(fraction), lw=1.8 if sparse else 1.2)
            if sparse:
                lower = mean-sd if differences else np.maximum(0, mean-sd)
                ax.fill_between(x, lower, mean+sd, color=base.CMAP(fraction), alpha=.12, linewidth=0)
                k = int(lookup[fraction]['coupled_updates'])
                if k < 199:
                    ax.scatter(x[k], mean[k], color=base.CMAP(fraction), marker='|', s=100)
            ax.set_ylabel(label)
    for ax in axes:
        if differences:
            ax.axhline(0, color='.4', lw=.7)
        else:
            ax.set_ylim(bottom=0)
        if sigma_axis:
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_xlabel('State sigma (Å; denoising →)')
        else:
            ax.set(xlim=(0, 1), xlabel='Completed denoising updates / 199')
    kind = 'paired_delta' if differences else 'ligand_trajectory'
    title = 'Paired change from fixed-ligand baseline' if differences else 'Divergence of 16 explicitly mapped ligand atoms'
    fig.suptitle(f'{title} · {int(rows[0]["raw_ca_rmsd_n"])} matched seeds · {len(selected)} conditions')
    fig.supxlabel('Per-pair measurements before seed averaging; sparse bands show sample SD; ticks mark release.', fontsize=9)
    base.color_key(fig, axes)
    base.save(fig, output, f'{kind}_{"sigma" if sigma_axis else "fraction"}_{"sparse" if sparse else "dense"}')


def final_comparison(rows, output, sparse):
    rows = [r for r in rows if not sparse or r['coupled_fraction'] in base.SPARSE]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    labels = base.LABELS | {'secondary_agreement': 'Same secondary label fraction',
                          'contact_jaccard': 'Long-range CA contact Jaccard'}
    x = np.array([r['coupled_fraction'] for r in rows])
    for ax, (metric, label) in zip(axes.flat, labels.items()):
        for prefix, title, color, style in [('', 'Co-diffused, mapped coupling', '#176ca4', '-'),
                                           ('fixed_', 'Fixed ligands', '#b06c24', '--')]:
            mean, sd = [np.array([r[f'{prefix}{metric}_{stat}'] for r in rows]) for stat in ('mean', 'sd')]
            ax.plot(x, mean, style, marker='o', ms=3, color=color, label=title)
            ax.fill_between(x, np.maximum(0, mean-sd), mean+sd, color=color, alpha=.12)
        ax.set(xlabel='Fraction of updates coupled', ylabel=label, xlim=(-.03, 1.03))
        ax.set_ylim(bottom=0)
        if metric in ('secondary_agreement', 'contact_jaccard'):
            ax.set_ylim(-.03, 1.03)
        ax.legend(fontsize=8)
    fig.suptitle(f'Final paired comparison · {int(rows[0]["raw_ca_rmsd_n"])} seeds · {len(rows)} release conditions')
    fig.supxlabel('Lines: means; bands: sample SD. Protein initialization/churn match exactly; all completed pairs retained.', fontsize=9)
    base.save(fig, output, f'fixed_vs_codiffused_{"sparse" if sparse else "dense"}')


def geometry(analysis, output):
    with (analysis/'designs.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    fields = {'ligand_internal_reference_rmsd': 'Whole-ligand reference-fitted RMSD (Å)',
              'ligand_bond_length_rmse': 'Bond-length RMSE versus input (Å)',
              'ligand_clash_atom_pairs': 'Protein–ligand clash atom pairs',
              'ligand_protein_centroid_distance': 'Ligand / protein CA centroid distance (Å)'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for track, color, label in [('1', '#ba5721', '4MU-Ac'), ('2', '#267dac', '4MU-Bu')]:
        fractions = sorted({float(r['coupled_fraction']) for r in rows})
        for ax, (metric, ylabel) in zip(axes.flat, fields.items()):
            samples = [np.array([float(r[metric]) for r in rows if r['track'] == track and
                                float(r['coupled_fraction']) == fraction]) for fraction in fractions]
            mean = np.array([a.mean() for a in samples])
            sd = np.array([a.std(ddof=1) if len(a) > 1 else np.nan for a in samples])
            ax.plot(fractions, mean, color=color, marker='o', ms=3, label=label)
            ax.fill_between(fractions, np.maximum(0, mean-sd), mean+sd, color=color, alpha=.12)
            ax.set(xlabel='Fraction of updates coupled', ylabel=ylabel)
            ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.legend()
    fig.suptitle('Final co-diffused ligand geometry · all 21 release conditions')
    fig.supxlabel('Mean ± sample SD. Input conformer is the bond-length reference; centroid distance is not a burial metric.', fontsize=9)
    base.save(fig, output, 'ligand_geometry')


def plot(analysis):
    if not (analysis/'_METRICS_COMPLETE').exists():
        raise ValueError('Metrics are incomplete')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42})
    rows, finals = base.load(analysis/'trajectory_summary.csv'), base.load(analysis/'final_summary.csv')
    for sparse in (False, True):
        for sigma in (False, True):
            base.trajectories(rows, finals, analysis, sparse, sigma)
            trajectory_panels(rows, finals, analysis, sparse, sigma)
            trajectory_panels(rows, finals, analysis, sparse, sigma, differences=True)
        base.endpoints(finals, analysis, sparse)
        base.topologies(finals, analysis, sparse)
        final_comparison(finals, analysis, sparse)
    geometry(analysis, analysis)
    (analysis/'_COMPLETE').touch()
    print(f'Saved 19 PNG/PDF figure pairs in {analysis}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    plot(parser.parse_args().analysis)
