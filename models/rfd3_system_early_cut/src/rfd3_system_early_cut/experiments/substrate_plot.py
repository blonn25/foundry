"""Plot the structural release sweep from analysis CSVs (no model imports)."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

SPARSE = {0., .2, .4, .6, .8, 1.}
CMAP = plt.get_cmap('viridis')
LABELS = {'raw_ca_rmsd': 'Common-frame Cα RMSD (Å)', 'aligned_ca_rmsd': 'Aligned Cα RMSD (Å)'}


def load(path):
    with path.open() as handle:
        return [{k: float(v) if v else np.nan for k, v in row.items()} for row in csv.DictReader(handle)]


def save(fig, output, name):
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def color_key(fig, axes):
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=axes, pad=.02, shrink=.88)
    bar.set_label('Fraction of updates coupled (0 = none; 1 = full)')


def trajectories(rows, finals, output, sparse, sigma_axis):
    selected = sorted({r['coupled_fraction'] for r in rows} & (SPARSE if sparse else {r['coupled_fraction'] for r in rows}))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    lookup = {r['coupled_fraction']: r for r in finals}
    xfield = 'sigma' if sigma_axis else 'denoising_fraction'
    for fraction in selected:
        subset = sorted([r for r in rows if r['coupled_fraction'] == fraction], key=lambda r: r['completed_updates'])
        x = np.array([r[xfield] for r in subset])
        for ax, metric in zip(axes, LABELS):
            mean = np.array([r[f'{metric}_mean'] for r in subset])
            sd = np.array([r[f'{metric}_sd'] for r in subset])
            ax.plot(x, mean, color=CMAP(fraction), lw=1.8 if sparse else 1.2, label=f'{fraction:.2f}')
            if sparse:
                ax.fill_between(x, np.maximum(0, mean-sd), mean+sd, color=CMAP(fraction), alpha=.12, linewidth=0)
                k = int(lookup[fraction]['coupled_updates'])
                if k < 199:
                    ax.scatter(x[k], mean[k], color=CMAP(fraction), marker='|', s=110, zorder=4)
            ax.set_ylabel(LABELS[metric])
    for ax in axes:
        ax.set_ylim(bottom=0)
        if sigma_axis:
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_xlabel('State sigma (Å; denoising →)')
        else:
            ax.set_xlim(0, 1)
            ax.set_xlabel('Completed denoising updates / 199')
    n = int(rows[0]['raw_ca_rmsd_n'])
    fig.suptitle(f'4MU-Ac versus 4MU-Bu · {n} matched seeds · ' + ('6' if sparse else '21') + ' release conditions')
    color_key(fig, axes)
    if sparse:
        fig.supxlabel('Lines: mean paired RMSD; bands: sample SD. Small ticks mark release; full coupling never releases.', fontsize=9)
    else:
        fig.supxlabel('Each line averages paired RMSDs across seeds; no structures excluded for geometry.', fontsize=9)
    save(fig, output, f'trajectory_{"sigma" if sigma_axis else "fraction"}_{"sparse" if sparse else "dense"}')


def endpoints(rows, output, sparse):
    rows = [r for r in rows if not sparse or r['coupled_fraction'] in SPARSE]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), width_ratios=[1, 1, .26], layout='constrained')
    for i, metric in enumerate(LABELS):
        for j in (0, 1):
            ax = axes[i, j]
            subset = rows if j == 0 else [r for r in rows if np.isfinite(r['release_sigma'])]
            field = 'coupled_fraction' if j == 0 else 'release_sigma'
            x = [r[field] for r in subset]
            y = [r[f'{metric}_mean'] for r in subset]
            ax.plot(x, y, color='.65', lw=1, zorder=1)
            for r in subset:
                ax.errorbar(r[field], r[f'{metric}_mean'], yerr=r[f'{metric}_sd'] if np.isfinite(r[f'{metric}_sd']) else None,
                            fmt='o', color=CMAP(r['coupled_fraction']), capsize=3, markersize=5)
            if j == 1:
                ax.set_xscale('log')
                ax.invert_xaxis()
                ax.set_xlabel('Release pre-churn sigma (Å; later release →)')
            else:
                ax.set_xlabel('Fraction of updates coupled')
                ax.set_xlim(-.04, 1.04)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(LABELS[metric])
        control = next(r for r in rows if r['coupled_fraction'] == 1)
        ax = axes[i, 2]
        ax.scatter([0], [control[f'{metric}_mean']], color=CMAP(1.), s=45)
        ax.set_xlim(-.5, .5)
        ax.set_ylim(axes[i, 1].get_ylim())
        ax.set_xticks([0], ['Never\nreleased'])
        ax.set_yticklabels([])
        ax.set_title('Full coupling', fontsize=10)
    fig.suptitle('Final backbone divergence versus release timing · ' + ('6 conditions' if sparse else 'all 21 conditions'))
    fig.supxlabel('Mean ± sample SD across matched seeds. The never-released control has no release sigma.', fontsize=9)
    save(fig, output, f'final_rmsd_{"sparse" if sparse else "dense"}')


def topologies(rows, output, sparse):
    rows = [r for r in rows if not sparse or r['coupled_fraction'] in SPARSE]
    x = np.array([r['coupled_fraction'] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    panels = [
        [('ac_helix_fraction', 'Ac helix', '#c15b22', '-'), ('bu_helix_fraction', 'Bu helix', '#c15b22', '--'),
         ('ac_strand_fraction', 'Ac strand', '#327eac', '-'), ('bu_strand_fraction', 'Bu strand', '#327eac', '--')],
        [('secondary_agreement', 'All labels', '#474747', '-'), ('structured_agreement', 'Structured positions', '#8055a1', '-')],
        [('helix_jaccard', 'Helix positions', '#c15b22', '-'), ('strand_jaccard', 'Strand positions', '#327eac', '-')],
        [('contact_jaccard', 'Long-range CA contacts', '#257d61', '-')],
    ]
    for ax, metrics, title in zip(axes.flat, panels, ['Per-track secondary structure', 'Same-label fraction', 'Secondary-structure Jaccard', 'Tertiary contact-map Jaccard']):
        for metric, label, color, style in metrics:
            y, sd = [np.array([r[f'{metric}_{stat}'] for r in rows]) for stat in ('mean', 'sd')]
            ax.plot(x, y, linestyle=style, marker='o', ms=3, color=color, label=label)
            ax.fill_between(x, np.maximum(0, y-sd), np.minimum(1, y+sd), color=color, alpha=.07)
        ax.set(title=title, xlabel='Fraction of updates coupled', ylabel='Fraction', ylim=(-.03, 1.03), xlim=(-.03, 1.03))
        ax.legend(fontsize=9, loc='best')
    fig.suptitle('Shared secondary structure and contact topology · ' + ('6 conditions' if sparse else 'all 21 conditions'))
    fig.supxlabel('CA geometry assignments; mean ± sample SD. Undefined Jaccards are omitted with valid counts retained in CSV.', fontsize=9)
    save(fig, output, f'topology_{"sparse" if sparse else "dense"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    args = parser.parse_args()
    if not (args.analysis / '_METRICS_COMPLETE').exists():
        raise ValueError('Metrics are incomplete')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42})
    rows = load(args.analysis / 'trajectory_summary.csv')
    finals = load(args.analysis / 'final_summary.csv')
    for sparse in (False, True):
        for sigma in (False, True):
            trajectories(rows, finals, args.analysis, sparse, sigma)
        endpoints(finals, args.analysis, sparse)
        topologies(finals, args.analysis, sparse)
    (args.analysis / '_COMPLETE').touch()
    print(f'Saved eight PNG/PDF figure pairs in {args.analysis}')


if __name__ == '__main__':
    main()
