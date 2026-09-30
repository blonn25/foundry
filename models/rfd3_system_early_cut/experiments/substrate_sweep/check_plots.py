"""Exercise all plotting paths with clearly synthetic data; never scientific results."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

MODEL = Path(__file__).resolve().parents[2]
PLOTTER = MODEL / 'src/rfd3_system_early_cut/experiments/substrate_plot.py'


def write(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


with tempfile.TemporaryDirectory(prefix='synthetic_plot_check_') as tmp:
    root = Path(tmp)
    states, finals = [], []
    for percent in range(0, 101, 5):
        fraction = percent/100
        cutoff = percent*199//100
        final = dict(coupled_fraction=fraction, coupled_updates=cutoff,
                     release_sigma=100*(1-fraction)+.01 if percent < 100 else None)
        for key in ('raw_ca_rmsd', 'aligned_ca_rmsd', 'ac_helix_fraction', 'bu_helix_fraction',
                    'ac_strand_fraction', 'bu_strand_fraction', 'secondary_agreement',
                    'structured_agreement', 'helix_jaccard', 'strand_jaccard', 'contact_jaccard'):
            # One absent category exercises NA handling and its valid-count contract.
            value = None if key == 'strand_jaccard' else (1-fraction)*.5
            final.update({f'{key}_mean': value, f'{key}_sd': None if value is None else .1,
                          f'{key}_n': 0 if value is None else 10})
        finals.append(final)
        for step in range(200):
            value = max(0, step-cutoff)/199
            states.append(dict(coupled_fraction=fraction, completed_updates=step,
                               denoising_fraction=step/199, sigma=100*(1-step/200)+.01,
                               raw_ca_rmsd_mean=value, raw_ca_rmsd_sd=.1, raw_ca_rmsd_n=10,
                               aligned_ca_rmsd_mean=value/2, aligned_ca_rmsd_sd=.05, aligned_ca_rmsd_n=10))
    write(root/'final_summary.csv', finals)
    write(root/'trajectory_summary.csv', states)
    (root/'_METRICS_COMPLETE').touch()
    subprocess.run([sys.executable, str(PLOTTER), str(root)], check=True)
    for ext in ('png', 'pdf'):
        files = list(root.glob(f'*.{ext}'))
        assert len(files) == 8 and all(p.stat().st_size > 1000 for p in files)
    assert (root/'_COMPLETE').exists()
    print('Synthetic plot validation passed: all eight PNG/PDF pairs, including undefined metrics.')
