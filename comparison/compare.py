"""Intuitive metrics of a recording, for the real data, the simulated file and many simulated recordings.
python comparison/compare.py  writes comparison/results.csv and figures/comparison.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spikesim import detect_spikes, load, simulate  # noqa: E402


def clusters(idx, fs, gap=0.05):
    # groups of 2 or more spikes, each one less than `gap` seconds after the previous one
    groups = np.split(idx, np.flatnonzero(np.diff(idx) / fs >= gap) + 1)
    return [g for g in groups if len(g) >= 2]


def most_in_window(idx, fs, k):
    # largest number of spikes inside any window of k seconds
    t = idx / fs
    return int((np.searchsorted(t, t + k) - np.arange(len(t))).max())


def metrics(t, x):
    fs = 1 / np.median(np.diff(t))
    idx, r = detect_spikes(x)
    peaks = r[idx]
    baseline = x - r
    groups = clusters(idx, fs)
    lag = np.arange(7)
    avg = np.mean([r[i + lag] for i in idx if i + 7 <= len(x)], axis=0)
    return {
        'spikes': len(idx),
        'mean spike height': peaks.mean(),
        'smallest spike height': peaks.min(),
        'largest spike height': peaks.max(),
        'signal min': x.min(),
        'signal max': x.max(),
        'noise sd': np.median(np.abs(r - np.median(r))) / 0.6745,
        'drift range': baseline.max() - baseline.min(),
        'spike decay (ms)': -1000 / fs / np.polyfit(lag, np.log(avg), 1)[0],
        'shortest gap between spikes (ms)': np.diff(idx).min() / fs * 1000,
        'most spikes in 100 ms': most_in_window(idx, fs, 0.1),
        'most spikes in 1 s': most_in_window(idx, fs, 1.0),
        'clusters (spikes < 50 ms apart)': len(groups),
        'biggest cluster': max((len(g) for g in groups), default=1),
    }


def compare(n_sim=500):
    real = metrics(*load(ROOT / 'spike_data.npz'))
    sim_file = metrics(*load(ROOT / 'simulated_spike_data.npz'))
    sims = pd.DataFrame([metrics(*simulate(10, seed=10000 + i)[:2]) for i in range(n_sim)])
    table = pd.DataFrame({'real': real, 'simulated file': sim_file, 'simulated mean': sims.mean(),
                          'low (2.5%)': sims.quantile(0.025, interpolation='lower'),
                          'high (97.5%)': sims.quantile(0.975, interpolation='higher')})
    table['real inside'] = (table['real'] >= table['low (2.5%)']) & (table['real'] <= table['high (97.5%)'])
    return table, sims


if __name__ == '__main__':
    table, sims = compare()
    table.round(4).to_csv(ROOT / 'comparison' / 'results.csv')
    print(table.round(3).to_string())

    fig, axes = plt.subplots(3, 5, figsize=(17, 8.5))
    for ax, name in zip(axes.ravel(), table.index):
        values = sims[name]
        ints = np.allclose(values, values.round())
        bins = np.arange(values.min(), values.max() + 2) - 0.5 if ints else 25
        ax.hist(values, bins=bins, color='C2', alpha=0.6)
        ax.axvline(table.loc[name, 'real'], color='C0', lw=3)
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color='C2', alpha=0.6), plt.Line2D([], [], color='C0', lw=3)],
                            labels=['500 simulated recordings', 'real recording'], loc='center', fontsize=12, frameon=False)
    plt.tight_layout()
    fig.savefig(ROOT / 'figures' / 'comparison.png', dpi=130, bbox_inches='tight')
