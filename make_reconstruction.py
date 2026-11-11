"""Part 1 of the inter-team evaluation: writes team_<ID>_reconstruction.npz from spike_data.npz.
python make_reconstruction.py [--team ID]"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from spikesim import detect_spikes, drift, estimate, load, noise

ROOT = Path(__file__).resolve().parent
TEAM_ID = 'OAM'   # change to the ID the class gave us
SEED = 42


def spike_kernel(p):
    k = np.arange(int(p.kernel_len * p.fs)) / p.fs
    return np.exp(-k / p.spike_tau)


def fit_spike_signal(x, base, idx, p):
    # one amplitude per spike, all fitted together (least squares), so overlapping spikes are handled
    h = spike_kernel(p)
    cols = np.zeros((len(x), len(idx)))
    for j, i in enumerate(idx):
        m = min(len(h), len(x) - i)
        cols[i:i + m, j] = h[:m]
    amps = np.linalg.lstsq(cols, x - base, rcond=None)[0]
    return cols @ amps, amps


def reconstruct(path=ROOT / 'spike_data.npz', seed=SEED):
    t, x = load(path)
    p = estimate(t, x)
    idx, _ = detect_spikes(x)            # one-sample jump, so the peak sample is the spike start
    d = drift(t, p)
    spikes, amps = fit_spike_signal(x, d, idx, p)
    # drift once more, now with the fitted spikes removed
    p = replace(p, **_refit_drift(t, x - spikes, p))
    d = drift(t, p)
    spikes, amps = fit_spike_signal(x, d, idx, p)
    nz = noise(len(x), p, np.random.default_rng(seed))
    return dict(t=t, drift=d, spike_signal=spikes, noise=nz,
                reconstruction=d + spikes,  # structured signal only, as the instructions ask
                spike_indices=idx.astype(np.int64)), p, amps


def _refit_drift(t, y, p):
    from scipy.optimize import curve_fit
    sine = lambda t, c, s, co: c + s * np.sin(2 * np.pi * p.drift_freq * t) + co * np.cos(2 * np.pi * p.drift_freq * t)
    sine_f = lambda t, c, s, co, f: c + s * np.sin(2 * np.pi * f * t) + co * np.cos(2 * np.pi * f * t)
    s0 = p.drift_amp * np.cos(p.drift_phase)
    c0 = p.drift_amp * np.sin(p.drift_phase)
    (c, s, co, f), _ = curve_fit(sine_f, t, y, p0=(p.offset, s0, c0, p.drift_freq))
    return dict(offset=c, drift_amp=np.hypot(s, co), drift_freq=abs(f),
                drift_phase=np.arctan2(co, s) if f > 0 else np.arctan2(co, -s),
                noise_sd=np.std(y - sine_f(t, c, s, co, f)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--team', default=TEAM_ID)
    a = ap.parse_args()
    out, p, amps = reconstruct()
    path = ROOT / f'team_{a.team}_reconstruction.npz'
    np.savez(path, **out)
    with np.load(path) as f:
        for k in f.files:
            print(f'{k:15s} {f[k].shape} {f[k].dtype}')
    res = load(ROOT / 'spike_data.npz')[1] - out['reconstruction']
    print(f'spikes {len(out["spike_indices"])}, amplitude {amps.mean():.3f} +- {amps.std():.3f}, '
          f'tau {p.spike_tau * 1000:.2f} ms, drift {p.offset:.4f} + {p.drift_amp:.4f} sin(2pi {p.drift_freq:.5f} t + {p.drift_phase:.3f})')
    print(f'residual sd {res.std():.4f} (noise sd {p.noise_sd:.4f}), R2 {1 - res.var() / load(ROOT / "spike_data.npz")[1].var():.4f}')
    print('saved', path.name)
