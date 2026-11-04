"""Model of spike_data.npz: generator (simulate) and estimation (estimate)."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, minimize


@dataclass
class Params:
    fs: float = 1000.0           # sample rate (Hz)
    offset: float = 0.0043       # drift = offset + drift_amp * sin(2 pi drift_freq t + drift_phase)
    drift_amp: float = 0.6468
    drift_freq: float = 0.09076  # Hz
    drift_phase: float = 0.2930  # rad
    noise_sd: float = 0.2985     # white gaussian noise
    rate: float = 4.2            # mean spikes per second
    refractory: float = 0.010    # s, shortest time between two spikes
    excite: float = 0.138        # extra spikes that one spike causes on average (0 = independent poisson)
    excite_tau: float = 0.0270   # s, how long this extra chance lasts after a spike
    spike_amp: float = 4.990     # jump of a spike
    spike_amp_sd: float = 0.077  # spike to spike spread of the jump
    spike_tau: float = 0.002944  # s, decay of a spike
    spike_tau_sd: float = 0.0    # spike to spike spread of tau
    kernel_len: float = 0.030    # s, length of one spike


def load(path):
    with np.load(path) as d:
        return d['t'], d['raw_data']


def drift(t, p):
    return p.offset + p.drift_amp * np.sin(2 * np.pi * p.drift_freq * t + p.drift_phase)


def noise(n, p, rng):
    return rng.normal(0, p.noise_sd, n)


def spike_times(duration, p, rng):
    # background spikes are poisson; every spike gets on average `excite` extra spikes, exp(excite_tau) later.
    # the clock of the neuron stops during the refractory time, so the mean rate is exactly p.rate
    assert 0 <= p.excite < 1 and p.rate * p.refractory < 1, 'need excite < 1 and refractory < 1 / rate'
    mu = p.rate * (1 - p.excite) / (1 - p.rate * p.refractory)
    gen = rng.uniform(0, duration, rng.poisson(mu * duration))
    live = [gen]
    while len(gen):
        kids = np.repeat(gen, rng.poisson(p.excite, len(gen)))
        gen = kids + rng.exponential(p.excite_tau, len(kids))
        gen = gen[gen < duration]
        live.append(gen)
    live = np.sort(np.concatenate(live))
    times = live + p.refractory * np.arange(len(live))
    return (times[times < duration] * p.fs).astype(int)


def spike_train(idx, n, p, rng):
    k = np.arange(int(p.kernel_len * p.fs)) / p.fs
    amps = p.spike_amp + p.spike_amp_sd * rng.standard_normal(len(idx))
    taus = np.maximum(p.spike_tau + p.spike_tau_sd * rng.standard_normal(len(idx)), 1e-4)
    s = np.zeros(n)
    for i, a, tau in zip(idx, amps, taus):
        m = min(len(k), n - i)
        s[i:i + m] += a * np.exp(-k[:m] / tau)
    return s


def simulate(duration=10.0, rate=None, seed=None, p=None, **changes):
    """Returns t, x and the true spike indices. rate or any other coefficient can be changed."""
    p = replace(p or Params(), **changes)
    if rate is not None:
        p = replace(p, rate=rate)
    rng = np.random.default_rng(seed)
    n = int(round(duration * p.fs))
    t = np.arange(n) / p.fs
    idx = spike_times(duration, p, rng)
    return t, drift(t, p) + noise(n, p, rng) + spike_train(idx, n, p, rng), idx


def detect_spikes(x, k=5, win=501, min_len=2, dead=8):
    """Spike detector of HW2. Returns the peak indices and the drift-removed signal."""
    r = x - median_filter(x, size=win, mode='reflect')
    sigma = np.median(np.abs(r - np.median(r))) / 0.6745
    above = np.concatenate(([False], r > k * sigma, [False]))
    starts, ends = np.flatnonzero(above[1:] & ~above[:-1]), np.flatnonzero(above[:-1] & ~above[1:])
    spikes = []
    for s, e in zip(starts, ends):
        if e - s < min_len:
            continue
        p = s + np.argmax(r[s:e])
        if spikes and p - spikes[-1] < dead:
            if r[p] > r[spikes[-1]]:
                spikes[-1] = p
        else:
            spikes.append(p)
    return np.array(spikes, dtype=int), r


def hawkes_fit(idx, fs, duration, d):
    """Maximum likelihood of the spike times: excite, excite_tau and the likelihood ratio against
    independent poisson spikes."""
    t = idx / fs
    s = t - d * np.arange(len(t))
    S = duration - np.minimum(d, duration - t).sum()

    def nll(q):
        mu, alpha, beta = np.exp(q[0]), 1 / (1 + np.exp(-np.clip(q[1], -20, 8))), np.exp(np.clip(q[2], -2, 9))
        E, e = np.zeros(len(s)), 0.0
        for i, a in enumerate(np.exp(-beta * np.diff(s)), 1):
            e = a * (1 + e)
            E[i] = e
        return mu * S + alpha * (1 - np.exp(-beta * np.clip(S - s, 0, None))).sum() - np.log(mu + alpha * beta * E).sum()

    fits = [minimize(nll, [np.log(len(s) / S), a0, np.log(b0)], method='Nelder-Mead', options={'maxiter': 3000})
            for a0 in (-2, 0) for b0 in (10, 50)]
    best = min(fits, key=lambda r: r.fun)
    nll_poisson = len(s) - len(s) * np.log(len(s) / S)
    return (1 / (1 + np.exp(-np.clip(best.x[1], -20, 8))), 1 / np.exp(np.clip(best.x[2], -2, 9)),
            2 * (nll_poisson - best.fun))


def estimate(t, x, kernel_len=25):
    """All the coefficients of a recording."""
    fs = 1 / np.median(np.diff(t))
    idx, r = detect_spikes(x)
    k = np.arange(kernel_len)
    iso = [i for i in idx if np.all(np.abs(idx[idx != i] - i) > kernel_len) and i + kernel_len <= len(x)]
    (a, tau), _ = curve_fit(lambda k, a, tau: a * np.exp(-k / tau), np.tile(k, len(iso)),
                            np.concatenate([r[i:i + kernel_len] for i in iso]), p0=(5, 3))
    p = Params(fs=fs, spike_amp=a, spike_amp_sd=0, spike_tau=tau / fs,
               refractory=min(0.010, np.diff(idx).min() / fs))
    y = x - spike_train(idx, len(x), p, np.random.default_rng(0))
    freqs = np.fft.rfftfreq(len(y), 1 / fs)
    f0 = freqs[1 + np.argmax(np.abs(np.fft.rfft(y - y.mean()))[1:])]
    sine = lambda t, c, s, co, f: c + s * np.sin(2 * np.pi * f * t) + co * np.cos(2 * np.pi * f * t)
    (c, s, co, f), _ = curve_fit(sine, t, y, p0=(y.mean(), y.std(), 0, f0))
    if f < 0:
        f, s = -f, -s
    noise_sd = np.std(y - sine(t, c, s, co, f))
    h = np.exp(-k / tau)
    amps = np.array([r[i:i + kernel_len] @ h / (h @ h) for i in iso])
    amp_sd = np.sqrt(max(amps.var(ddof=1) - noise_sd ** 2 / (h @ h), 0))  # spread minus the part of the noise
    excite, excite_tau, _ = hawkes_fit(idx, fs, len(x) / fs, p.refractory)
    return replace(p, offset=c, drift_amp=np.hypot(s, co), drift_freq=f, drift_phase=np.arctan2(co, s),
                   noise_sd=noise_sd, rate=len(idx) / (len(x) / fs), spike_amp_sd=amp_sd,
                   excite=excite, excite_tau=excite_tau)
