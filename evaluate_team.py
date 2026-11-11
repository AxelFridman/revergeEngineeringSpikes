"""Part 2 of the inter-team evaluation.

    from evaluate_team import evaluate
    result = evaluate('spike_data.npz', 'team_X_reconstruction.npz')
    result['final_score']          # 0 to 100

    python evaluate_team.py spike_data.npz team_X_reconstruction.npz [more files ...]

We only have the recording, so a reconstruction is judged by what it leaves behind: if drift and spikes are right,
recording - reconstruction is white gaussian noise with the noise level of the recording, with nothing left at the
spikes and no slow trend. Only numpy, scipy and matplotlib are needed.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter

KEYS = ['t', 'drift', 'spike_signal', 'noise', 'reconstruction', 'spike_indices']
WEIGHTS = {'fit': 30, 'spike_detection': 25, 'spike_shape': 15, 'drift': 15, 'noise': 10, 'whiteness': 5}


def _load_reference(path):
    d = np.load(path, allow_pickle=False)
    if isinstance(d, np.ndarray):
        return np.arange(len(d)) / 1000.0, d.astype(float)
    with d:
        x = d['raw_data'] if 'raw_data' in d.files else d[d.files[0]]
        t = d['t'] if 't' in d.files else np.arange(len(x)) / 1000.0
        return np.asarray(t, float), np.asarray(x, float)


def robust_sd(r):
    return float(np.median(np.abs(r - np.median(r))) / 0.6745)


def reference_spikes(x, k=5, win=501, dead=8):
    # threshold detector on the drift-removed recording (peaks; with a one-sample jump this is the spike start)
    r = x - median_filter(x, size=win, mode='reflect')
    above = np.concatenate(([False], r > k * robust_sd(r), [False]))
    starts, ends = np.flatnonzero(above[1:] & ~above[:-1]), np.flatnonzero(above[:-1] & ~above[1:])
    peaks = []
    for s, e in zip(starts, ends):
        p = s + int(np.argmax(r[s:e]))
        if peaks and p - peaks[-1] < dead:
            if r[p] > r[peaks[-1]]:
                peaks[-1] = p
        else:
            peaks.append(p)
    return np.array(peaks, dtype=int), robust_sd(r)


def match_spikes(ref, cand, tol):
    # greedy one-to-one matching inside +-tol samples
    used, hits, offsets = np.zeros(len(cand), bool), 0, []
    for i in ref:
        d = np.abs(cand - i).astype(float)
        d[used] = np.inf
        if len(d) and d.min() <= tol:
            j = int(np.argmin(d))
            used[j] = True
            hits += 1
            offsets.append(int(cand[j] - i))
    precision = hits / len(cand) if len(cand) else 0.0
    recall = hits / len(ref) if len(ref) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return precision, recall, f1, offsets


def validity(cand, n, t_ref):
    checks, fatal = {}, []
    missing = [k for k in KEYS if k not in cand]
    checks['all keys present'] = not missing
    if missing:
        fatal.append(f'missing keys {missing}')
        return checks, fatal
    for k in KEYS[:-1]:
        a = cand[k]
        ok = a.shape == (n,) and np.issubdtype(a.dtype, np.number) and np.all(np.isfinite(a))
        checks[f'{k} shape ({n},), finite'] = bool(ok)
        if not ok:
            fatal.append(f'{k} has shape {a.shape} / non finite values')
    si = cand['spike_indices']
    ok = si.ndim == 1 and np.issubdtype(si.dtype, np.integer) and (len(si) == 0 or (si.min() >= 0 and si.max() < n))
    checks['spike_indices 1-d integers inside the recording'] = bool(ok)
    if not ok:
        fatal.append('bad spike_indices')
    if fatal:
        return checks, fatal
    checks['t equals the reference time vector'] = bool(np.allclose(cand['t'], t_ref))
    structured = cand['drift'] + cand['spike_signal']
    checks['reconstruction = drift + spike_signal'] = bool(np.allclose(cand['reconstruction'], structured, atol=1e-6))
    checks['reconstruction = drift + spike_signal + noise'] = bool(
        np.allclose(cand['reconstruction'], structured + cand['noise'], atol=1e-6))
    checks['spike_indices sorted, no duplicates'] = bool(np.all(np.diff(si) > 0))
    return checks, fatal


def _score(value, good, bad):
    # 1 when value <= good, 0 when value >= bad, linear in between
    return float(np.clip((bad - value) / (bad - good), 0, 1))


def evaluate(reference_file, candidate_file, figure=True, tol=2):
    """Returns validity checks, metrics, sub-scores (0..1), an overlay figure and final_score (0..100)."""
    t, x = _load_reference(reference_file)
    n = len(x)
    with np.load(candidate_file, allow_pickle=False) as f:
        cand = {k: np.asarray(f[k]) for k in f.files}
    checks, fatal = validity(cand, n, t)
    out = {'reference_file': Path(reference_file).name, 'candidate_file': Path(candidate_file).name,
           'validity': not fatal,
           'validity_checks': checks, 'validity_details': '; '.join(fatal) or 'passed'}
    if fatal:
        out.update(final_score=0.0, metrics={}, scores={}, overlay_figure=None)
        return out

    rec = cand['reconstruction'].astype(float)
    # if a team added a noise realization to the reconstruction, judge only the structured part
    if checks['reconstruction = drift + spike_signal + noise'] and not checks['reconstruction = drift + spike_signal']:
        rec = rec - cand['noise']
    res = x - rec
    ref_idx, sigma = reference_spikes(x)
    cand_idx = np.unique(cand['spike_indices'].astype(int))
    prec, recall, f1, offsets = match_spikes(ref_idx, cand_idx, tol)

    # residual just after the spikes of the recording (spike shape and amplitude errors)
    lag = np.arange(8)
    post = np.array([res[i + lag] for i in ref_idx if i + 8 <= n])
    post_err = float(np.sqrt(np.mean(post.mean(0) ** 2))) if len(post) else np.inf
    peak_err = float(np.sqrt(np.mean(post[:, 0] ** 2))) if len(post) else np.inf
    # slow part of the residual (drift errors): 0.5 s moving average
    w = 501
    slow = np.convolve(res - res.mean(), np.ones(w) / w, mode='same')[w:-w]
    slow_rms = float(np.sqrt(np.mean(slow ** 2)) + abs(res.mean()))
    rms = float(np.sqrt(np.mean(res ** 2)))
    rho1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
    noise_sd = float(np.std(cand['noise']))
    ks_p = float(stats.kstest((res - res.mean()) / res.std(), 'norm').pvalue)

    metrics = {
        'n spikes reference': int(len(ref_idx)), 'n spikes candidate': int(len(cand_idx)),
        'spike precision': prec, 'spike recall': recall, 'spike F1': f1,
        'mean spike offset (samples)': float(np.mean(offsets)) if offsets else None,
        'noise sd of recording (robust)': sigma, 'residual rms': rms, 'residual rms / noise sd': rms / sigma,
        'R2': float(1 - np.var(res) / np.var(x)),
        'residual after spikes rms (8 ms average)': post_err, 'residual at spike peak rms': peak_err,
        'residual slow part rms': slow_rms,
        'candidate noise sd': noise_sd, 'candidate noise sd / noise sd': noise_sd / sigma,
        'residual lag-1 autocorrelation': rho1, 'residual normality KS p': ks_p,
    }
    scores = {
        'fit': _score(rms / sigma, 0.95, 1.2),
        'spike_detection': f1,
        # average residual over all spikes, so fitting the noise of single spikes does not help
        'spike_shape': _score(post_err / sigma, 0.1, 0.5),
        'drift': _score(slow_rms / sigma, 0.02, 0.3),
        'noise': _score(abs(np.log(noise_sd / sigma)) if noise_sd > 0 else np.inf, 0.05, 0.7),
        'whiteness': _score(abs(rho1), 0.05, 0.4),
    }
    final = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    if not checks['t equals the reference time vector']:
        final *= 0.9   # wrong time vector: 10% penalty
    out.update(metrics=metrics, scores=scores, weights=WEIGHTS, final_score=round(float(final), 2))
    out['overlay_figure'] = _overlay(t, x, rec, ref_idx, cand_idx, res, Path(candidate_file).stem, out['final_score']) \
        if figure else None
    return out


def _overlay(t, x, rec, ref_idx, cand_idx, res, name, score):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2, 2, 1]})
    ax[0].plot(t, x, color='0.6', lw=0.6, label='recording')
    ax[0].plot(t, rec, color='C3', lw=0.8, label='reconstruction')
    ax[0].set_title(f'{name}   score {score:.1f} / 100')
    i0 = int(ref_idx[0]) if len(ref_idx) else 0
    sl = slice(max(i0 - 200, 0), min(i0 + 800, len(x)))
    ax[1].plot(t[sl], x[sl], color='0.6', lw=1, label='recording')
    ax[1].plot(t[sl], rec[sl], color='C3', lw=1.2, label='reconstruction')
    for i in ref_idx[(ref_idx >= sl.start) & (ref_idx < sl.stop)]:
        ax[1].axvline(t[i], color='C0', lw=0.8, ls=':')
    m = (cand_idx >= sl.start) & (cand_idx < sl.stop)
    ax[1].plot(t[cand_idx[m]], rec[cand_idx[m]], 'v', color='C3', label='candidate spike starts')
    ax[2].plot(t, res, color='C2', lw=0.5)
    ax[2].set_ylabel('residual')
    ax[2].set_xlabel('time (s)')
    for a in ax[:2]:
        a.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig


def save_result(result, out_dir):
    """Writes <candidate>.json and <candidate>.png into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result['candidate_file']).stem
    fig = result.get('overlay_figure')
    if fig is not None:
        fig.savefig(out_dir / f'{stem}.png', dpi=110)
        import matplotlib.pyplot as plt
        plt.close(fig)
    with open(out_dir / f'{stem}.json', 'w') as f:
        json.dump({k: v for k, v in result.items() if k != 'overlay_figure'}, f, indent=2)


if __name__ == '__main__':
    matplotlib.use('Agg')
    if len(sys.argv) < 3:
        sys.exit('python evaluate_team.py spike_data.npz team_X_reconstruction.npz [more ...]')
    for c in sys.argv[2:]:
        r = evaluate(sys.argv[1], c)
        save_result(r, Path('evaluation_results'))
        print(f"{Path(c).name:45s} valid={r['validity']}  score={r['final_score']:6.2f}  {r['validity_details']}")
