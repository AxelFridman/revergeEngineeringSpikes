"""Run from the repo folder:  python -m pytest comparison"""
import numpy as np
import pytest

from compare import ROOT, compare, metrics
from spikesim import estimate, load, simulate

NAMES = list(metrics(*load(ROOT / 'spike_data.npz')))


@pytest.fixture(scope='module')
def table():
    return compare(n_sim=300)[0]


@pytest.mark.parametrize('name', NAMES)
def test_real_recording_looks_like_a_simulated_one(table, name):
    # the real value must be inside the range of 95% of the simulated recordings
    row = table.loc[name]
    assert row['low (2.5%)'] <= row['real'] <= row['high (97.5%)']


def test_simulated_file_has_the_same_format():
    with np.load(ROOT / 'spike_data.npz') as real, np.load(ROOT / 'simulated_spike_data.npz') as sim:
        assert real.files == sim.files
        for k in real.files:
            assert real[k].shape == sim[k].shape and real[k].dtype == sim[k].dtype


def test_simulated_file_comes_from_the_generator():
    t, x = load(ROOT / 'spike_data.npz')
    _, xs, _ = simulate(10, seed=6, p=estimate(t, x))
    assert np.array_equal(load(ROOT / 'simulated_spike_data.npz')[1], xs)
