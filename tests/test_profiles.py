"""Summary profiles: bands, NaN handling, bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from enriched_heatmap.profiles import compute_profile


def _data(n=2000, bins=30, nan_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.gamma(2.0, 1.0, size=(n, bins))
    a[rng.random(a.shape) < nan_frac] = np.nan
    return a


def test_bootstrap_band_agrees_with_the_normal_interval_and_is_seeded():
    a = _data()
    boot = compute_profile(a, "mean_boot_ci", n_bootstrap=400, random_state=3)
    norm = compute_profile(a, "mean_ci")
    ratio = (boot.upper - boot.lower) / (norm.upper - norm.lower)
    assert 0.85 < np.median(ratio) < 1.15
    np.testing.assert_allclose(boot.center, norm.center)
    assert np.all(boot.lower <= boot.center) and np.all(boot.center <= boot.upper)
    again = compute_profile(a, "mean_boot_ci", n_bootstrap=400, random_state=3)
    np.testing.assert_array_equal(boot.lower, again.lower)
    other = compute_profile(a, "mean_boot_ci", n_bootstrap=400, random_state=4)
    assert not np.array_equal(boot.lower, other.lower)


def test_bootstrap_handles_empty_columns_and_single_rows():
    a = _data(n=50)
    a[:, 4] = np.nan
    res = compute_profile(a, "mean_boot_ci", n_bootstrap=20)
    assert np.isnan(res.center[4]) and np.isnan(res.lower[4]) and np.isnan(res.upper[4])
    assert np.isfinite(np.delete(res.lower, 4)).all()

    one = compute_profile(a[:1], "mean_boot_ci")
    np.testing.assert_array_equal(one.lower, one.center)


def test_bootstrap_with_many_resamples_stays_in_memory_budget():
    import tracemalloc

    a = _data(n=40_000, bins=20, nan_frac=0.0)
    tracemalloc.start()
    compute_profile(a, "mean_boot_ci", n_bootstrap=300)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 400e6  # chunked: never the full 300 x 40,000 counts at once


@pytest.mark.parametrize("mode", ["mean", "median", "mean_sem", "mean_ci", "median_iqr"])
def test_bands_ignore_nan(mode):
    a = _data(n=300)
    res = compute_profile(a, mode)
    assert np.isfinite(res.center).all()
    assert res.n_valid.max() <= 300 and res.n_valid.min() > 200
