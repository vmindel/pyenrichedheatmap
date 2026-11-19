"""Tests for smart row downsampling and raster pixel-budget calculations."""

from __future__ import annotations

import numpy as np
import pytest

import enriched_heatmap.raster as eh


def test_downsample_passthrough_when_small():
    arr = np.arange(40, dtype=np.float32).reshape(10, 4)
    out = eh.downsample_rows(arr, target_rows=20, agg="mean")
    assert out.shape == (10, 4)
    np.testing.assert_array_equal(out, arr)


def test_downsample_mean_max_min_absmax_quantiles():
    # 100 rows grouped into 10 buckets of 10 rows each
    arr = np.zeros((100, 3), dtype=np.float64)
    # Put a single sparse sharp spike in each bucket (simulating narrow TF ChEC peak)
    for b in range(10):
        arr[b * 10 + 3, 0] = 10.0
        arr[b * 10 + 5, 1] = -8.0 if b % 2 == 0 else 6.0
        arr[b * 10 : (b + 1) * 10, 2] = np.arange(10, dtype=np.float64)

    mean_ds = eh.downsample_rows(arr, target_rows=10, agg="mean")
    assert mean_ds.shape == (10, 3)
    np.testing.assert_allclose(mean_ds[:, 0], 1.0)
    np.testing.assert_allclose(mean_ds[:, 2], 4.5)

    max_ds = eh.downsample_rows(arr, target_rows=10, agg="max")
    np.testing.assert_allclose(max_ds[:, 0], 10.0)  # sparse peak preserved!

    absmax_ds = eh.downsample_rows(arr, target_rows=10, agg="absmax")
    expected_col1 = [-8.0 if b % 2 == 0 else 6.0 for b in range(10)]
    np.testing.assert_allclose(absmax_ds[:, 1], expected_col1)

    q90_ds = eh.downsample_rows(arr, target_rows=10, agg="q90")
    np.testing.assert_allclose(q90_ds[:, 2], 8.1)

    med_ds = eh.downsample_rows(arr, target_rows=10, agg="median")
    np.testing.assert_allclose(med_ds[:, 2], 4.5)

    custom_ds = eh.downsample_rows(arr, target_rows=10, agg=lambda chunk: np.sum(chunk, axis=0))
    np.testing.assert_allclose(custom_ds[:, 0], 10.0)


def test_downsample_nan_safety():
    arr = np.full((20, 2), np.nan, dtype=np.float64)
    arr[0:5, 0] = 4.0
    arr[10:20, 1] = 2.0

    out_mean = eh.downsample_rows(arr, target_rows=2, agg="mean")
    assert np.isclose(out_mean[0, 0], 4.0)
    assert np.isnan(out_mean[0, 1])
    assert np.isnan(out_mean[1, 0])
    assert np.isclose(out_mean[1, 1], 2.0)

    out_max = eh.downsample_rows(arr, target_rows=2, agg="max")
    assert np.isclose(out_max[0, 0], 4.0)
    assert np.isnan(out_max[0, 1])


def test_downsample_1d_and_categorical():
    v = np.repeat(np.array([1.0, 5.0, 9.0, 2.0]), 25)
    ds_v = eh.downsample_1d_rows(v, target_rows=4, agg="mean")
    np.testing.assert_allclose(ds_v, [1.0, 5.0, 9.0, 2.0])

    # Majority mode in each bucket
    codes = np.array([0, 0, 1, 0, 0, 2, 2, 2, -1, 1])
    ds_c = eh.downsample_categorical_rows(codes, target_rows=2)
    np.testing.assert_array_equal(ds_c, [0, 2])


def test_compute_target_raster_rows():
    px = eh.compute_target_raster_rows(fig_height_inches=6.0, axes_height_fraction=0.7, raster_dpi=300)
    assert px == 1260
    assert eh.compute_target_raster_rows(6.0, 0.7, 300, max_rows=500) == 500

    with pytest.raises(ValueError, match="Unsupported"):
        eh.downsample_rows(np.zeros((20, 2)), target_rows=5, agg="not_a_mode")
