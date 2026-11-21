"""Tests for NormalizedMatrix data model, alignment, transforms, and coordinate geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import enriched_heatmap as eh


def _make_mat(n_regions: int = 10, n_bins: int = 20, seed: int = 0) -> eh.NormalizedMatrix:
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0.0, 10.0, size=(n_regions, n_bins)).astype(np.float32)
    idx = pd.Index([f"peak_{i}" for i in range(n_regions)])
    return eh.NormalizedMatrix(
        values=vals,
        region_index=idx,
        mode="reference_point",
        upstream=n_bins * 5,
        downstream=n_bins * 5,
        bin_size=10,
        name="ATAC",
    )


def test_matrix_validation_and_defaults():
    mat = eh.NormalizedMatrix(values=np.zeros((5, 20)), upstream=100, downstream=100, bin_size=10)
    assert mat.shape == (5, 20)
    assert mat.n_regions == 5
    assert mat.n_bins == 20
    assert isinstance(mat.region_index, pd.RangeIndex)
    assert mat.values.dtype == np.float64

    with pytest.raises(ValueError, match="2D array"):
        eh.NormalizedMatrix(values=np.zeros(10))

    with pytest.raises(ValueError, match="unique"):
        eh.NormalizedMatrix(values=np.zeros((2, 10)), region_index=pd.Index(["a", "a"]),
                           upstream=50, downstream=50, bin_size=10)

    with pytest.raises(ValueError, match="target_bins"):
        eh.NormalizedMatrix(
            values=np.zeros((2, 10)),
            mode="reference_point",
            target_bins=5,
        )


def test_bin_centers_and_guides_reference_point():
    mat = _make_mat(n_regions=4, n_bins=20)
    centers = mat.bin_centers()
    assert centers.shape == (20,)
    assert np.isclose(centers[0], -95.0)
    assert np.isclose(centers[-1], 95.0)
    assert mat.guide_positions() == [0.0]
    ticks, labels = mat.default_xticks()
    assert ticks == [-100.0, 0.0, 100.0]
    assert labels == ["-100 bp", "0", "+100 bp"]


def test_bin_centers_and_guides_scale_regions():
    vals = np.ones((3, 40), dtype=np.float32)
    mat = eh.NormalizedMatrix(
        values=vals,
        mode="scale_regions",
        upstream=1000,
        downstream=1000,
        bin_size=100,
        target_bins=20,
        name="GeneBody",
    )
    assert mat.n_upstream_bins == 10
    assert mat.n_downstream_bins == 10
    assert mat.body_length == 2000.0
    assert mat.x_extent == (-1000.0, 3000.0)
    assert mat.guide_positions() == [0.0, 2000.0]
    ticks, labels = mat.default_xticks(anchor_labels=("TSS", "TES"))
    assert ticks == [-1000.0, 0.0, 2000.0, 3000.0]
    assert labels == ["-1 kb", "TSS", "TES", "+1 kb"]


def test_alignment_reindex_loc_iloc_dropna():
    mat = _make_mat(n_regions=5, n_bins=10)
    sub = mat.loc[["peak_3", "peak_1"]]
    assert list(sub.region_index) == ["peak_3", "peak_1"]
    np.testing.assert_allclose(sub.values[0], mat.values[3])
    np.testing.assert_allclose(sub.values[1], mat.values[1])

    mask = np.array([True, False, True, False, False])
    sub_mask = mat.loc[mask]
    assert list(sub_mask.region_index) == ["peak_0", "peak_2"]

    reindexed = mat.reindex(["peak_2", "missing_peak", "peak_0"], fill_value=-1.0)
    assert list(reindexed.region_index) == ["peak_2", "missing_peak", "peak_0"]
    np.testing.assert_allclose(reindexed.values[1], -1.0)
    np.testing.assert_allclose(reindexed.values[0], mat.values[2])

    with_nan = mat.copy()
    with_nan.values[1, 3] = np.nan
    with_nan.values[4, :] = np.nan
    assert with_nan.dropna_regions(how="any").n_regions == 3
    assert with_nan.dropna_regions(how="all").n_regions == 4


def test_signal_transforms_non_destructive():
    mat = _make_mat(n_regions=6, n_bins=20)
    orig = mat.values.copy()

    l1 = mat.log1p()
    np.testing.assert_allclose(l1.values, np.log1p(orig))
    np.testing.assert_allclose(mat.values, orig)

    l2 = mat.log2p(pseudocount=0.5)
    np.testing.assert_allclose(l2.values, np.log2(orig + 0.5))

    z_row = mat.zscore(axis=1)
    np.testing.assert_allclose(z_row.values.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(z_row.values.std(axis=1), 1.0, atol=1e-5)

    # Constant row should z-score safely to 0 without NaNs
    const_mat = eh.NormalizedMatrix(values=np.full((2, 10), 5.0), upstream=50, downstream=50, bin_size=10)
    np.testing.assert_allclose(const_mat.zscore(axis=1).values, 0.0)

    clipped = mat.clip_quantile(0.1, 0.9)
    assert clipped.values.max() <= np.quantile(orig, 0.9) + 1e-6
    assert clipped.values.min() >= np.quantile(orig, 0.1) - 1e-6

    sm = mat.smooth(sigma=1.5)
    assert sm.shape == mat.shape
    assert sm.values.std() < orig.std()

    # NaN-aware smoothing neither spreads NaNs to neighbours nor fills them in
    nan_mat = mat.copy()
    nan_mat.values[0, 5] = np.nan
    sm_nan = nan_mat.smooth(sigma=1.0)
    assert np.isnan(sm_nan.values[0, 5])
    assert np.all(np.isfinite(np.delete(sm_nan.values[0], 5)))


def test_differential_helpers_and_alignment():
    a = _make_mat(n_regions=5, n_bins=10, seed=1)
    a.name = "Mut"
    b = _make_mat(n_regions=5, n_bins=10, seed=2)
    b.name = "WT"
    # Permute b's rows to ensure automatic region_index alignment
    b_perm = b.loc[["peak_4", "peak_2", "peak_0", "peak_3", "peak_1"]]

    diff = a.subtract(b_perm)
    assert diff.metadata.get("is_differential") is True
    np.testing.assert_allclose(diff.values, a.values - b.values)

    lfc = a.log2_ratio(b_perm, pseudocount=1.0)
    assert lfc.metadata.get("is_differential") is True
    expected = np.log2((a.values + 1.0) / (b.values + 1.0))
    np.testing.assert_allclose(lfc.values, expected, rtol=1e-5)


def test_interchange_roundtrip():
    mat = _make_mat(n_regions=4, n_bins=20)
    payload = mat.to_dict()
    rebuilt = eh.NormalizedMatrix.from_dict(payload)
    np.testing.assert_allclose(rebuilt.values, mat.values)
    assert list(rebuilt.region_index) == list(mat.region_index)
    assert rebuilt.upstream == mat.upstream
    assert rebuilt.name == mat.name


# ---------------------------------------------------------------------------
# Regressions: smoothing, geometry validation, equality
# ---------------------------------------------------------------------------
def _flat(n_bins: int = 20, **kw) -> eh.NormalizedMatrix:
    return eh.NormalizedMatrix(
        np.ones((3, n_bins)), upstream=n_bins * 5, downstream=n_bins * 5, bin_size=10, **kw
    )


def test_smooth_never_fills_nan_bins():
    vals = np.ones((2, 20))
    vals[0, 5] = np.nan
    vals[1, :4] = np.nan  # an edge run, e.g. bins before a chromosome start
    mat = eh.NormalizedMatrix(vals, upstream=100, downstream=100, bin_size=10)
    out = mat.smooth(1.5).values
    assert np.array_equal(np.isnan(out), np.isnan(vals))
    np.testing.assert_allclose(out[np.isfinite(out)], 1.0)  # neighbours ignore the gaps


def test_smooth_modes_validated_and_numpy_fallback_matches_scipy(monkeypatch):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    vals = rng.random((4, 30))
    vals[1, 7] = np.nan
    mat = eh.NormalizedMatrix(vals, upstream=150, downstream=150, bin_size=10)

    with pytest.raises(ValueError, match="mode"):
        mat.smooth(1.0, mode="bogus")

    with_scipy = {m: mat.smooth(2.0, mode=m).values
                  for m in ("nearest", "reflect", "mirror", "wrap", "constant")}
    monkeypatch.setitem(__import__("sys").modules, "scipy.ndimage", None)  # force ImportError
    for m, expected in with_scipy.items():
        np.testing.assert_allclose(mat.smooth(2.0, mode=m).values, expected, atol=1e-10,
                                   err_msg=m)


def test_smooth_zero_sigma_is_identity():
    mat = _make_mat()
    np.testing.assert_array_equal(mat.smooth(0.0).values, mat.values)


def test_constructor_rejects_columns_that_do_not_match_the_geometry():
    with pytest.raises(ValueError, match="imply 20"):
        eh.NormalizedMatrix(np.ones((2, 7)), upstream=100, downstream=100, bin_size=10)
    with pytest.raises(ValueError, match="multiples"):
        eh.NormalizedMatrix(np.ones((2, 20)), upstream=95, downstream=105, bin_size=10)
    with pytest.raises(ValueError, match="imply 30"):
        eh.NormalizedMatrix(np.ones((2, 31)), mode="scale_regions", upstream=100,
                            downstream=100, bin_size=10, target_bins=10)
    # a matrix with no flanks keeps its own column count
    assert eh.NormalizedMatrix(np.ones((2, 9)), upstream=0, downstream=0, bin_size=10).n_bins == 9


def test_differential_ops_refuse_mismatched_geometry():
    a = _flat(20)
    other_flank = eh.NormalizedMatrix(np.ones((3, 10)), upstream=50, downstream=50, bin_size=10)
    other_bin = eh.NormalizedMatrix(np.ones((3, 20)), upstream=200, downstream=200, bin_size=20)
    scaled = eh.NormalizedMatrix(np.ones((3, 20)), mode="scale_regions", upstream=50,
                                 downstream=50, bin_size=10, target_bins=10)
    for other in (other_flank, other_bin, scaled):
        with pytest.raises(ValueError, match="different bin geometry"):
            a.subtract(other)
        with pytest.raises(ValueError, match="different bin geometry"):
            a.log2_ratio(other)
    with pytest.raises(TypeError):
        a.subtract(np.ones((3, 20)))
    assert a.subtract(_flat(20)).metadata["is_differential"]


def test_matrices_compare_by_identity_without_raising():
    a = _make_mat()
    b = a.copy()
    assert a == a
    assert (a == b) is False  # no "truth value of an array is ambiguous"
    assert len({a, b}) == 2   # and they stay hashable
