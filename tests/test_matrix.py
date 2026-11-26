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


def test_loc_aligns_boolean_series_by_label_and_supports_reverse_slices():
    mat = _make_mat(n_regions=4, n_bins=10)
    # Reversed index: True only for peak_3
    mask = pd.Series([True, False, False, False], index=mat.region_index[::-1])
    assert list(mat.loc[mask].region_index) == ["peak_3"]

    nullable = pd.Series([True, False, True, False], index=mat.region_index, dtype="boolean")
    assert list(mat.loc[nullable].region_index) == ["peak_0", "peak_2"]

    assert list(mat.loc[::-1].region_index) == list(mat.region_index[::-1])
    assert list(mat.loc["peak_2":"peak_0":-1].region_index) == ["peak_2", "peak_1", "peak_0"]

    with pytest.raises(ValueError, match="unique identifiers"):
        _ = mat.loc[pd.Series([True, False, True, False], index=["peak_0", "peak_0", "peak_1", "peak_2"])]
    with pytest.raises(ValueError, match="Boolean mask length"):
        _ = mat.loc[np.array([], dtype=bool)]
    with pytest.raises(ValueError, match="Boolean mask length"):
        _ = mat.iloc[np.array([True, False])]
    with pytest.raises(TypeError):
        _ = mat.iloc[True]


def test_flankless_matrices_with_different_column_counts_refuse_combine():
    a = eh.NormalizedMatrix(np.ones((2, 9)), upstream=0, downstream=0, bin_size=10)
    b = eh.NormalizedMatrix(np.ones((2, 5)), upstream=0, downstream=0, bin_size=10)
    with pytest.raises(ValueError, match="n_bins: 9 vs 5"):
        a.subtract(b)


def test_zscore_with_all_nan_rows_does_not_warn():
    import warnings

    mat = _make_mat(n_regions=4, n_bins=10)
    mat.values[1, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = mat.zscore(axis=1)
    assert np.isnan(out.values[1]).all()


# ---------------------------------------------------------------------------
# M1: Zero-overlap guard (F02), strict .iloc/.loc (F13), interchange (F22),
#     and dataclass/static contract (F23)
# ---------------------------------------------------------------------------
def test_differential_ops_reject_zero_overlap_and_allow_partial_or_empty():
    a = _make_mat(n_regions=3, n_bins=10, seed=10)
    b_range = eh.NormalizedMatrix(
        values=np.ones((3, 10), dtype=np.float32),
        region_index=pd.RangeIndex(3),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    b_disjoint = eh.NormalizedMatrix(
        values=np.ones((2, 10), dtype=np.float32),
        region_index=pd.Index(["other_0", "other_1"]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    mi2 = eh.NormalizedMatrix(
        values=np.ones((2, 10), dtype=np.float32),
        region_index=pd.MultiIndex.from_tuples([("chr1", 10), ("chr1", 20)]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    mi3 = eh.NormalizedMatrix(
        values=np.ones((2, 10), dtype=np.float32),
        region_index=pd.MultiIndex.from_tuples([("chr1", 10, "+"), ("chr1", 20, "-")]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )

    for left, right in [(a, b_range), (a, b_disjoint), (mi2, b_range), (mi2, mi3)]:
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            left.subtract(right)
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            left.log2_ratio(right)

    # Partial overlap aligns shared IDs and leaves unmatched rows as NaN
    b_partial = eh.NormalizedMatrix(
        values=np.full((2, 10), 2.0, dtype=np.float32),
        region_index=pd.Index(["peak_1", "unseen"]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    diff = a.subtract(b_partial)
    np.testing.assert_allclose(diff.values[1], a.values[1] - 2.0)
    assert np.isnan(diff.values[0]).all() and np.isnan(diff.values[2]).all()

    lfc = a.log2_ratio(b_partial, pseudocount=1.0)
    np.testing.assert_allclose(lfc.values[1], np.log2((a.values[1] + 1.0) / 3.0), rtol=1e-5)
    assert np.isnan(lfc.values[0]).all() and np.isnan(lfc.values[2]).all()

    # 0-row inputs remain allowed on either or both sides
    assert a.iloc[:0].subtract(b_disjoint).shape == (0, 10)
    assert a.iloc[:0].log2_ratio(b_disjoint).shape == (0, 10)
    empty_rhs_sub = a.subtract(b_disjoint.iloc[:0])
    assert empty_rhs_sub.shape == (3, 10) and np.isnan(empty_rhs_sub.values).all()
    empty_rhs_lfc = a.log2_ratio(b_disjoint.iloc[:0])
    assert empty_rhs_lfc.shape == (3, 10) and np.isnan(empty_rhs_lfc.values).all()
    assert a.iloc[:0].subtract(b_disjoint.iloc[:0]).shape == (0, 10)

    # Direct reindex remains permissive even with zero overlap, including tuple/ndarray targets
    for target in (["unseen_0", "unseen_1"], ("unseen_0", "unseen_1"), np.array(["u0", "u1"])):
        reindexed = a.reindex(target)
        assert reindexed.shape == (2, 10)
        assert np.isnan(reindexed.values).all()

    # Cross-dtype numeric vs boolean & disjoint NA-bearing indices also raise ValueError
    m_range = eh.NormalizedMatrix(np.ones((2, 10)), upstream=50, downstream=50, bin_size=10)
    m_int64 = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index([0, 1], dtype="Int64"), upstream=50, downstream=50, bin_size=10
    )
    m_obj_int = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index([0, 1], dtype=object), upstream=50, downstream=50, bin_size=10
    )
    m_cat_int = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.CategoricalIndex([0, 1]), upstream=50, downstream=50, bin_size=10
    )
    bool_mats = [
        eh.NormalizedMatrix(
            np.full((2, 10), 2.0), region_index=pd.Index([False, True], dtype=bool), upstream=50, downstream=50, bin_size=10
        ),
        eh.NormalizedMatrix(
            np.full((2, 10), 2.0), region_index=pd.Index([False, True], dtype="boolean"), upstream=50, downstream=50, bin_size=10
        ),
        eh.NormalizedMatrix(
            np.full((2, 10), 2.0), region_index=pd.Index([False, True], dtype=object), upstream=50, downstream=50, bin_size=10
        ),
        eh.NormalizedMatrix(
            np.full((2, 10), 2.0), region_index=pd.CategoricalIndex([False, True]), upstream=50, downstream=50, bin_size=10
        ),
    ]
    for num_m in (m_range, m_int64, m_obj_int, m_cat_int):
        for bool_m in bool_mats:
            with pytest.raises(ValueError, match="shares no region identifiers with"):
                num_m.subtract(bool_m)
            with pytest.raises(ValueError, match="shares no region identifiers with"):
                bool_m.log2_ratio(num_m)

    m_str_na = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index(["x", pd.NA], dtype="string"), upstream=50, downstream=50, bin_size=10
    )
    m_str_na2 = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index(["y", pd.NA], dtype="string"), upstream=50, downstream=50, bin_size=10
    )
    m_f64_nan = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index([9.0, np.nan], dtype="float64"), upstream=50, downstream=50, bin_size=10
    )
    m_obj_none1 = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index(["a", None], dtype=object), upstream=50, downstream=50, bin_size=10
    )
    m_obj_none2 = eh.NormalizedMatrix(
        np.ones((2, 10)), region_index=pd.Index(["b", None], dtype=object), upstream=50, downstream=50, bin_size=10
    )
    for na_left, na_right in [
        (m_str_na, m_f64_nan),
        (m_str_na, m_str_na2),
        (m_obj_none1, m_obj_none2),
    ]:
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            na_left.subtract(na_right)
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            na_left.log2_ratio(na_right)

    # MultiIndex level mismatch in direct reindex returns all-NaN without AssertionError or prefix match
    r_3to2 = mi3.reindex(mi2.region_index)
    assert r_3to2.shape == (2, 10) and np.isnan(r_3to2.values).all()
    r_2to3 = mi2.reindex(mi3.region_index)
    assert r_2to3.shape == (2, 10) and np.isnan(r_2to3.values).all()
    tup3 = pd.Index([("chr1", 10, "+"), ("chr1", 20, "-")], tupleize_cols=False)
    assert np.isnan(mi2.reindex(tup3).values).all()


def test_iloc_and_loc_reject_2d_tuples_and_multidim_arrays():
    mat = _make_mat(n_regions=4, n_bins=10)

    for bad_tuple in [(0, 0), (slice(None), slice(None)), (0,), ()]:
        with pytest.raises(IndexError):
            _ = mat.iloc[bad_tuple]

    for bad_loc_tuple in [("peak_0", "peak_1"), ("peak_0", slice(None)), ("peak_0",), ()]:
        with pytest.raises(IndexError):
            _ = mat.loc[bad_loc_tuple]

    with pytest.raises(TypeError):
        _ = mat.loc[pd.DataFrame({"x": [True, False, True, False]})]
    for bad_loc_arr in [
        np.array([["peak_0"]]),
        np.array([[True, False, True, False]]),
        np.array(True),
    ]:
        with pytest.raises(IndexError):
            _ = mat.loc[bad_loc_arr]

    for bad_iloc_container in [
        np.array([[0, 1]]),
        np.array([[True, False, True, False]]),
        np.array(0),
        pd.DataFrame({"x": [0, 1]}),
        pd.MultiIndex.from_tuples([("a", 0), ("b", 1)]),
        {"a": 0},
        {0, 1},
        frozenset([0, 1]),
        b"01",
        bytearray(b"01"),
    ]:
        with pytest.raises(TypeError):
            _ = mat.iloc[bad_iloc_container]


def test_iloc_and_loc_strict_scalar_slice_mask_and_sequence_validation():
    mat = _make_mat(n_regions=4, n_bins=10)

    # Scalar rejections in iloc
    for bad_scalar in [
        True,
        False,
        np.bool_(True),
        "2",
        b"2",
        1.9,
        1.0,
        np.float64(1.0),
        np.nan,
        None,
        ...,
        pd.NA,
    ]:
        with pytest.raises(TypeError):
            _ = mat.iloc[bad_scalar]

    # Valid scalar integers & out-of-bounds IndexError (including uint64 >= 2**63)
    assert list(mat.iloc[0].region_index) == ["peak_0"]
    assert list(mat.iloc[-1].region_index) == ["peak_3"]
    assert list(mat.iloc[np.int64(-2)].region_index) == ["peak_2"]
    assert list(mat.iloc[np.uint8(1)].region_index) == ["peak_1"]
    for oob in [4, -5, 10**30, -(10**30), np.uint64(1 << 63), np.uint64((1 << 64) - 1)]:
        with pytest.raises(IndexError):
            _ = mat.iloc[oob]

    # Slice bound validation
    for bad_slice in [
        slice("0", "2"),
        slice(0, 1.5),
        slice(0, True),
        slice(False, 2),
        slice(0, 2, True),
    ]:
        with pytest.raises(TypeError):
            _ = mat.iloc[bad_slice]
    assert list(mat.iloc[1:3].region_index) == ["peak_1", "peak_2"]
    assert mat.iloc[:0].shape == (0, 10)

    # 1D boolean masks across container types
    for good_mask in [
        np.array([True, False, True, False]),
        pd.Series([True, False, True, False], index=["x", "y", "z", "w"]),
        pd.Index([True, False, True, False]),
        pd.array([True, False, True, False], dtype="boolean"),
        [True, False, True, False],
        pd.Categorical([True, False, True, False]),
        np.array([True, False, True, False], dtype=object),
    ]:
        assert list(mat.iloc[good_mask].region_index) == ["peak_0", "peak_2"]

    # NA in boolean masks -> ValueError in both iloc and loc
    for na_mask in [
        pd.Series([True, pd.NA, False, True], index=mat.region_index, dtype="boolean"),
        pd.array([True, pd.NA, False, True], dtype="boolean"),
        [True, pd.NA, False, True],
        [True, None, False, True],
        [True, np.nan, False, True],
        np.array([True, None, False, True], dtype=object),
        pd.Categorical([True, None, False, True], categories=[True, False]),
    ]:
        with pytest.raises(ValueError, match="Boolean mask cannot contain NA"):
            _ = mat.iloc[na_mask]
        with pytest.raises(ValueError, match="Boolean mask cannot contain NA"):
            _ = mat.loc[na_mask]

    # Wrong-length boolean masks -> ValueError
    for short_mask in [
        [True, False],
        pd.Series([True, False]),
        pd.Categorical([True, False]),
    ]:
        with pytest.raises(ValueError, match="Boolean mask length"):
            _ = mat.iloc[short_mask]
        with pytest.raises(ValueError, match="Boolean mask length"):
            _ = mat.loc[short_mask]

    # Empty & valid integer sequences/iterators/arrays
    for empty_idx in [
        [],
        range(0),
        np.array([], dtype=int),
        pd.Index([], dtype=int),
        pd.Series([], dtype=int),
    ]:
        assert mat.iloc[empty_idx].shape == (0, 10)

    for int_idx in [
        [3, 0, -2],
        range(3, 0, -2),
        (i for i in [3, 1]),
        np.array([3, 1]),
        pd.Index([3, 1]),
        pd.Series([3, 1], dtype="Int64"),
        pd.Categorical([3, 1]),
    ]:
        sub = mat.iloc[int_idx]
        assert sub.n_regions >= 2
        assert sub.region_index[0] == "peak_3"

    # Non-integer, mixed, and NA-integer sequences -> TypeError
    for bad_seq in [
        [0, True],
        [True, 0],
        ["2"],
        [1.9],
        [1.0],
        [None],
        [[0, 1]],
        np.array([], dtype=float),
        np.array(["2"]),
        np.array([1.0]),
        np.array([0, True], dtype=object),
        pd.Index(["peak_0"]),
        pd.Series([0, pd.NA], dtype="Int64"),
        pd.Series([0, pd.NA], dtype="UInt64"),
        pd.Index([0, pd.NA], dtype="Int64"),
        pd.Categorical([0, None], categories=[0, 1]),
        (str(i) for i in range(2)),
    ]:
        with pytest.raises(TypeError):
            _ = mat.iloc[bad_seq]

    # Out-of-bounds uint64 >= 2**63 sequences/arrays/Series and huge range -> IndexError
    for oob_seq in [
        [np.uint64(1 << 63)],
        np.array([1 << 63], dtype=np.uint64),
        pd.Series([1 << 63], dtype="UInt64"),
        pd.Index([1 << 63], dtype="uint64"),
        [10**30],
        (10**30 for _ in range(1)),
        range(10**30, 10**30 + 1),
        range(0, 10**30),
        range(0, -(10**30), -1),
    ]:
        with pytest.raises(IndexError):
            _ = mat.iloc[oob_seq]
    assert mat.iloc[range(10**30, 10**30)].shape == (0, 10)

    # Boolean generators / non-list iterables in .loc across string, object-int, and bool indices
    from collections import UserList, deque

    assert list(mat.loc[(i % 2 == 0 for i in range(4))].region_index) == ["peak_0", "peak_2"]
    assert list(mat.loc[deque([False, True, False, True])].region_index) == ["peak_1", "peak_3"]
    assert list(mat.loc[UserList([True, False, False, True])].region_index) == ["peak_0", "peak_3"]

    m_obj = eh.NormalizedMatrix(
        np.ones((4, 10)),
        region_index=pd.Index([0, 1, 2, 3], dtype=object),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    with pytest.raises(ValueError, match="Boolean mask length"):
        _ = m_obj.loc[(b for b in [True, False])]
    with pytest.raises(ValueError, match="Boolean mask cannot contain NA"):
        _ = m_obj.loc[(x for x in [True, pd.NA, False, True])]

    m_bool_idx = eh.NormalizedMatrix(
        np.ones((2, 10)),
        region_index=pd.Index([False, True], dtype=bool),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    assert list(m_bool_idx.loc[(b for b in [False, True])].region_index) == [True]

    # MultiIndex boolean Series with mismatched nlevels raises KeyError instead of AssertionError
    mi2 = eh.NormalizedMatrix(
        np.ones((2, 10)),
        region_index=pd.MultiIndex.from_tuples([("a", 0), ("b", 1)]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    mi3 = eh.NormalizedMatrix(
        np.ones((2, 10)),
        region_index=pd.MultiIndex.from_tuples([("a", 0, "+"), ("b", 1, "-")]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    s_mi3 = pd.Series([True, False], index=mi3.region_index)
    s_mi2 = pd.Series([True, False], index=mi2.region_index)
    with pytest.raises(KeyError, match="Boolean Series index"):
        _ = mi2.loc[s_mi3]
    with pytest.raises(KeyError, match="Boolean Series index"):
        _ = mi3.loc[s_mi2]


def test_interchange_preserves_region_index_name_and_legacy_compat():
    from types import MappingProxyType

    for idx_name in ["peak", "name", 0, ("chr", "pos"), None]:
        mat = _make_mat(n_regions=3, n_bins=10)
        mat.region_index = mat.region_index.rename(idx_name)
        payload = mat.to_dict()
        assert "region_index_name" in payload
        assert payload["region_index_name"] == idx_name
        rebuilt = eh.NormalizedMatrix.from_dict(MappingProxyType(payload))
        assert rebuilt.region_index.name == idx_name
        assert list(rebuilt.region_index) == list(mat.region_index)

    # Named RangeIndex roundtrip
    rmat = eh.NormalizedMatrix(
        np.ones((3, 10)),
        region_index=pd.RangeIndex(3, name="row_id"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    assert eh.NormalizedMatrix.from_dict(rmat.to_dict()).region_index.name == "row_id"

    # Legacy payloads omitting region_index_name or region_index
    legacy = {
        "values": np.ones((2, 10)),
        "region_index": ["a", "b"],
        "upstream": 50,
        "downstream": 50,
        "bin_size": 10,
    }
    assert eh.NormalizedMatrix.from_dict(legacy).region_index.name is None

    legacy_named_idx = {
        "values": np.ones((2, 10)),
        "region_index": pd.Index(["a", "b"], name="kept"),
        "upstream": 50,
        "downstream": 50,
        "bin_size": 10,
    }
    assert eh.NormalizedMatrix.from_dict(legacy_named_idx).region_index.name == "kept"

    minimal = eh.NormalizedMatrix.from_dict({"values": np.zeros((3, 200))})
    assert isinstance(minimal.region_index, pd.RangeIndex)
    assert minimal.region_index.name is None

    named_no_index = eh.NormalizedMatrix.from_dict(
        {"values": np.zeros((3, 200)), "region_index_name": "auto_id", "name": None}
    )
    assert named_no_index.region_index.name == "auto_id"
    assert named_no_index.name == ""

    # 1D tuple Index (named, unnamed, tuple-named) stays 1D Index rather than MultiIndex
    for tup_name in ["interval", None, ("chr", "pos")]:
        idx_tup = pd.Index([("chr1", 10), ("chr1", 20)], tupleize_cols=False, name=tup_name)
        m_tup = eh.NormalizedMatrix(
            np.ones((2, 10)), region_index=idx_tup, upstream=50, downstream=50, bin_size=10
        )
        rebuilt_tup = eh.NormalizedMatrix.from_dict(m_tup.to_dict())
        assert not isinstance(rebuilt_tup.region_index, pd.MultiIndex)
        assert rebuilt_tup.region_index.name == tup_name
        assert list(rebuilt_tup.region_index) == [("chr1", 10), ("chr1", 20)]

    # Named, partially-named, unnamed, and 0-row MultiIndex roundtrip
    for mi_names in [["chrom", "pos"], [None, None], ["chrom", None]]:
        mi = pd.MultiIndex.from_tuples([("chr1", 100), ("chr1", 200)], names=mi_names)
        m_mi = eh.NormalizedMatrix(
            np.ones((2, 10)), region_index=mi, upstream=50, downstream=50, bin_size=10
        )
        rebuilt_mi = eh.NormalizedMatrix.from_dict(m_mi.to_dict())
        assert isinstance(rebuilt_mi.region_index, pd.MultiIndex)
        assert list(rebuilt_mi.region_index.names) == mi_names
        assert rebuilt_mi.region_index.equals(mi)

        m_mi_empty = m_mi.iloc[:0]
        rebuilt_mi_empty = eh.NormalizedMatrix.from_dict(m_mi_empty.to_dict())
        assert isinstance(rebuilt_mi_empty.region_index, pd.MultiIndex)
        assert rebuilt_mi_empty.region_index.nlevels == 2
        assert list(rebuilt_mi_empty.region_index.names) == mi_names
        assert rebuilt_mi_empty.region_index.equals(m_mi_empty.region_index)


def test_matrix_dataclass_init_and_static_contract():
    import copy
    import dataclasses
    import pickle
    from types import MappingProxyType

    assert dataclasses.is_dataclass(eh.NormalizedMatrix)
    expected_fields = [
        "values",
        "region_index",
        "mode",
        "upstream",
        "downstream",
        "bin_size",
        "target_bins",
        "name",
        "strand_oriented",
        "metadata",
        "centered",
    ]
    assert [f.name for f in dataclasses.fields(eh.NormalizedMatrix)] == expected_fields

    # Positional construction across all 11 arguments
    vals = [[1.0] * 10, [2.0] * 10]
    pos_mat = eh.NormalizedMatrix(
        vals,
        ("r0", "r1"),
        "reference_point",
        50,
        50,
        10,
        0,
        "ChIP",
        False,
        MappingProxyType({"unit": "rpm"}),
        False,
    )
    assert isinstance(pos_mat.values, np.ndarray)
    assert isinstance(pos_mat.region_index, pd.Index)
    assert list(pos_mat.region_index) == ["r0", "r1"]
    assert pos_mat.name == "ChIP"
    assert pos_mat.strand_oriented is False
    assert pos_mat.metadata == {"unit": "rpm"} and isinstance(pos_mat.metadata, dict)
    assert "bin_size=10 name='ChIP')" in repr(pos_mat)

    # None normalization for name & metadata, numpy array region_index, replace/deepcopy/pickle
    norm_mat = eh.NormalizedMatrix(
        np.ones((2, 10)),
        np.array(["x", "y"]),
        upstream=50,
        downstream=50,
        bin_size=10,
        name=None,
        metadata=None,
    )
    assert norm_mat.name == ""
    assert norm_mat.metadata == {}
    replaced = dataclasses.replace(norm_mat, name="Updated")
    assert replaced.name == "Updated" and list(replaced.region_index) == ["x", "y"]
    unpickled = pickle.loads(pickle.dumps(pos_mat))
    assert list(unpickled.region_index) == ["r0", "r1"] and unpickled.name == "ChIP"
    copied = copy.deepcopy(pos_mat)
    assert list(copied.region_index) == ["r0", "r1"]


def test_categorical_nan_and_single_row_nat_alignment():
    # Disjoint CategoricalIndex with NaN vs Int64(pd.NA) / UInt64(pd.NA) / boolean(pd.NA) / object(None)
    m_cat_disjoint = eh.NormalizedMatrix(
        np.full((2, 10), 10.0),
        region_index=pd.CategoricalIndex([9, np.nan], categories=[1, 2, 9]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_i64_na = eh.NormalizedMatrix(
        np.full((2, 10), 3.0),
        region_index=pd.Index([2, pd.NA], dtype="Int64"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_u64_na = eh.NormalizedMatrix(
        np.full((2, 10), 3.0),
        region_index=pd.Index([2, pd.NA], dtype="UInt64"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_bool_na = eh.NormalizedMatrix(
        np.full((2, 10), 3.0),
        region_index=pd.Index([True, pd.NA], dtype="boolean"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_obj_disjoint = eh.NormalizedMatrix(
        np.full((2, 10), 3.0),
        region_index=pd.Index([2, None], dtype=object),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    for other in (m_i64_na, m_u64_na, m_bool_na, m_obj_disjoint):
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            m_cat_disjoint.subtract(other)
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            other.log2_ratio(m_cat_disjoint)

    # Direct reindex never raises KeyError: nan; incompatible or NA-free sources produce all-NaN
    r_i64_to_cat = m_i64_na.reindex(m_cat_disjoint.region_index)
    assert np.isnan(r_i64_to_cat.values[0]).all()
    np.testing.assert_allclose(r_i64_to_cat.values[1], 3.0)
    assert np.isnan(m_bool_na.reindex(m_cat_disjoint.region_index).values).all()
    m_i64_no_na = eh.NormalizedMatrix(
        np.full((2, 10), 3.0),
        region_index=pd.Index([2, 5], dtype="Int64"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    assert np.isnan(m_i64_no_na.reindex(m_cat_disjoint.region_index).values).all()

    # Partial overlap on valid region 1 with NaN / pd.NA / None succeeds in both directions
    m_cat_overlap = eh.NormalizedMatrix(
        np.array([[10.0] * 10, [20.0] * 10]),
        region_index=pd.CategoricalIndex([1, np.nan], categories=[1, 2, 9]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_i64_perm = eh.NormalizedMatrix(
        np.array([[99.0] * 10, [3.0] * 10]),
        region_index=pd.Index([pd.NA, 1], dtype="Int64"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_obj_none = eh.NormalizedMatrix(
        np.array([[4.0] * 10, [88.0] * 10]),
        region_index=pd.Index([1, None], dtype=object),
        upstream=50,
        downstream=50,
        bin_size=10,
    )

    diff_ci = m_cat_overlap.subtract(m_i64_perm)
    np.testing.assert_allclose(diff_ci.values[0], 7.0)
    np.testing.assert_allclose(diff_ci.values[1], -79.0)
    diff_ic = m_i64_perm.subtract(m_cat_overlap)
    np.testing.assert_allclose(diff_ic.values[0], 79.0)
    np.testing.assert_allclose(diff_ic.values[1], -7.0)

    diff_co = m_cat_overlap.subtract(m_obj_none)
    np.testing.assert_allclose(diff_co.values[0], 6.0)
    np.testing.assert_allclose(diff_co.values[1], -68.0)
    diff_oc = m_obj_none.subtract(m_cat_overlap)
    np.testing.assert_allclose(diff_oc.values[0], -6.0)
    np.testing.assert_allclose(diff_oc.values[1], 68.0)

    np.testing.assert_allclose(m_i64_perm.reindex(m_cat_overlap.region_index).values[0], 3.0)
    np.testing.assert_allclose(m_cat_overlap.reindex(m_i64_perm.region_index).values[1], 10.0)
    np.testing.assert_allclose(m_obj_none.reindex(m_cat_overlap.region_index).values[0], 4.0)
    np.testing.assert_allclose(m_cat_overlap.reindex(m_obj_none.region_index).values[0], 10.0)

    s_i64 = pd.Series([False, True], index=pd.Index([pd.NA, 1], dtype="Int64"))
    s_obj = pd.Series([True, False], index=pd.Index([1, None], dtype=object))
    s_cat = pd.Series([True, False], index=m_cat_overlap.region_index)
    assert list(m_cat_overlap.loc[s_i64].region_index) == [1]
    assert list(m_cat_overlap.loc[s_obj].region_index) == [1]
    assert list(m_i64_perm.loc[s_cat].region_index) == [1]
    assert list(m_obj_none.loc[s_cat].region_index) == [1]

    # 1-row float64([NaN]) vs DatetimeIndex([NaT]) and TimedeltaIndex([NaT]) must raise ValueError
    m_f64_1 = eh.NormalizedMatrix(
        np.full((1, 10), 5.0),
        region_index=pd.Index([np.nan], dtype="float64"),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_dt_1 = eh.NormalizedMatrix(
        np.full((1, 10), 2.0),
        region_index=pd.DatetimeIndex([pd.NaT]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_td_1 = eh.NormalizedMatrix(
        np.full((1, 10), 2.0),
        region_index=pd.TimedeltaIndex([pd.NaT]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    for m_temporal in (m_dt_1, m_td_1):
        assert np.isnan(m_temporal.reindex(m_f64_1.region_index).values).all()
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            m_f64_1.subtract(m_temporal)
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            m_temporal.log2_ratio(m_f64_1)


def test_loc_rejects_unordered_containers_and_handles_exotic_scalars():
    class _RegionID:
        def __init__(self, label: str) -> None:
            self.label = label

        def __hash__(self) -> int:
            return hash(("_RegionID", self.label))

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _RegionID) and self.label == other.label

    class _Unhashable:
        __hash__ = None  # type: ignore[assignment]

    m2 = eh.NormalizedMatrix(
        np.array([[1.0] * 10, [2.0] * 10]),
        region_index=["r0", "r1"],
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    for bad_key in [
        {True, False},
        frozenset({True, False}),
        {"r0"},
        {"r0": 999},
        ...,
        bytearray(b"r0"),
        memoryview(b"r0"),
        _Unhashable(),
        [_Unhashable()],
        [["r0"]],
    ]:
        with pytest.raises(TypeError):
            _ = m2.loc[bad_key]

    # Non-NumPy scalar labels: Interval, Timestamp, Timedelta, Period, NaT, custom hashable object
    iv0, iv1, iv_miss = pd.Interval(0, 100), pd.Interval(100, 200), pd.Interval(200, 300)
    ts0, ts1, ts_miss = (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2025-01-01"),
    )
    td0, td1, td_miss = pd.Timedelta("1D"), pd.Timedelta("2D"), pd.Timedelta("9D")
    p0, p1, p_miss = (
        pd.Period("2020-01", freq="M"),
        pd.Period("2020-02", freq="M"),
        pd.Period("2025-01", freq="M"),
    )
    c0, c1, c_miss = _RegionID("r0"), _RegionID("r1"), _RegionID("r9")

    for idx_vals, hit_key, miss_key in [
        (pd.IntervalIndex([iv0, iv1]), iv0, iv_miss),
        (pd.DatetimeIndex([ts0, ts1]), ts0, ts_miss),
        (pd.TimedeltaIndex([td0, td1]), td0, td_miss),
        (pd.PeriodIndex([p0, p1], freq="M"), p0, p_miss),
        (pd.Index([c0, c1], dtype=object), c0, c_miss),
    ]:
        mat_s = eh.NormalizedMatrix(
            np.array([[10.0] * 10, [20.0] * 10]),
            region_index=idx_vals,
            upstream=50,
            downstream=50,
            bin_size=10,
        )
        selected = mat_s.loc[hit_key]
        assert selected.shape == (1, 10)
        np.testing.assert_allclose(selected.values, 10.0)
        with pytest.raises(KeyError):
            _ = mat_s.loc[miss_key]

    m_dt_nat = eh.NormalizedMatrix(
        np.array([[10.0] * 10, [20.0] * 10]),
        region_index=pd.DatetimeIndex([ts0, pd.NaT]),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    np.testing.assert_allclose(m_dt_nat.loc[pd.NaT].values, 20.0)
    with pytest.raises(KeyError):
        _ = m_dt_nat.loc[np.nan]

    # Scalar bool vs int hash collision on dtype=object region_index must raise KeyError
    m_obj_int = eh.NormalizedMatrix(
        np.array([[10.0] * 10, [20.0] * 10]),
        region_index=pd.Index([0, 1], dtype=object),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    m_obj_bool = eh.NormalizedMatrix(
        np.array([[10.0] * 10, [20.0] * 10]),
        region_index=pd.Index([False, True], dtype=object),
        upstream=50,
        downstream=50,
        bin_size=10,
    )
    for bool_k in (True, False, np.bool_(True), np.bool_(False)):
        with pytest.raises(KeyError):
            _ = m_obj_int.loc[bool_k]
    for int_k in (0, 1, 1.0, np.int64(1)):
        with pytest.raises(KeyError):
            _ = m_obj_bool.loc[int_k]
    assert list(m_obj_int.loc[1].region_index) == [1]
    assert list(m_obj_bool.loc[True].region_index) == [True]


