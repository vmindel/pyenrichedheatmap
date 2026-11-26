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
# Region identifiers, alignment and the .loc / .iloc contract
# ---------------------------------------------------------------------------
def _ids_mat(region_index, values=None, n_bins=10):
    n = len(region_index)
    vals = np.ones((n, n_bins)) if values is None else values
    return eh.NormalizedMatrix(vals, region_index=region_index, upstream=50, downstream=50, bin_size=10)


def test_region_index_rejects_missing_and_boolean_identifiers():
    for bad in [
        pd.Index(["a", None], dtype=object),
        pd.Index(["a", np.nan], dtype=object),
        pd.Index([1.0, np.nan]),
        pd.Index([1, pd.NA], dtype="Int64"),
        pd.Index(["x", pd.NA], dtype="string"),
        pd.CategoricalIndex(["a", None]),
        pd.DatetimeIndex(["2020-01-01", pd.NaT]),
        pd.MultiIndex.from_tuples([("chr1", 1), ("chr1", np.nan)]),
    ]:
        with pytest.raises(ValueError, match="missing"):
            _ids_mat(bad)
    for bad in [
        pd.Index([False, True]),
        pd.Index([False, True], dtype="boolean"),
        pd.Index([False, True], dtype=object),
        pd.CategoricalIndex([False, True]),
    ]:
        with pytest.raises(ValueError, match="boolean"):
            _ids_mat(bad)
    with pytest.raises(ValueError, match="unique"):
        _ids_mat(pd.Index(["a", "a"]))

    regions = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 500], "end": [100, 500],
                            "name": ["p1", None]})
    with pytest.raises(ValueError, match="missing"):
        eh.bins(regions, upstream=50, downstream=50, bin_size=10, id="name")


def test_differential_ops_reject_zero_overlap_and_allow_partial_or_empty():
    a = _make_mat(n_regions=3, n_bins=10, seed=10)
    b_range = _ids_mat(pd.RangeIndex(3))
    b_disjoint = _ids_mat(pd.Index(["other_0", "other_1"]))
    mi2 = _ids_mat(pd.MultiIndex.from_tuples([("chr1", 10), ("chr1", 20)]))
    mi3 = _ids_mat(pd.MultiIndex.from_tuples([("chr1", 10, "+"), ("chr1", 20, "-")]))

    # mi2 vs mi3: pandas would match the deeper MultiIndex by prefix
    for left, right in [(a, b_range), (a, b_disjoint), (mi2, b_range), (mi2, mi3), (mi3, mi2)]:
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            left.subtract(right)
        with pytest.raises(ValueError, match="shares no region identifiers with"):
            left.log2_ratio(right)

    # Partial overlap aligns shared IDs and leaves unmatched rows as NaN
    b_partial = _ids_mat(pd.Index(["peak_1", "unseen"]), values=np.full((2, 10), 2.0))
    diff = a.subtract(b_partial)
    np.testing.assert_allclose(diff.values[1], a.values[1] - 2.0)
    assert np.isnan(diff.values[0]).all() and np.isnan(diff.values[2]).all()
    lfc = a.log2_ratio(b_partial, pseudocount=1.0)
    np.testing.assert_allclose(lfc.values[1], np.log2((a.values[1] + 1.0) / 3.0), rtol=1e-5)

    # Equal labels in another dtype still align (object ints vs int64)
    obj = _ids_mat(pd.Index([0, 1, 2], dtype=object), values=np.full((3, 10), 2.0))
    np.testing.assert_allclose(_ids_mat(pd.RangeIndex(3)).subtract(obj).values, -1.0)

    # 0-row inputs remain allowed on either or both sides
    assert a.iloc[:0].subtract(b_disjoint).shape == (0, 10)
    empty_rhs = a.subtract(b_disjoint.iloc[:0])
    assert empty_rhs.shape == (3, 10) and np.isnan(empty_rhs.values).all()

    # Direct reindex stays permissive, including MultiIndex depth mismatches
    for target in (["unseen_0", "unseen_1"], ("unseen_0", "unseen_1"), np.array(["u0", "u1"])):
        assert np.isnan(a.reindex(target).values).all()
    assert np.isnan(mi3.reindex(mi2.region_index).values).all()
    assert np.isnan(mi2.reindex(mi3.region_index).values).all()


def test_loc_contract():
    mat = _make_mat(n_regions=4, n_bins=10)
    assert list(mat.loc["peak_2"].region_index) == ["peak_2"]
    assert list(mat.loc[["peak_3", "peak_0"]].region_index) == ["peak_3", "peak_0"]
    assert list(mat.loc["peak_1":"peak_2"].region_index) == ["peak_1", "peak_2"]
    assert list(mat.loc[(x for x in ["peak_1"])].region_index) == ["peak_1"]
    with pytest.raises(KeyError, match="not found"):
        _ = mat.loc[["peak_0", "nope"]]
    with pytest.raises(KeyError):
        _ = mat.loc["nope"]

    # Masks: positional for arrays/lists, by label for a boolean Series
    for mask in [np.array([True, False, True, False]), [True, False, True, False],
                 pd.array([True, False, True, False], dtype="boolean")]:
        assert list(mat.loc[mask].region_index) == ["peak_0", "peak_2"]
    s = pd.Series([True, False, False, True], index=["peak_3", "peak_2", "peak_1", "peak_0"])
    assert list(mat.loc[s].region_index) == ["peak_0", "peak_3"]
    with pytest.raises(KeyError, match="Boolean Series index"):
        _ = mat.loc[pd.Series([True] * 4)]
    with pytest.raises(ValueError, match="NA"):
        _ = mat.loc[[True, None, False, True]]
    with pytest.raises(ValueError, match="length"):
        _ = mat.loc[[True, False]]

    for bad in [("peak_0", "peak_1"), np.array([["peak_0"]])]:
        with pytest.raises(IndexError):
            _ = mat.loc[bad]
    for bad in [{"peak_0"}, {"peak_0": 1}, ..., pd.DataFrame({"x": [True] * 4}), [["peak_0"]]]:
        with pytest.raises(TypeError):
            _ = mat.loc[bad]

    # True would hash-match 1 on an integer index; region IDs are never booleans
    ints = _ids_mat(pd.Index([0, 1], dtype=object))
    with pytest.raises(KeyError):
        _ = ints.loc[True]
    assert list(ints.loc[1].region_index) == [1]

    # A MultiIndex row is selected by its full tuple
    mi = _ids_mat(pd.MultiIndex.from_tuples([("chr1", 10), ("chr1", 20)]))
    assert list(mi.loc[("chr1", 20)].region_index) == [("chr1", 20)]
    assert list(mi.loc[[("chr1", 20), ("chr1", 10)]].region_index) == [("chr1", 20), ("chr1", 10)]

    # Non-string scalar labels, and no containment match on an IntervalIndex
    iv = _ids_mat(pd.IntervalIndex.from_tuples([(0, 100), (100, 200)]))
    assert iv.loc[pd.Interval(0, 100)].n_regions == 1
    with pytest.raises(KeyError):
        _ = iv.loc[50]
    assert np.isnan(iv.reindex([50]).values).all()
    ts = _ids_mat(pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))
    assert ts.loc[pd.Timestamp("2020-01-02")].n_regions == 1


def test_iloc_contract():
    mat = _make_mat(n_regions=4, n_bins=10)
    assert list(mat.iloc[-1].region_index) == ["peak_3"]
    assert list(mat.iloc[np.uint8(1)].region_index) == ["peak_1"]
    assert list(mat.iloc[1:3].region_index) == ["peak_1", "peak_2"]
    for idx in [[3, 0, -2], np.array([3, 0, 2]), pd.Index([3, 0, 2]), range(3, -1, -3),
                (i for i in [3, 0])]:
        assert mat.iloc[idx].region_index[0] == "peak_3"
    for empty in [[], range(0), np.array([], dtype=int)]:
        assert mat.iloc[empty].shape == (0, 10)
    for mask in [[True, False, True, False], np.array([True, False, True, False]),
                 pd.Series([True, False, True, False], index=list("wxyz"))]:
        assert list(mat.iloc[mask].region_index) == ["peak_0", "peak_2"]

    for oob in [4, -5, 10**30, [0, 4], np.array([1 << 63], dtype=np.uint64), range(0, 10**30)]:
        with pytest.raises(IndexError):
            _ = mat.iloc[oob]
    for bad in [True, np.bool_(False), "2", 1.0, None, slice(0, 1.5), slice(0, True),
                [1.0], ["2"], [0, True], np.array(["2"]), pd.Series([0, pd.NA], dtype="Int64"),
                {0, 1}, {"a": 0}, b"01", pd.MultiIndex.from_tuples([("a", 0)])]:
        with pytest.raises(TypeError):
            _ = mat.iloc[bad]
    with pytest.raises(IndexError):
        _ = mat.iloc[(0, 0)]
    with pytest.raises(ValueError, match="length"):
        _ = mat.iloc[[True, False]]
    with pytest.raises(ValueError, match="NA"):
        _ = mat.iloc[pd.array([True, pd.NA, False, True], dtype="boolean")]


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
