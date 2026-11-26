"""Bin tables, the pivot to matrices, and the pymisha / bigWig paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import enriched_heatmap as eh


def _regions():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [1037, 3000, 9950],
            "end": [1537, 3010, 9960],
            "strand": ["+", -1, "+"],
        },
        index=["a", "b", "c"],
    )


GENOME = {"chr1": 50_000, "chr2": 10_000}


def test_reference_bins_are_anchor_relative_not_on_a_global_grid():
    b = eh.bins(_regions(), upstream=200, downstream=100, bin_size=50, anchor="center")
    a = b[b.region == "a"]
    mid = (1037 + 1537) // 2
    assert a.start.tolist() == [mid - 200 + 50 * k for k in range(6)]
    assert a.bin.tolist() == list(range(6))


def test_minus_strand_bins_run_5_to_3():
    b = eh.bins(_regions(), upstream=200, downstream=100, bin_size=50, anchor="start")
    m = b[b.region == "b"].sort_values("bin")
    # TSS of a minus-strand region is its end; upstream lies to the genomic right.
    assert m.iloc[0].start == 3010 + 150 and m.iloc[0].end == 3010 + 200
    assert m.iloc[-1].end == 3010 - 50
    assert (np.diff(m.start.to_numpy()) < 0).all()


def test_genome_clips_bins_and_matrix_fills_nan():
    b = eh.bins(_regions(), upstream=200, downstream=100, bin_size=50, genome=GENOME)
    assert b.end.max() <= 50_000
    assert len(b[b.region == "c"]) == 4  # the last two bins run past chr2's end
    vals = b[["chrom", "start", "end"]].assign(x=1.0, intervalID=np.arange(1, len(b) + 1))
    m = eh.matrix(vals, b)
    assert m.shape == (3, 6)
    assert np.isnan(m.values[2, 4:]).all() and np.isfinite(m.values[2, :4]).all()


def test_chr_prefix_is_matched_to_genome():
    r = _regions().assign(chrom=["1", "1", "2"])
    b = eh.bins(r, upstream=100, downstream=100, bin_size=50, genome=GENOME)
    assert set(b.chrom) == {"chr1", "chr2"}
    with pytest.raises(KeyError):
        eh.bins(r.assign(chrom="chrM"), genome=GENOME)


def test_scale_regions_bins():
    r = _regions()
    b = eh.bins(r, upstream=100, downstream=200, bin_size=50, body_bins=5)
    a = b[b.region == "a"]
    assert a.bin.tolist() == list(range(2 + 5 + 4))
    body = a[(a.bin >= 2) & (a.bin < 7)]
    assert body.start.iloc[0] == 1037 and body.end.iloc[-1] == 1537
    # A 10 bp body is sampled at 1 bp; flanks keep full bins.
    small = b[b.region == "b"]
    assert (small.end - small.start).tolist().count(2) == 5
    # Zero-length bodies keep their flanks and leave body bins empty.
    zero = eh.bins(r.assign(end=r.start), upstream=100, downstream=200, bin_size=50, body_bins=5)
    assert sorted(zero[zero.region == "a"].bin) == [0, 1, 7, 8, 9, 10]


def test_matrices_by_interval_id_in_any_row_order():
    b = eh.bins(_regions(), upstream=100, downstream=100, bin_size=50)
    vals = b[["chrom", "start", "end"]].copy()
    vals["x"] = np.arange(len(b), dtype=float)
    vals["y"] = -vals["x"]
    vals["intervalID"] = np.arange(1, len(b) + 1)
    shuffled = vals.sample(frac=1.0, random_state=0)
    mats = eh.matrices(shuffled, b)
    assert set(mats) == {"x", "y"}
    np.testing.assert_array_equal(mats["x"].values[0], [0, 1, 2, 3])
    assert list(mats["x"].region_index) == ["a", "b", "c"]
    # Without intervalID the rows must line up with bins.
    np.testing.assert_array_equal(eh.matrix(vals.drop(columns="intervalID"), b, "x").values,
                                  mats["x"].values)
    with pytest.raises(ValueError):
        eh.matrix(vals.iloc[:-1].drop(columns="intervalID"), b, "x")
    # Same length, wrong order, no ids: must refuse rather than scramble.
    with pytest.raises(ValueError, match="row for row"):
        eh.matrix(shuffled.drop(columns="intervalID"), b, "x")
    # region/bin columns place rows in any order.
    by_label = shuffled.drop(columns="intervalID").assign(
        region=b["region"].to_numpy()[shuffled["intervalID"] - 1],
        bin=b["bin"].to_numpy()[shuffled["intervalID"] - 1])
    np.testing.assert_array_equal(eh.matrix(by_label, b, "x").values, mats["x"].values)
    with pytest.raises(ValueError):
        eh.matrix(vals, b)  # two value columns, none named


def test_snap_puts_anchor_stored_bin_in_the_middle_bin_on_both_strands():
    b = eh.bins(_regions(), upstream=100, downstream=100, bin_size=50, anchor="start", snap=50)
    assert (b.start % 50 == 0).all()
    assert b.groupby("region").size().max() == 2 + 1 + 2
    a = b[b.region == "a"]
    assert a.start.min() == 1000 - 100
    # true centre base 1013 (13 bp into stored bin [1000, 1020)), opposite strands
    regs = pd.DataFrame({"chrom": ["chr1"] * 2, "start": [1003, 1003], "end": [1024, 1024],
                         "strand": ["+", "-"]}, index=["plus", "minus"])
    for anchor, base in [("center", {"plus": 1013, "minus": 1013}),
                         ("start", {"plus": 1003, "minus": 1023}),
                         ("end", {"plus": 1023, "minus": 1003})]:
        for bin_size in (20, 60):
            bb = eh.bins(regs, upstream=120, downstream=120, bin_size=bin_size,
                         anchor=anchor, snap=20)
            n_up = 120 // bin_size
            for rid in regs.index:
                r = bb[bb.region == rid]
                mid = r[r.bin == n_up].iloc[0]
                assert mid.start <= base[rid] < mid.end, (anchor, bin_size, rid)
                # centred on the stored cell holding the anchor base
                cell = base[rid] - base[rid] % 20
                assert (mid.start + mid.end) // 2 == cell + 10
                assert len(r) == 2 * n_up + 1


def test_snap_end_boundary_on_stored_grid():
    # end % snap == 0: the last base (end - 1) is in the previous stored bin
    regs = pd.DataFrame({"chrom": ["chr1"], "start": [900], "end": [1000], "strand": ["+"]},
                        index=["g"])
    bb = eh.bins(regs, upstream=40, downstream=40, bin_size=20, anchor="end", snap=20)
    assert bb[bb.bin == 2].iloc[0].start == 980


def test_snap_validation_and_centred_matrix_geometry():
    r = _regions()
    for bad_snap in (0, -20, 0.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            eh.bins(r, upstream=100, downstream=100, bin_size=50, snap=bad_snap)
    with pytest.raises(ValueError, match="odd multiple"):
        eh.bins(r, upstream=120, downstream=120, bin_size=40, snap=20)
    with pytest.raises(ValueError, match="odd multiple"):
        eh.bins(r, upstream=90, downstream=90, bin_size=30, snap=20)
    with pytest.raises(ValueError, match="body_bins"):
        eh.bins(r, upstream=100, downstream=100, bin_size=50, body_bins=4, snap=50)
    b = eh.bins(r, upstream=100, downstream=100, bin_size=50, snap=50)
    vals = b.copy()
    vals["x"] = 1.0
    m = eh.matrix(vals, b, "x")
    assert m.centered and m.n_bins == 5
    assert m.x_extent == (-125.0, 125.0)
    np.testing.assert_allclose(m.bin_centers(), [-100, -50, 0, 50, 100])
    assert m.guide_positions() == [0.0]
    ticks, labels = m.default_xticks()
    assert ticks == [-100.0, 0.0, 100.0]
    for t in (m.log1p(), m.log2p(1), m.zscore(), m.smooth(1), m.clip_quantile(0, 1),
              m.loc[["a"]], m.iloc[[0]], m.reindex(["b", "a"]), m.dropna_regions(),
              m.subtract(m), m.log2_ratio(m), m.copy(),
              eh.NormalizedMatrix.from_dict(m.to_dict())):
        assert t.centered and t.x_extent == m.x_extent
    plain = eh.NormalizedMatrix(np.zeros((3, 4)), upstream=100, downstream=100, bin_size=50)
    with pytest.raises(ValueError):
        m.subtract(plain)
    with pytest.raises(ValueError, match="centered"):
        eh.NormalizedMatrix(np.zeros((3, 4)), upstream=100, downstream=100, bin_size=50,
                            centered=True)
    # center_mean includes the anchor bin; enriched_score is symmetric about it
    sym = eh.NormalizedMatrix(np.array([[1.0, 2, 5, 2, 1], [1.0, 2, 5, 2, 9]]),
                              upstream=100, downstream=100, bin_size=50, centered=True)
    assert eh.score(sym, "center_max").iloc[0] == 5
    flipped = eh.NormalizedMatrix(sym.values[:, ::-1].copy(), upstream=100, downstream=100,
                                  bin_size=50, centered=True)
    np.testing.assert_allclose(eh.score(sym, "enriched_score").values,
                               eh.score(flipped, "enriched_score").values)


def test_bins_validation():
    r = _regions()
    with pytest.raises(ValueError):
        eh.bins(r, upstream=110, bin_size=20)
    with pytest.raises(ValueError):
        eh.bins(r, anchor="tss")
    with pytest.raises(ValueError):
        eh.bins(pd.concat([r, r]))
    lost = eh.bins(r)
    lost.attrs.clear()
    with pytest.raises(ValueError, match="lost its layout"):
        eh.matrix(lost.assign(v=1.0), lost, "v")


def test_example_values_match_gextract_layout():
    peaks = eh.example_peaks(n=200)
    b = eh.bins(peaks, upstream=500, downstream=500, bin_size=50, genome=eh.example_genome())
    v = eh.example_values(b, ["chec", "k4me3"], peaks=peaks)
    assert list(v.columns) == ["chrom", "start", "end", "chec", "k4me3", "intervalID"]
    m = eh.matrices(v, b)
    bound = peaks["bound"].to_numpy()
    centre = eh.score(m["chec"]).to_numpy()
    assert centre[bound].mean() > 5 * centre[~bound].mean()


def test_read_bigwig(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    path = tmp_path / "x.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", 10_000)])
    bw.addEntries(["chr1"] * 4, [0, 1000, 1100, 1200], ends=[1000, 1100, 1200, 10_000],
                  values=[0.0, 1.0, 3.0, 0.0])
    bw.close()
    r = pd.DataFrame({"chrom": ["1"], "start": [1100], "end": [1100]})
    b = eh.bins(r, upstream=100, downstream=100, bin_size=50, anchor="start")
    v = eh.read_bigwig({"sig": path}, b)
    assert list(v.columns) == ["chrom", "start", "end", "sig", "intervalID"]
    np.testing.assert_allclose(eh.matrix(v, b).values[0], [1, 1, 3, 3])
    np.testing.assert_allclose(eh.read_bigwig(path, b, agg="max")["x"], [1, 1, 3, 3])


@pytest.fixture(scope="module")
def misha_db():
    pm = pytest.importorskip("pymisha")
    pm.gdb_init_examples()
    return pm


def test_pymisha_roundtrip_with_unaligned_overlapping_windows(misha_db):
    pm = misha_db
    regions = pd.DataFrame(
        {"chrom": ["2", "1", "1"], "start": [5037, 1037, 1061], "end": [5537, 1537, 1561],
         "strand": [1, -1, 1]},
        index=["x", "y", "z"],
    )
    b = eh.bins(regions, upstream=200, downstream=200, bin_size=20, genome=pm.gintervals_all())
    v = pm.gextract(["dense_track", "dense_track * 2"], intervals=b, iterator=b,
                    colnames=["d", "d2"])
    assert len(v) == len(b)
    mats = eh.matrices(v, b)
    np.testing.assert_allclose(mats["d2"].values, 2 * mats["d"].values, rtol=1e-5)
    # Each bin's value matches a direct extraction of that bin alone.
    row = b[(b.region == "y") & (b.bin == 3)].iloc[0]
    one = pd.DataFrame({"chrom": [row.chrom], "start": [row.start], "end": [row.end]})
    direct = pm.gextract("dense_track", intervals=one, iterator=one)["dense_track"].iloc[0]
    assert mats["d"].values[1, 3] == pytest.approx(direct, rel=1e-5)


def test_pymisha_intervals_join_matches_default(misha_db):
    pm = misha_db
    regions = pd.DataFrame({"chrom": ["2", "1"], "start": [5000, 1037], "end": [5000, 1037]},
                           index=["p", "q"])
    b = eh.bins(regions, upstream=100, downstream=100, bin_size=50, genome=pm.gintervals_all())
    default = eh.matrix(pm.gextract("dense_track", intervals=b, iterator=b), b)
    joined = eh.matrix(pm.gextract("dense_track", intervals=b, iterator=b,
                                   intervals_join="intervals"), b, "dense_track")
    np.testing.assert_array_equal(joined.values, default.values)
    with pytest.raises(ValueError, match="row for row"):
        eh.matrix(pm.gextract("dense_track", intervals=b, iterator=b, intervals_join="none"), b)


# ---------------------------------------------------------------------------
# Regressions: input validation, layout metadata, dtype
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("col", ["chrom", "start", "end"])
def test_bins_rejects_missing_coordinates(col):
    r = _regions().astype({"chrom": object, "start": float, "end": float})
    r.loc["b", col] = np.nan
    with pytest.raises(ValueError, match=f"missing values in '{col}'"):
        eh.bins(r)


def test_snap_point_intervals_agree_across_strands():
    regs = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [1000, 1000], "end": [1000, 1000], "strand": ["+", "-"]},
        index=["plus", "minus"],
    )
    for anchor in ("center", "start", "end"):
        b = eh.bins(regs, upstream=40, downstream=40, bin_size=20, anchor=anchor, snap=20)
        plus_mid = b[(b.region == "plus") & (b.bin == 2)].iloc[0]
        minus_mid = b[(b.region == "minus") & (b.bin == 2)].iloc[0]
        assert plus_mid.start == minus_mid.start == 1000


def test_matrices_rejects_out_of_range_bin_and_chrom_mismatch():
    regs = pd.DataFrame(
        {"chrom": ["chr2", "chr1"], "start": [1000, 1000], "end": [1000, 1000]},
        index=["on_chr2", "on_chr1"],
    )
    b = eh.bins(regs, upstream=20, downstream=20, bin_size=20)
    with pytest.raises(ValueError, match="bin"):
        eh.matrix(pd.DataFrame({"region": ["on_chr2"], "bin": [-1], "v": [1.0]}), b, "v")
    with pytest.raises(ValueError, match="bin"):
        eh.matrix(pd.DataFrame({"region": ["on_chr2"], "bin": [2], "v": [1.0]}), b, "v")

    swapped = b.sort_values("chrom").drop(columns=["region", "bin"]).reset_index(drop=True)
    swapped["v"] = [1.0, 1.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="row for row"):
        eh.matrix(swapped, b, "v")


def test_example_values_allows_one_sided_boundary_windows():
    peaks = eh.example_peaks(n=10)
    for anchor, up, down in [("end", 0, 200), ("start", 200, 0)]:
        b = eh.bins(peaks, upstream=up, downstream=down, bin_size=20, anchor=anchor)
        v = eh.example_values(b, ["chec"], peaks=peaks)
        assert len(v) == len(b)


def test_bins_layout_survives_pandas_operations():
    import copy

    r = _regions()
    b = eh.bins(r, upstream=100, downstream=100, bin_size=50, genome=GENOME)
    vals = b.assign(v=np.arange(len(b), dtype=float), intervalID=np.arange(1, len(b) + 1))
    ref = eh.matrix(vals, b, "v").values

    # Whether merge/concat carry attrs depends on the pandas version (3.x does, 2.x does
    # not). Either way they must not raise; with pandas 3 merge used to fail comparing
    # the region Index stored in attrs.
    merged = b.merge(b[["region"]].drop_duplicates(), on="region")
    if merged.attrs:
        assert np.array_equal(eh.matrix(vals, merged, "v").values, ref, equal_nan=True)
    else:
        with pytest.raises(ValueError, match="lost its layout"):
            eh.matrix(vals, merged, "v")
    assert len(pd.concat([b, b])) == 2 * len(b)
    # copies and reorderings keep it on every supported pandas
    assert eh.matrix(vals, copy.deepcopy(b), "v").values.shape == ref.shape
    assert eh.matrix(vals, b.copy(), "v").values.shape == ref.shape
    assert b.sort_values("start").attrs == b.attrs


def test_bins_keeps_region_index_name():
    r = _regions()
    r.index.name = "peak"
    b = eh.bins(r, upstream=100, downstream=100, bin_size=50)
    assert eh.matrix(b.assign(v=1.0), b, "v").region_index.name == "peak"


def test_strand_orientation_is_recorded():
    r = _regions()
    b = eh.bins(r, upstream=100, downstream=100, bin_size=50)
    v = b.assign(v=1.0)
    assert eh.matrix(v, b, "v").strand_oriented is True
    flat = eh.bins(r, upstream=100, downstream=100, bin_size=50, strand=None)
    assert eh.matrix(flat.assign(v=1.0), flat, "v").strand_oriented is False
    no_col = eh.bins(r.drop(columns="strand"), upstream=100, downstream=100, bin_size=50)
    assert eh.matrix(no_col.assign(v=1.0), no_col, "v").strand_oriented is False


def test_matrices_dtype_keeps_large_counts_exact():
    b = eh.bins(_regions(), upstream=100, downstream=100, bin_size=50)
    v = b.assign(v=16_777_217.0)
    assert eh.matrix(v, b, "v").values.dtype == np.float32
    exact = eh.matrix(v, b, "v", dtype=np.float64)
    assert exact.values.dtype == np.float64
    assert np.nanmax(exact.values) == 16_777_217.0
    assert np.nanmax(eh.matrices(v, b, dtype=np.float64)["v"].values) == 16_777_217.0


def test_example_peaks_size_limits():
    assert len(eh.example_peaks(n=1000)) == 1000
    with pytest.raises(ValueError, match="room for about"):
        eh.example_peaks(n=2500)
    with pytest.raises(ValueError, match="positive"):
        eh.example_peaks(n=0)


def test_example_values_reject_bins_from_other_peaks():
    custom = eh.example_peaks(n=300, seed=3)
    b = eh.bins(custom, upstream=500, downstream=500, bin_size=50)
    with pytest.raises(ValueError, match="peaks=peaks"):
        eh.example_values(b)  # would silently use the 1,500 default peaks
    out = eh.example_values(b, ["chec"], peaks=custom)
    assert len(out) == len(b)


def test_public_names_are_not_modules():
    import types

    for name in eh.__all__:
        assert not isinstance(getattr(eh, name), types.ModuleType), name
    # eh.bins / eh.matrix are functions, so their modules must not share the name
    import importlib

    for name in ("bins", "matrix"):
        with pytest.raises(ImportError):
            importlib.import_module(f"enriched_heatmap.{name}")


# ---------------------------------------------------------------------------
# M1: Duplicate & pre-cast index validation (F01), strict bins params (F16),
#     and read_bigwig chrom-end NaN parity & paths validation (F17)
# ---------------------------------------------------------------------------
def test_matrices_rejects_duplicate_cells_across_all_branches():
    b = eh.bins(_regions(), upstream=100, downstream=100, bin_size=50)
    n_bins = b.attrs["enriched_heatmap"]["n_bins"]

    # Branch 2 (intervalID): duplicate intervalID=1 (e.g. omitted iterator=b)
    dup_id = pd.DataFrame({"intervalID": [1, 1, 2], "v": [1.0, 99.0, 2.0]})
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrices(dup_id, b)
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrix(dup_id, b, "v")

    # Branch 1 (explicit region + bin columns): duplicate ("a", 0)
    dup_rb = pd.DataFrame({"region": ["a", "a", "b"], "bin": [0, 0, 1], "v": [1.0, 5.0, 2.0]})
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrix(dup_rb, b, "v")

    # Branch 3 (positional row-for-row) and Branch 2 with concatenated bins
    b_dup = pd.concat([b, b], ignore_index=True)
    b_dup.attrs = dict(b.attrs)
    pos_vals = b_dup[["chrom", "start", "end"]].assign(v=1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrix(pos_vals, b_dup, "v")
    id_vals = pos_vals.assign(intervalID=np.arange(1, len(b_dup) + 1))
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrix(id_vals, b_dup, "v")

    # Out-of-bounds bin checked BEFORE r * n_bins + b (where r=0, b=n_bins aliases r=1, b=0)
    oob_alias = pd.DataFrame({"region": ["a", "b"], "bin": [n_bins, 0], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"0\.\."):
        eh.matrix(oob_alias, b, "v")

    # Tampered bins["region"] or bins["bin"] in Branch 2 / Branch 3 also caught
    b_bad_reg = b.copy()
    b_bad_reg["region"] = "unknown_region"
    with pytest.raises(ValueError, match="not in its layout"):
        eh.matrix(b_bad_reg[["chrom", "start", "end"]].assign(v=1.0), b_bad_reg, "v")

    b_bad_bin = b.copy()
    b_bad_bin["bin"] = n_bins
    with pytest.raises(ValueError, match=r"0\.\."):
        eh.matrix(b_bad_bin[["chrom", "start", "end"]].assign(v=1.0), b_bad_bin, "v")


def test_matrices_validates_bin_and_interval_id_before_int64_cast():
    import warnings

    b = eh.bins(_regions(), upstream=100, downstream=100, bin_size=50)

    bad_columns = [
        pd.Series([0.0, np.nan]),
        pd.Series([0, pd.NA], dtype="Int64"),
        pd.Series([0.0, pd.NA], dtype="Float64"),
        pd.Series([1, None], dtype=object),
        pd.Series([1.0, np.inf]),
        pd.Series([1.0, -np.inf]),
        pd.Series([True, False]),
        pd.Series([True, False], dtype="boolean"),
        pd.Series(pd.Categorical([True, False])),
        pd.Series([1, True], dtype=object),
        pd.Series([0.0, 0.5]),
        pd.Series([-0.5, 1.0]),
        pd.Series([1.9, 2.0], dtype="Float64"),
        pd.Series(pd.Categorical([0.0, 1.5])),
        pd.Series([1, 1.5], dtype=object),
        pd.Series([1e20, 1.0]),
        pd.Series([float(1 << 63), 1.0]),
        pd.Series([float(1 << 54), 1.0]),
        pd.Series([-float(1 << 54), 1.0]),
        pd.Series([1, float(1 << 54)], dtype=object),
        pd.Series(np.array([1 << 63, 1], dtype=np.uint64)),
        pd.Series([1 << 63, 1], dtype="UInt64"),
        pd.Series([1 << 100, 1], dtype=object),
        pd.Series([-(1 << 100), 1], dtype=object),
        pd.Series(["1", "2"]),
        pd.Series(pd.Categorical(["1", "2"])),
        pd.Series([object(), object()], dtype=object),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for bad_s in bad_columns:
            df_bin = pd.DataFrame({"region": ["a", "b"], "bin": bad_s, "v": [1.0, 2.0]})
            with pytest.raises(ValueError):
                eh.matrix(df_bin, b, "v")

            df_id = pd.DataFrame({"intervalID": bad_s, "v": [1.0, 2.0]})
            with pytest.raises(ValueError):
                eh.matrix(df_id, b, "v")

        # Empty StringDtype index column is rejected; intervalID bounds checked before - 1
        with pytest.raises(ValueError):
            eh.matrix(
                pd.DataFrame(
                    {
                        "region": pd.Series([], dtype="str"),
                        "bin": pd.Series([], dtype="str"),
                        "v": pd.Series([], dtype=float),
                    }
                ),
                b,
                "v",
            )
        for bad_id in [0, -1, int(np.iinfo(np.int64).min), len(b) + 1]:
            with pytest.raises(ValueError, match="intervalID"):
                eh.matrix(pd.DataFrame({"intervalID": [bad_id], "v": [1.0]}), b, "v")

        # Native-precision np.longdouble checks (fractional, overflow, int64 bounds, > 2**53 exactness)
        from enriched_heatmap._bins import _as_int64_index

        frac_ld = np.longdouble(1) + np.longdouble(2) ** -60
        huge_ld = np.longdouble(10.0) ** 1000
        ld_bad_series = [
            pd.Series(np.array([np.longdouble("inf"), np.longdouble(1.0)], dtype=np.longdouble)),
            pd.Series(np.array([-np.longdouble("inf"), np.longdouble(1.0)], dtype=np.longdouble)),
            pd.Series(np.array([np.longdouble(1 << 63), np.longdouble(1.0)], dtype=np.longdouble)),
            pd.Series([np.longdouble(1 << 63), np.longdouble(1.0)], dtype=object),
        ]
        if frac_ld != np.floor(frac_ld):
            ld_bad_series.extend(
                [
                    pd.Series(np.array([frac_ld, np.longdouble(2.0)], dtype=np.longdouble)),
                    pd.Series([frac_ld, np.longdouble(2.0)], dtype=object),
                    pd.Series(
                        pd.Categorical.from_codes(
                            [0, 1], categories=pd.Index([frac_ld, np.longdouble(2.0)], dtype=object)
                        )
                    ),
                    pd.Series(pd.arrays.SparseArray(np.array([frac_ld, np.longdouble(2.0)], dtype=np.longdouble))),
                ]
            )
        if np.isfinite(huge_ld):
            ld_bad_series.extend(
                [
                    pd.Series(np.array([huge_ld, np.longdouble(1.0)], dtype=np.longdouble)),
                    pd.Series(np.array([-huge_ld, np.longdouble(1.0)], dtype=np.longdouble)),
                    pd.Series([huge_ld, np.longdouble(1.0)], dtype=object),
                ]
            )
        for bad_ld in ld_bad_series:
            with pytest.raises(ValueError):
                eh.matrix(pd.DataFrame({"region": ["a", "b"], "bin": bad_ld, "v": [1.0, 2.0]}), b, "v")
            with pytest.raises(ValueError):
                eh.matrix(pd.DataFrame({"intervalID": bad_ld, "v": [1.0, 2.0]}), b, "v")

        if int(np.finfo(np.longdouble).nmant) + 1 >= 63:
            odd_ld = np.longdouble(1 << 53) + np.longdouble(1)
            i64_max_ld = np.longdouble((1 << 63) - 1)
            i64_min_ld = np.longdouble(-(1 << 63))
            got_ld = _as_int64_index(
                pd.Series(np.array([odd_ld, i64_max_ld, i64_min_ld], dtype=np.longdouble)), "bin"
            )
            assert got_ld.tolist() == [(1 << 53) + 1, (1 << 63) - 1, -(1 << 63)]
            got_obj_ld = _as_int64_index(
                pd.Series([odd_ld, i64_max_ld, i64_min_ld], dtype=object), "bin"
            )
            assert got_obj_ld.tolist() == [(1 << 53) + 1, (1 << 63) - 1, -(1 << 63)]

    # Exact integer-valued floats, nullable Int64/Float64, Categorical[int], and 0-row frames succeed
    ref_df = pd.DataFrame({"region": ["a", "a", "b"], "bin": [0, 1, 2], "v": [3.0, 4.0, 5.0]})
    ref_mat = eh.matrix(ref_df, b, "v")
    for bin_col in [
        pd.Series([0.0, 1.0, 2.0]),
        pd.Series([0.0, 1.0, 2.0], dtype="Float64"),
        pd.Series(np.array([0.0, 1.0, 2.0], dtype=np.longdouble)),
        pd.Series([np.longdouble(0.0), np.longdouble(1.0), np.longdouble(2.0)], dtype=object),
        pd.Series([0, 1, 2], dtype="Int64"),
        pd.Series(pd.Categorical([0, 1, 2])),
        pd.Series(pd.Categorical([0.0, 1.0, 2.0])),
        pd.Series([0.0, 1.0, 2.0], dtype=object),
    ]:
        got = eh.matrix(ref_df.assign(bin=bin_col), b, "v")
        np.testing.assert_array_equal(got.values, ref_mat.values)

    float_id = pd.DataFrame({"intervalID": [1.0, 2.0, 3.0], "v": [10.0, 20.0, 30.0]})
    int_id = pd.DataFrame({"intervalID": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    np.testing.assert_array_equal(
        eh.matrix(float_id, b, "v").values,
        eh.matrix(int_id, b, "v").values,
    )

    # Empty 0-row values and 0-region / all-off-chromosome bins
    for empty_vals in [
        ref_df.iloc[0:0],
        pd.DataFrame(columns=["region", "bin", "v"]),
        int_id.iloc[0:0],
    ]:
        m_empty = eh.matrix(empty_vals, b, "v")
        assert m_empty.shape == (3, 4) and np.isnan(m_empty.values).all()

    b_zero_reg = eh.bins(_regions().iloc[0:0], upstream=100, downstream=100, bin_size=50)
    m_zero_reg = eh.matrix(pd.DataFrame(columns=["intervalID", "v"]), b_zero_reg, "v")
    assert m_zero_reg.shape == (0, 4)

    off_chrom = pd.DataFrame({"chrom": ["chr2"], "start": [20_000], "end": [20_010]}, index=["z"])
    b_off = eh.bins(off_chrom, upstream=50, downstream=50, bin_size=50, genome=GENOME)
    assert len(b_off) == 0
    m_off = eh.matrix(b_off.assign(v=pd.Series([], dtype=float)), b_off, "v")
    assert m_off.shape == (1, 2) and np.isnan(m_off.values).all()


def test_pymisha_without_iterator_raises_duplicate_interval_id(misha_db):
    pm = misha_db
    regions = pd.DataFrame(
        {"chrom": ["1", "2"], "start": [1050, 5050], "end": [1050, 5050]},
        index=["p", "q"],
    )
    b = eh.bins(regions, upstream=100, downstream=100, bin_size=50, genome=pm.gintervals_all())
    v_sub = pm.gextract("dense_track", intervals=b, iterator=20)
    assert len(v_sub) > len(b)
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrix(v_sub, b)
    with pytest.raises(ValueError, match="Duplicate"):
        eh.matrices(v_sub, b)


def test_bins_strict_integer_params_and_non_empty_layout():
    r = _regions()

    for bad in (True, False, np.bool_(True), 20.0, 0.5, 0, -10, "20"):
        with pytest.raises(ValueError, match="positive integer"):
            eh.bins(r, upstream=100, downstream=100, bin_size=bad)
        with pytest.raises(ValueError, match="positive integer"):
            eh.bins(r, upstream=100, downstream=100, bin_size=50, body_bins=bad)
        with pytest.raises(ValueError, match="positive integer"):
            eh.bins(r, upstream=100, downstream=100, bin_size=50, snap=bad)

    for bad_flank in (True, False, np.bool_(False), 100.0, -50, "100"):
        with pytest.raises(ValueError, match="non-negative"):
            eh.bins(r, upstream=bad_flank, downstream=100, bin_size=50)
        with pytest.raises(ValueError, match="non-negative"):
            eh.bins(r, upstream=100, downstream=bad_flank, bin_size=50)

    # NumPy integer scalars are accepted
    b_np = eh.bins(
        r,
        upstream=np.int64(100),
        downstream=np.int32(100),
        bin_size=np.int64(50),
        body_bins=np.int32(4),
    )
    assert b_np.attrs["enriched_heatmap"]["n_bins"] == 8

    # n_bins == 0 rejected (even on 0-row regions)
    for regs in (r, r.iloc[0:0]):
        with pytest.raises(ValueError, match="0 bins"):
            eh.bins(regs, upstream=0, downstream=0, bin_size=50)

    # Valid zero-flank combinations succeed
    b_right = eh.bins(r, upstream=0, downstream=200, bin_size=50)
    assert eh.matrix(b_right.assign(v=1.0), b_right, "v").shape == (3, 4)

    b_left = eh.bins(r, upstream=200, downstream=0, bin_size=50)
    assert eh.matrix(b_left.assign(v=1.0), b_left, "v").shape == (3, 4)

    b_body_only = eh.bins(r, upstream=0, downstream=0, bin_size=50, body_bins=5)
    m_body_only = eh.matrix(b_body_only.assign(v=1.0), b_body_only, "v")
    assert m_body_only.shape == (3, 5) and m_body_only.mode == "scale_regions"

    b_snap_only = eh.bins(r, upstream=0, downstream=0, bin_size=20, snap=20)
    m_snap_only = eh.matrix(b_snap_only.assign(v=1.0), b_snap_only, "v")
    assert m_snap_only.shape == (3, 1)
    assert m_snap_only.centered is True
    assert m_snap_only.x_extent == (-10.0, 10.0)

    # Clarified docstring & _layout guidance
    assert eh.bins.__doc__ is not None and "eh.arrange" in eh.bins.__doc__
    lost = eh.bins(r)
    lost.attrs.clear()
    with pytest.raises(ValueError, match=r"lost its layout.*eh\.arrange"):
        eh.matrix(lost.assign(v=1.0), lost, "v")


def test_read_bigwig_paths_validation_and_chrom_end_nan_parity(tmp_path):
    import os
    from types import MappingProxyType

    pyBigWig = pytest.importorskip("pyBigWig")
    path = tmp_path / "track.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries(["chr1"], [0], ends=[1000], values=[2.0])
    bw.close()

    class _StrPath:
        def __fspath__(self) -> str:
            return str(path)

    class _BytesPath:
        def __fspath__(self) -> bytes:
            return os.fsencode(path)

    r = pd.DataFrame({"chrom": ["chr1"], "start": [975], "end": [975]}, index=["edge"])
    b_raw = eh.bins(r, upstream=100, downstream=100, bin_size=50, anchor="start")
    b_gen = eh.bins(
        r, upstream=100, downstream=100, bin_size=50, anchor="start", genome={"chr1": 1000}
    )

    # Upfront paths type & emptiness checks
    for bad_paths in [
        [path],
        (path,),
        {path},
        123,
        None,
        b"track.bw",
        _BytesPath(),
        {"sig": [path]},
        {"sig": None},
        {"sig": b"track.bw"},
        {"sig": _BytesPath()},
    ]:
        with pytest.raises(TypeError):
            eh.read_bigwig(bad_paths, b_raw)

    for empty_map in [{}, MappingProxyType({})]:
        with pytest.raises(ValueError, match="empty"):
            eh.read_bigwig(empty_map, b_raw)

    # Custom os.PathLike[str] and MappingProxyType accepted
    v_pathlike = eh.read_bigwig(_StrPath(), b_gen)
    v_proxy = eh.read_bigwig(MappingProxyType({"track": _StrPath()}), b_gen)
    np.testing.assert_array_equal(v_pathlike["track"].to_numpy(), v_proxy["track"].to_numpy())

    # Exact parity with bins(..., genome=...) across all 4 agg modes on straddling [975, 1025) bin
    expected_by_agg = {
        "mean": [2.0, 2.0, np.nan, np.nan],
        "sum": [100.0, 100.0, np.nan, np.nan],
        "max": [2.0, 2.0, np.nan, np.nan],
        "min": [2.0, 2.0, np.nan, np.nan],
    }
    for agg, expected_row in expected_by_agg.items():
        m_raw = eh.matrix(eh.read_bigwig(str(path), b_raw, agg=agg), b_raw)
        m_gen = eh.matrix(eh.read_bigwig(path, b_gen, agg=agg), b_gen)
        np.testing.assert_allclose(m_raw.values, m_gen.values, equal_nan=True)
        np.testing.assert_allclose(m_raw.values[0], expected_row, equal_nan=True)

    # Filtered non-RangeIndex bins, straddling 0, negative, zero-width, unknown chrom, and 0-row
    filtered = b_raw.iloc[[0, 2]]  # non-contiguous index [0, 2], second row straddles 1000
    v_filt = eh.read_bigwig(path, filtered, agg="sum")
    assert list(v_filt.index) == [0, 2]
    np.testing.assert_allclose(v_filt["track"].to_numpy(), [100.0, np.nan], equal_nan=True)

    custom_bins = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr2", "chrUn"],
            "start": [-50, -25, 100, 100, 1200, 10],
            "end": [-10, 25, 100, 150, 1250, 60],
            "region": ["r0", "r0", "r0", "r0", "r1", "r2"],
            "bin": [0, 1, 2, 3, 0, 0],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    v_custom = eh.read_bigwig(path, custom_bins, agg="sum")
    np.testing.assert_allclose(
        v_custom["track"].to_numpy(),
        [np.nan, np.nan, np.nan, 100.0, np.nan, np.nan],
        equal_nan=True,
    )

    v_empty = eh.read_bigwig({"sig": path}, b_raw.iloc[0:0])
    assert list(v_empty.columns) == ["chrom", "start", "end", "sig", "intervalID"]
    assert len(v_empty) == 0

