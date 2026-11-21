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


def test_snap_rounds_anchors_down():
    b = eh.bins(_regions(), upstream=100, downstream=100, bin_size=50, anchor="start", snap=50)
    assert (b.start % 50 == 0).all()
    a = b[b.region == "a"]
    assert a.start.min() == 1000 - 100


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
@pytest.mark.parametrize("col", ["start", "end"])
def test_bins_rejects_missing_coordinates(col):
    r = _regions().astype({"start": float, "end": float})
    r.loc["b", col] = np.nan
    with pytest.raises(ValueError, match=f"missing values in '{col}'"):
        eh.bins(r)


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
