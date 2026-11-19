"""Tests for array, DataFrame, pyBigWig, and pymisha loaders (including strand & boundary rigor)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import enriched_heatmap as eh


# ------------------------------------------------------------------
# Array & DataFrame loaders
# ------------------------------------------------------------------
def test_from_array_and_dataframe():
    arr = np.arange(60, dtype=np.float32).reshape(3, 20)
    m1 = eh.from_array(arr, upstream=1000, downstream=1000, name="ArrayTrack")
    assert m1.bin_size == 100
    assert m1.shape == (3, 20)
    assert m1.name == "ArrayTrack"

    df = pd.DataFrame(arr, index=["g1", "g2", "g3"])
    m2 = eh.from_dataframe(df, upstream=500, downstream=500, name="DFTrack")
    assert list(m2.region_index) == ["g1", "g2", "g3"]
    assert m2.bin_size == 50
    np.testing.assert_allclose(m2.values, arr)


# ------------------------------------------------------------------
# Mock pyBigWig helper
# ------------------------------------------------------------------
class _FakeBigWig:
    def __init__(self, tracks: dict[str, np.ndarray]):
        self._tracks = tracks
        self.closed = False

    def chroms(self) -> dict[str, int]:
        return {c: len(v) for c, v in self._tracks.items()}

    def values(self, chrom: str, start: int, end: int, numpy: bool = True) -> np.ndarray:
        arr = self._tracks[chrom]
        assert 0 <= start <= end <= len(arr), f"OOB query {chrom}:{start}-{end}"
        return arr[start:end].astype(np.float64)

    def close(self) -> None:
        self.closed = True


def _install_fake_pybigwig(monkeypatch: pytest.MonkeyPatch, tracks: dict[str, np.ndarray]) -> None:
    mod = types.ModuleType("pyBigWig")
    mod.open = lambda path: _FakeBigWig(tracks)
    monkeypatch.setitem(sys.modules, "pyBigWig", mod)


def test_from_bigwig_reference_point_and_strand_flipping(monkeypatch: pytest.MonkeyPatch):
    # Signal equals genomic coordinate position so we can verify exact bin alignment!
    chrom_len = 5000
    chr1 = np.arange(chrom_len, dtype=np.float64)
    _install_fake_pybigwig(monkeypatch, {"chr1": chr1})

    # Asymmetric flanks: upstream=40, downstream=20, bin_size=10 -> 4 upstream + 2 downstream = 6 bins
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1200, 1200],
            "strand": ["+", "-"],
        },
        index=["gene_plus", "gene_minus"],
    )

    mat = eh.from_bigwig(
        "dummy.bw",
        intervals,
        mode="reference_point",
        reference_point="tss",
        upstream=40,
        downstream=20,
        bin_size=10,
        agg="mean",
    )
    assert list(mat.region_index) == ["gene_plus", "gene_minus"]
    assert mat.shape == (2, 6)

    # Plus strand: TSS=1000, window=[960, 1020), bins: [960..969]=964.5, ..., [1010..1019]=1014.5
    expected_plus = np.array([964.5, 974.5, 984.5, 994.5, 1004.5, 1014.5], dtype=np.float32)
    np.testing.assert_allclose(mat.values[0], expected_plus)

    # Minus strand: TSS=1200, 5' upstream=[1200, 1240), 3' downstream=[1180, 1200)
    # Reversed 5'->3' bins: [1230..1239]=1234.5, [1220..1229]=1224.5, [1210..1219]=1214.5,
    #                       [1200..1209]=1204.5, [1190..1199]=1194.5, [1180..1189]=1184.5
    expected_minus = np.array([1234.5, 1224.5, 1214.5, 1204.5, 1194.5, 1184.5], dtype=np.float32)
    np.testing.assert_allclose(mat.values[1], expected_minus)


def test_from_bigwig_oob_chr_prefix_and_scale_regions(monkeypatch: pytest.MonkeyPatch):
    chr1 = np.arange(200, dtype=np.float64)
    chr1[50:60] = np.nan
    _install_fake_pybigwig(monkeypatch, {"chr1": chr1})

    # Interval near chromosome start (will go negative) + missing "chr" prefix
    intervals = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [10, 100],
            "end": [30, 140],
            "strand": ["+", "-"],
        },
        index=["edge", "body_minus"],
    )

    mat_ref = eh.from_bigwig(
        "dummy.bw",
        intervals,
        mode="reference_point",
        reference_point="start",
        upstream=30,
        downstream=20,
        bin_size=10,
        out_of_bounds="nan",
        chr_prefix="auto",
    )
    # Row 0: start=10, window=[-20, 30) -> first 2 bins are OOB (NaN), next 3 bins are valid
    assert np.all(np.isnan(mat_ref.values[0, :2]))
    np.testing.assert_allclose(mat_ref.values[0, 2:], [4.5, 14.5, 24.5])

    with pytest.raises(KeyError, match="not found"):
        eh.from_bigwig("dummy.bw", intervals, upstream=20, downstream=20, bin_size=10, chr_prefix="strict")

    # scale_regions test (asymmetric flanks + minus strand!)
    mat_scale = eh.from_bigwig(
        "dummy.bw",
        intervals.iloc[[1]],  # [100, 140) on "-" strand
        mode="scale_regions",
        upstream=20,
        downstream=10,
        bin_size=10,
        target_bins=4,
        out_of_bounds="zero",
    )
    # 2 upstream + 4 body + 1 downstream = 7 bins, reversed 5' -> 3':
    # 5' upstream: [150..159]=154.5, [140..149]=144.5
    # body [100..140) reversed: [130..139]=134.5, [120..1229? 120..129]=124.5, 114.5, 104.5
    # 3' downstream: [90..99]=94.5
    expected = np.array([154.5, 144.5, 134.5, 124.5, 114.5, 104.5, 94.5], dtype=np.float32)
    np.testing.assert_allclose(mat_scale.values[0], expected)


def test_from_bigwig_parallel_and_short_body(monkeypatch: pytest.MonkeyPatch):
    chr1 = np.arange(5000, dtype=np.float64)
    _install_fake_pybigwig(monkeypatch, {"chr1": chr1})

    n = 100
    starts = np.arange(200, 200 + n * 20, 20)
    intervals = pd.DataFrame({"chrom": "chr1", "start": starts, "end": starts + 4})

    mat_seq = eh.from_bigwig(
        "dummy.bw", intervals, mode="scale_regions", upstream=20, downstream=20, bin_size=10, target_bins=8, n_jobs=1
    )
    mat_par = eh.from_bigwig(
        "dummy.bw", intervals, mode="scale_regions", upstream=20, downstream=20, bin_size=10, target_bins=8, n_jobs=2
    )
    np.testing.assert_allclose(mat_seq.values, mat_par.values)


# ------------------------------------------------------------------
# Mock pymisha helper
# ------------------------------------------------------------------
def _install_fake_pymisha(monkeypatch: pytest.MonkeyPatch, tracks: dict[str, dict[str, np.ndarray]]) -> None:
    """Install a mock pymisha module enforcing canonical non-overlapping intervals & 1-based intervalID."""
    mod = types.ModuleType("pymisha")

    first_track = next(iter(tracks.values()))
    all_iv = pd.DataFrame(
        {
            "chrom": list(first_track.keys()),
            "start": [0] * len(first_track),
            "end": [len(v) for v in first_track.values()],
        }
    )
    mod.gintervals_all = lambda: all_iv.copy()

    def gextract(expr: str, intervals: pd.DataFrame, iterator: int | pd.DataFrame = 10, **kwargs):
        # Verify caller passed strictly sorted, non-overlapping intervals!
        prev_c, prev_e = None, -1
        for c, s, e in zip(intervals["chrom"], intervals["start"], intervals["end"], strict=True):
            assert 0 <= s < e <= len(first_track[c]), f"Invalid/OOB interval {c}:{s}-{e}"
            if prev_c == c:
                assert s >= prev_e, f"Overlapping intervals passed to gextract: {prev_e} > {s}"
            prev_c, prev_e = c, e

        def _eval_slice(c: str, s: int, e: int) -> float:
            # Support simple "a - b" expressions in addition to plain track names
            if " - " in expr:
                left_t, right_t = [x.strip() for x in expr.split(" - ", 1)]
                return float(np.nanmean(tracks[left_t][c][s:e]) - np.nanmean(tracks[right_t][c][s:e]))
            return float(np.nanmean(tracks[expr][c][s:e]))

        rows = []
        if isinstance(iterator, pd.DataFrame):
            for idx_0, (c, s, e) in enumerate(
                zip(intervals["chrom"], intervals["start"], intervals["end"], strict=True)
            ):
                rows.append((c, int(s), int(e), _eval_slice(c, int(s), int(e)), idx_0 + 1))
        else:
            step = int(iterator)
            for idx_0, (c, s, e) in enumerate(
                zip(intervals["chrom"], intervals["start"], intervals["end"], strict=True)
            ):
                pos = int(s)
                while pos < int(e):
                    nxt = min(pos + step, int(e))
                    rows.append((c, pos, nxt, _eval_slice(c, pos, nxt), idx_0 + 1))
                    pos = nxt

        return pd.DataFrame(rows, columns=["chrom", "start", "end", expr, "intervalID"])

    mod.gextract = gextract
    monkeypatch.setitem(sys.modules, "pymisha", mod)


def test_from_pymisha_overlapping_windows_expressions_and_strand(monkeypatch: pytest.MonkeyPatch):
    t_a = {"chr1": np.arange(2000, dtype=np.float64) * 2.0}
    t_b = {"chr1": np.arange(2000, dtype=np.float64)}
    _install_fake_pymisha(monkeypatch, {"track_a": t_a, "track_b": t_b})

    # Deliberately unsorted, overlapping windows + minus strand + OOB start!
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [500, 510, 10],
            "end": [520, 530, 30],
            "strand": ["+", "-", "+"],
        },
        index=["p_overlap_1", "p_overlap_2", "p_oob"],
    )

    mat = eh.from_pymisha(
        "track_a - track_b",
        intervals,
        mode="reference_point",
        reference_point="tss",
        upstream=30,
        downstream=20,
        bin_size=10,
        out_of_bounds="nan",
    )
    assert list(mat.region_index) == ["p_overlap_1", "p_overlap_2", "p_oob"]
    assert mat.shape == (3, 5)

    # track_a - track_b == coord!
    # Row 0 (+): TSS=500, [470, 520) -> 474.5, 484.5, 494.5, 504.5, 514.5
    np.testing.assert_allclose(mat.values[0], [474.5, 484.5, 494.5, 504.5, 514.5])
    # Row 1 (-): TSS=530, 5' up=[530, 560), 3' down=[510, 530), reversed -> 554.5, 544.5, 534.5, 524.5, 514.5
    np.testing.assert_allclose(mat.values[1], [554.5, 544.5, 534.5, 524.5, 514.5])
    # Row 2 (+): TSS=10, [-20, 30) -> first 2 bins NaN, rest 4.5, 14.5, 24.5
    assert np.all(np.isnan(mat.values[2, :2]))
    np.testing.assert_allclose(mat.values[2, 2:], [4.5, 14.5, 24.5])


def test_from_pymisha_scale_regions(monkeypatch: pytest.MonkeyPatch):
    t_a = {"chr1": np.arange(1000, dtype=np.float64)}
    _install_fake_pymisha(monkeypatch, {"track_a": t_a})

    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [200, 210], "end": [240, 250], "strand": ["+", "-"]},
        index=["g_plus", "g_minus"],
    )
    mat = eh.from_pymisha(
        "track_a",
        intervals,
        mode="scale_regions",
        upstream=20,
        downstream=10,
        bin_size=10,
        target_bins=4,
    )
    np.testing.assert_allclose(mat.values[0], [184.5, 194.5, 204.5, 214.5, 224.5, 234.5, 244.5])
    np.testing.assert_allclose(mat.values[1], [264.5, 254.5, 244.5, 234.5, 224.5, 214.5, 204.5])
