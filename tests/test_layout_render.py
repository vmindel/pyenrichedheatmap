"""Smoke, geometry alignment, shared-scale, and hybrid vector/raster PDF tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import enriched_heatmap as eh


def _synthetic_multiomics(n_regions: int = 180, n_bins: int = 80, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"peak_{i:05d}" for i in range(n_regions)])
    x = np.linspace(-1000, 1000, n_bins)

    n1 = n_regions // 3
    n2 = n_regions // 3
    n3 = n_regions - n1 - n2

    # 3 distinct chromatin states
    amp1 = rng.uniform(5.0, 12.0, size=(n1, 1))
    amp2 = rng.uniform(2.0, 5.0, size=(n2, 1))
    amp3 = rng.uniform(0.1, 0.8, size=(n3, 1))

    chec_vals = np.vstack([
        amp1 * np.exp(-0.5 * (x / 90.0) ** 2)[None, :],
        amp2 * np.exp(-0.5 * ((x - 180.0) / 140.0) ** 2)[None, :],
        amp3 * np.exp(-0.5 * (x / 200.0) ** 2)[None, :],
    ]) + rng.gamma(1.2, 0.25, size=(n_regions, n_bins))

    atac_vals = np.vstack([
        amp1 * 0.8 * np.exp(-0.5 * (x / 180.0) ** 2)[None, :],
        amp2 * 0.7 * np.exp(-0.5 * (x / 240.0) ** 2)[None, :],
        amp3 * 0.3 * np.exp(-0.5 * (x / 300.0) ** 2)[None, :],
    ]) + rng.gamma(1.0, 0.2, size=(n_regions, n_bins))

    # MNase shows nucleosome-depleted center + phased +/-1 nucleosomes for open peaks
    nuc_wave = (
        np.exp(-0.5 * ((x + 170) / 60) ** 2)
        + np.exp(-0.5 * ((x - 170) / 60) ** 2)
        + 0.7 * np.exp(-0.5 * ((x + 360) / 65) ** 2)
        + 0.7 * np.exp(-0.5 * ((x - 360) / 65) ** 2)
    )
    mnase_vals = np.vstack([
        3.5 * nuc_wave[None, :] * np.ones((n1, 1)),
        2.0 * nuc_wave[None, :] * np.ones((n2, 1)),
        np.full((n3, n_bins), 1.8),
    ]) + rng.normal(0.5, 0.2, size=(n_regions, n_bins))
    mnase_vals = np.clip(mnase_vals, 0.0, None)

    chec = eh.from_array(
        chec_vals.astype(np.float32),
        region_index=idx,
        upstream=1000,
        downstream=1000,
        bin_size=25,
        name="TF ChEC (<120bp)",
    )
    atac = eh.from_array(
        atac_vals.astype(np.float32),
        region_index=idx,
        upstream=1000,
        downstream=1000,
        bin_size=25,
        name="ATAC-seq",
    )
    mnase = eh.from_array(
        mnase_vals.astype(np.float32),
        region_index=idx,
        upstream=1000,
        downstream=1000,
        bin_size=25,
        name="MNase (140-180bp)",
    )

    rna_df = pd.DataFrame(
        {
            "1h": np.concatenate([rng.normal(1.8, 0.5, n1), rng.normal(0.3, 0.4, n2), rng.normal(-0.8, 0.4, n3)]),
            "4h": np.concatenate([rng.normal(2.5, 0.6, n1), rng.normal(0.5, 0.4, n2), rng.normal(-1.4, 0.5, n3)]),
            "24h": np.concatenate([rng.normal(2.2, 0.6, n1), rng.normal(-0.2, 0.5, n2), rng.normal(-1.8, 0.5, n3)]),
        },
        index=idx,
    )

    peak_class = pd.Series(
        np.concatenate([
            rng.choice(["Promoter", "Enhancer"], size=n1, p=[0.75, 0.25]),
            rng.choice(["Promoter", "Enhancer", "Intergenic"], size=n2, p=[0.2, 0.6, 0.2]),
            rng.choice(["Enhancer", "Intergenic"], size=n3, p=[0.25, 0.75]),
        ]),
        index=idx,
        name="Class",
    )

    motif_score = pd.Series(
        np.concatenate([
            rng.normal(11.0, 2.0, n1),
            rng.normal(6.5, 2.0, n2),
            rng.normal(2.0, 1.5, n3),
        ]),
        index=idx,
        name="Motif Score",
    )
    return chec, atac, mnase, rna_df, peak_class, motif_score


def test_full_multiomics_composition_and_pixel_alignment(tmp_path: Path):
    chec, atac, mnase, rna_df, peak_class, motif_score = _synthetic_multiomics()

    # Deliberately shuffle RNA & annotation indices to verify zero-desync alignment!
    rna_shuf = rna_df.sample(frac=1.0, random_state=1)
    class_shuf = peak_class.sample(frac=1.0, random_state=2)
    motif_shuf = motif_score.sample(frac=1.0, random_state=3)

    hm = (
        eh.EnrichedTrack(chec, cmap="Reds", vmax_q=0.99, top_profile="mean_ci")
        + eh.EnrichedTrack(atac, cmap="Blues", vmax_q=0.99, top_profile="mean_sem")
        + eh.EnrichedTrack(mnase, cmap="Purples", top_profile="median_iqr")
        + eh.MatrixTrack(rna_shuf, name="RNA log2FC", cmap="RdBu_r", center=0.0, width=1.2)
        + eh.CategoricalTrack(class_shuf, name="Class", palette="Set2", width=0.25)
        + eh.BarTrack(motif_shuf, name="Motif Score", width=0.55)
        + eh.PointTrack(motif_shuf, name="Motif Pts", smooth_line=True, width=0.55)
    )
    assert len(hm) == 7
    assert hm["ATAC-seq"].name == "ATAC-seq"

    fig, info = hm.plot(
        order_by="TF ChEC (<120bp)",
        order_metric="center_mean",
        split_by=3,
        cluster_palette="tab10",
        row_gap=0.03,
        figsize=(13, 7),
        raster_dpi=200,
        title="Multi-omics Chromatin Landscape",
        return_axes=True,
    )

    assert len(info["split_slices"]) == 3
    assert len(info["region_order"]) == 180

    # Verify pixel-identical vertical cluster boundaries across ALL 7 tracks and the cluster strip!
    body_axes = info["axes"]["body"]
    strip_axes = info["axes"]["cluster_strip"]
    assert len(strip_axes) == 3

    for cluster_i in range(3):
        ref_bbox = strip_axes[cluster_i].get_position()
        for track_key, ax_list in body_axes.items():
            bbox = ax_list[cluster_i].get_position()
            assert np.isclose(bbox.y0, ref_bbox.y0, atol=1e-6), f"y0 mismatch on {track_key}"
            assert np.isclose(bbox.y1, ref_bbox.y1, atol=1e-6), f"y1 mismatch on {track_key}"

    # Verify rasterization contract: heatmap images rasterized, top metaplot lines vector!
    chec_body_ax = body_axes["TF ChEC (<120bp)"][0]
    assert len(chec_body_ax.images) == 1
    assert chec_body_ax.images[0].get_rasterized() is True

    chec_top_ax = info["axes"]["top"]["TF ChEC (<120bp)"]
    assert len(chec_top_ax.lines) >= 3
    assert all(not ln.get_rasterized() for ln in chec_top_ax.lines)

    out_pdf = tmp_path / "multiomics.pdf"
    out_png = tmp_path / "multiomics.png"
    fig.savefig(out_pdf, dpi=200)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    assert out_pdf.exists() and out_pdf.stat().st_size > 1000
    assert out_png.exists() and out_png.stat().st_size > 1000


def test_shared_scale_group_and_differential_defaults(tmp_path: Path):
    chec, atac, _, _, _, _ = _synthetic_multiomics(n_regions=80, n_bins=40)
    wt = atac.copy()
    wt.name = "WT ATAC"
    mut = eh.NormalizedMatrix(
        values=wt.values * 1.8 + 0.5,
        region_index=wt.region_index,
        upstream=wt.upstream,
        downstream=wt.downstream,
        bin_size=wt.bin_size,
        name="Mut ATAC",
    )
    diff = mut.log2_ratio(wt, pseudocount=1.0)

    t_diff = eh.EnrichedTrack(diff)
    assert t_diff.cmap == "RdBu_r"
    assert t_diff.center == 0.0

    hm = (
        eh.EnrichedTrack(wt, cmap="Blues", scale_group="ATAC signal")
        + eh.EnrichedTrack(mut, cmap="Blues", scale_group="ATAC signal")
        + t_diff
    )
    fig, info = hm.plot(order_by=diff.name, return_axes=True)

    # Shared scale group -> exact same Normalize object and ONE shared colorbar!
    assert info["norms"][0] is info["norms"][1]
    assert isinstance(info["norms"][2], mcolors.TwoSlopeNorm)
    assert "group:ATAC signal" in info["axes"]["colorbars"]
    assert len(info["axes"]["colorbars"]) == 2  # 1 shared + 1 differential

    fig.savefig(tmp_path / "differential.pdf", dpi=150)
    plt.close(fig)


def test_scale_regions_and_join_modes(tmp_path: Path):
    rng = np.random.default_rng(10)
    idx_a = pd.Index([f"gene_{i}" for i in range(40)])
    idx_b = pd.Index([f"gene_{i}" for i in range(10, 50)])

    m_a = eh.NormalizedMatrix(
        values=rng.uniform(0, 5, (40, 50)).astype(np.float32),
        region_index=idx_a,
        mode="scale_regions",
        upstream=1000,
        downstream=1000,
        bin_size=100,
        target_bins=30,
        name="Pol2",
    )
    m_b = eh.NormalizedMatrix(
        values=rng.uniform(0, 5, (40, 50)).astype(np.float32),
        region_index=idx_b,
        mode="scale_regions",
        upstream=1000,
        downstream=1000,
        bin_size=100,
        target_bins=30,
        name="H3K36me3",
    )
    hm = eh.EnrichedTrack(m_a, anchor_labels=("TSS", "TES")) + eh.EnrichedTrack(
        m_b, anchor_labels=("TSS", "TES")
    )

    assert len(hm.align_index(join="inner")) == 30
    assert len(hm.align_index(join="left")) == 40
    assert len(hm.align_index(join="outer")) == 50

    fig, info = hm.plot(order_by="Pol2", order_metric="enriched_score", join="inner", return_axes=True)
    assert len(info["region_order"]) == 30
    fig.savefig(tmp_path / "scale_regions.png", dpi=120)
    plt.close(fig)

    # Disjoint index raises clear ValueError
    m_disjoint = eh.from_array(np.ones((5, 20)), region_index=[f"x_{i}" for i in range(5)])
    with pytest.raises(ValueError, match="Zero overlapping regions"):
        (eh.EnrichedTrack(m_a) + eh.EnrichedTrack(m_disjoint)).align_index("inner")


def test_large_scale_25k_rows_hybrid_pdf_stays_compact(tmp_path: Path):
    """25,000 regions should downsample cleanly and produce a compact vector/raster PDF (< 1.5 MB)."""
    n_rows = 25_000
    n_bins = 100
    rng = np.random.default_rng(99)
    x = np.linspace(-1000, 1000, n_bins)
    amps = np.linspace(10.0, 0.2, n_rows, dtype=np.float32)[:, None]
    vals = amps * np.exp(-0.5 * (x / 120.0) ** 2)[None, :].astype(np.float32)
    vals += rng.uniform(0, 0.3, size=(n_rows, n_bins)).astype(np.float32)

    mat = eh.from_array(vals, upstream=1000, downstream=1000, bin_size=20, name="25k ATAC")
    hm = eh.EnrichedTrack(mat, cmap="Blues", top_profile="mean_ci", downsample_agg="q90") + eh.EnrichedTrack(
        mat, name="25k Mean", cmap="Reds", top_profile="mean", downsample_agg="mean"
    )

    fig, info = hm.plot(figsize=(8, 6), raster_dpi=200, return_axes=True)
    # Confirm downsampling happened (image array height << 25,000)
    img_arr = info["axes"]["body"]["25k ATAC"][0].images[0].get_array()
    assert img_arr.shape[0] < 2000

    pdf_path = tmp_path / "large_25k.pdf"
    fig.savefig(pdf_path, dpi=200)
    plt.close(fig)

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    assert size_mb < 1.5, f"PDF too large ({size_mb:.2f} MB); rasterization/downsampling failed."
