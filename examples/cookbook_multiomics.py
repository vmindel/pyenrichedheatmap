"""End-to-end multi-omics cookbook demonstrating pyenrichedheatmap capabilities.

Generates:
1. Multi-omics figure: TF ChEC (<120bp) + ATAC-seq + MNase (140-180bp) + RNA log2FC
   + Peak Class + Motif Score, split by k-means (k=3) and ordered by TF ChEC center signal.
2. Differential chromatin accessibility figure: WT ATAC + Mutant ATAC (shared colorbar!)
   + log2(Mut/WT) differential heatmap with automatic diverging TwoSlopeNorm.
3. Scale-regions gene-body figure: Pol II + H3K36me3 across TSS -> TES.

Run with:
    python examples/cookbook_multiomics.py --out-dir /tmp/eh_cookbook
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import enriched_heatmap as eh


def generate_multiomics_figure(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    n_regions = 1500
    n_bins = 200  # ±1000 bp at 10 bp resolution
    idx = pd.Index([f"peak_{i:05d}" for i in range(n_regions)])
    x = np.linspace(-1000, 1000, n_bins)

    n1, n2 = 450, 550
    n3 = n_regions - n1 - n2

    amp1 = rng.uniform(5.0, 14.0, (n1, 1))
    amp2 = rng.uniform(2.0, 6.0, (n2, 1))
    amp3 = rng.uniform(0.1, 0.9, (n3, 1))

    # 1) Narrow TF ChEC-seq (<120bp fragments) centered at motif anchor
    chec_vals = np.vstack([
        amp1 * np.exp(-0.5 * (x / 55.0) ** 2)[None, :],
        amp2 * np.exp(-0.5 * ((x - 120.0) / 85.0) ** 2)[None, :],
        amp3 * np.exp(-0.5 * (x / 180.0) ** 2)[None, :],
    ]) + rng.gamma(1.1, 0.2, (n_regions, n_bins))

    # 2) ATAC-seq accessibility
    atac_vals = np.vstack([
        amp1 * 0.75 * np.exp(-0.5 * (x / 160.0) ** 2)[None, :],
        amp2 * 0.65 * np.exp(-0.5 * (x / 210.0) ** 2)[None, :],
        amp3 * 0.25 * np.exp(-0.5 * (x / 250.0) ** 2)[None, :],
    ]) + rng.gamma(1.0, 0.18, (n_regions, n_bins))

    # 3) MNase nucleosome occupancy (140-180bp): NDR at center + phased +/-1, +/-2 nucleosomes
    phased = (
        1.1 * np.exp(-0.5 * ((x + 170) / 55) ** 2)
        + 1.1 * np.exp(-0.5 * ((x - 170) / 55) ** 2)
        + 0.75 * np.exp(-0.5 * ((x + 355) / 60) ** 2)
        + 0.75 * np.exp(-0.5 * ((x - 355) / 60) ** 2)
        + 0.45 * np.exp(-0.5 * ((x + 540) / 65) ** 2)
        + 0.45 * np.exp(-0.5 * ((x - 540) / 65) ** 2)
    )
    mnase_vals = np.vstack([
        3.2 * phased[None, :] * rng.uniform(0.8, 1.2, (n1, 1)),
        1.9 * phased[None, :] * rng.uniform(0.7, 1.2, (n2, 1)),
        np.full((n3, n_bins), 1.6) + rng.normal(0, 0.15, (n3, n_bins)),
    ]) + rng.normal(0.35, 0.15, (n_regions, n_bins))
    mnase_vals = np.clip(mnase_vals, 0.0, None)

    chec = eh.from_array(chec_vals.astype(np.float32), region_index=idx, upstream=1000, downstream=1000, bin_size=10, name="TF ChEC (<120bp)")
    atac = eh.from_array(atac_vals.astype(np.float32), region_index=idx, upstream=1000, downstream=1000, bin_size=10, name="ATAC-seq")
    mnase = eh.from_array(mnase_vals.astype(np.float32), region_index=idx, upstream=1000, downstream=1000, bin_size=10, name="MNase (140-180bp)")

    rna_df = pd.DataFrame(
        {
            "1h": np.concatenate([rng.normal(1.6, 0.5, n1), rng.normal(0.3, 0.4, n2), rng.normal(-0.9, 0.4, n3)]),
            "4h": np.concatenate([rng.normal(2.4, 0.6, n1), rng.normal(0.6, 0.5, n2), rng.normal(-1.4, 0.5, n3)]),
            "24h": np.concatenate([rng.normal(2.0, 0.6, n1), rng.normal(-0.2, 0.5, n2), rng.normal(-1.9, 0.5, n3)]),
        },
        index=idx,
    )
    peak_class = pd.Series(
        np.concatenate([
            rng.choice(["Promoter", "Enhancer", "Intergenic"], size=n1, p=[0.70, 0.25, 0.05]),
            rng.choice(["Promoter", "Enhancer", "Intergenic"], size=n2, p=[0.20, 0.65, 0.15]),
            rng.choice(["Promoter", "Enhancer", "Intergenic"], size=n3, p=[0.10, 0.25, 0.65]),
        ]),
        index=idx,
        name="Class",
    )
    motif_score = pd.Series(
        np.concatenate([
            rng.normal(11.5, 2.0, n1),
            rng.normal(6.5, 2.0, n2),
            rng.normal(2.2, 1.4, n3),
        ]),
        index=idx,
        name="Motif Score",
    )

    hm = (
        eh.EnrichedTrack(chec, cmap="Reds", vmax_q=0.99, top_profile="mean_ci", downsample_agg="q90", colorbar_label="CPM")
        + eh.EnrichedTrack(atac, cmap="Blues", vmax_q=0.99, top_profile="mean_sem", colorbar_label="CPM")
        + eh.EnrichedTrack(mnase, cmap="Purples", vmax_q=0.99, top_profile="mean", colorbar_label="Occ.")
        + eh.MatrixTrack(rna_df, name="RNA log2FC", cmap="RdBu_r", center=0.0, width=1.1, colorbar_label="log2FC")
        + eh.CategoricalTrack(peak_class, name="Class", palette="Set2", width=0.25)
        + eh.BarTrack(motif_score, name="Motif Score", width=0.65, xlabel="PWM")
    )

    fig = hm.plot(
        order_by="TF ChEC (<120bp)",
        order_metric="center_mean",
        split_by=3,
        cluster_palette="tab10",
        row_gap=0.03,
        figsize=(12.5, 7.2),
        raster_dpi=300,
        title="Multi-Omics TF Binding & Chromatin Landscape",
    )
    fig.savefig(out_dir / "cookbook_multiomics.png", dpi=180)
    fig.savefig(out_dir / "cookbook_multiomics.pdf", dpi=300)
    plt.close(fig)


def generate_differential_figure(out_dir: Path) -> None:
    rng = np.random.default_rng(7)
    n_regions = 900
    n_bins = 100
    idx = pd.Index([f"site_{i:04d}" for i in range(n_regions)])
    x = np.linspace(-1000, 1000, n_bins)

    base = rng.uniform(2.0, 9.0, (n_regions, 1)) * np.exp(-0.5 * (x / 180.0) ** 2)[None, :]
    wt_vals = base + rng.gamma(1.0, 0.25, (n_regions, n_bins))

    # First 300 sites gain accessibility in mutant, middle 300 unchanged, last 300 lose accessibility
    fold = np.concatenate([
        rng.uniform(2.2, 4.0, 300),
        rng.uniform(0.9, 1.1, 300),
        rng.uniform(0.15, 0.45, 300),
    ])[:, None]
    mut_vals = base * fold + rng.gamma(1.0, 0.25, (n_regions, n_bins))

    wt = eh.from_array(wt_vals.astype(np.float32), region_index=idx, upstream=1000, downstream=1000, bin_size=20, name="WT ATAC")
    mut = eh.from_array(mut_vals.astype(np.float32), region_index=idx, upstream=1000, downstream=1000, bin_size=20, name="Mutant ATAC")
    diff = mut.log2_ratio(wt, pseudocount=1.0)
    diff.name = "log2(Mut / WT)"

    groups = pd.Series(
        pd.Categorical(
            ["Gained"] * 300 + ["Unchanged"] * 300 + ["Lost"] * 300,
            categories=["Gained", "Unchanged", "Lost"],
            ordered=True,
        ),
        index=idx,
    )

    hm = (
        eh.EnrichedTrack(wt, cmap="Blues", scale_group="ATAC-seq (CPM)", top_profile="mean_ci")
        + eh.EnrichedTrack(mut, cmap="Blues", scale_group="ATAC-seq (CPM)", top_profile="mean_ci")
        + eh.EnrichedTrack(diff, top_profile="mean_ci", colorbar_label="log2FC")
    )

    fig = hm.plot(
        order_by="log2(Mut / WT)",
        order_metric="center_mean",
        split_by=groups,
        cluster_palette={"Gained": "#d62728", "Unchanged": "#7f7f7f", "Lost": "#1f77b4"},
        figsize=(9.5, 6.5),
        raster_dpi=300,
        title="Differential Chromatin Accessibility (Shared WT/Mut Scale)",
    )
    fig.savefig(out_dir / "cookbook_differential.png", dpi=180)
    fig.savefig(out_dir / "cookbook_differential.pdf", dpi=300)
    plt.close(fig)


def generate_scale_regions_figure(out_dir: Path) -> None:
    rng = np.random.default_rng(19)
    n_genes = 600
    n_up, n_body, n_down = 20, 60, 20  # -1 kb, 60 scaled body bins, +1 kb (bin_size=50)
    n_bins = n_up + n_body + n_down
    idx = pd.Index([f"gene_{i:04d}" for i in range(n_genes)])

    n_high, n_low = 320, n_genes - 320
    # Coordinate grid in visual bin units: [-1000..0] up, [0..3000] body, [3000..4000] down
    coords = np.linspace(-975, 3975, n_bins)

    # Pol II: sharp promoter-proximal pause peak near TSS (0) + gene-body elongation + small TES dip
    pol2_profile = (
        7.5 * np.exp(-0.5 * ((coords - 50) / 140) ** 2)
        + 2.2 * ((coords >= 0) & (coords <= 3000)).astype(np.float64)
        + 1.6 * np.exp(-0.5 * ((coords - 3050) / 180) ** 2)
    )
    pol2_vals = np.vstack([
        rng.uniform(0.7, 1.5, (n_high, 1)) * pol2_profile[None, :],
        rng.uniform(0.05, 0.25, (n_low, 1)) * pol2_profile[None, :],
    ]) + rng.gamma(1.0, 0.15, (n_genes, n_bins))

    # H3K36me3: enriched across gene body, ramping toward 3' TES
    body_ramp = np.where(
        (coords >= 0) & (coords <= 3000),
        0.5 + 3.8 * (np.maximum(coords, 0.0) / 3000.0) ** 1.3,
        0.25 * np.exp(-0.5 * ((coords - 3000) / 350) ** 2) * (coords > 3000),
    )
    k36_vals = np.vstack([
        rng.uniform(0.7, 1.4, (n_high, 1)) * body_ramp[None, :],
        rng.uniform(0.05, 0.25, (n_low, 1)) * body_ramp[None, :],
    ]) + rng.gamma(1.0, 0.12, (n_genes, n_bins))

    pol2 = eh.NormalizedMatrix(
        values=pol2_vals.astype(np.float32),
        region_index=idx,
        mode="scale_regions",
        upstream=1000,
        downstream=1000,
        bin_size=50,
        target_bins=n_body,
        name="Pol II ChIP",
    )
    h3k36 = eh.NormalizedMatrix(
        values=k36_vals.astype(np.float32),
        region_index=idx,
        mode="scale_regions",
        upstream=1000,
        downstream=1000,
        bin_size=50,
        target_bins=n_body,
        name="H3K36me3",
    )
    expr_grp = pd.Series(
        pd.Categorical(
            ["Active"] * n_high + ["Inactive"] * n_low,
            categories=["Active", "Inactive"],
            ordered=True,
        ),
        index=idx,
    )

    hm = (
        eh.EnrichedTrack(
            pol2,
            cmap="Reds",
            anchor_labels=("TSS", "TES"),
            top_profile="mean_ci",
            colorbar_label="CPM",
        )
        + eh.EnrichedTrack(
            h3k36,
            cmap="Greens",
            anchor_labels=("TSS", "TES"),
            top_profile="mean_ci",
            colorbar_label="CPM",
        )
    )
    fig = hm.plot(
        order_by="Pol II ChIP",
        order_metric="enriched_score",
        split_by=expr_grp,
        cluster_palette={"Active": "#c44e52", "Inactive": "#8c8c8c"},
        figsize=(8.2, 6.0),
        raster_dpi=300,
        title="Gene-Body Scale-Regions Profile (TSS to TES)",
    )
    fig.savefig(out_dir / "cookbook_scale_regions.png", dpi=180)
    fig.savefig(out_dir / "cookbook_scale_regions.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyenrichedheatmap cookbook examples.")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/eh_cookbook"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generate_multiomics_figure(args.out_dir)
    generate_differential_figure(args.out_dir)
    generate_scale_regions_figure(args.out_dir)
    print(f"Saved cookbook figures to {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
