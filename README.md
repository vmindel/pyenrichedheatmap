# `pyenrichedheatmap` (`enriched_heatmap`)

A fast, lightweight, publication-grade Python equivalent of R's `EnrichedHeatmap` + `ComplexHeatmap` horizontal composition for genomics and epigenomics (ChEC-seq, ChIP-seq, ATAC-seq, MNase-seq, RNA-seq).

![Multi-omics chromatin landscape](docs/img/cookbook_multiomics.png)

## Key Features

- **Zero-desync alignment:** Every track carries a unique pandas `region_index`. Row ordering, filtering, and cluster splitting are guaranteed consistent across all horizontally composed tracks.
- **Pluggable data sources:** First-class loaders for `pyBigWig`, `pymisha`, raw NumPy/Pandas arrays, and plain `(values, metadata)` interchange payloads via optional dependencies.
- **Vector/raster hybrid PDFs:** Only heatmap bodies are rasterized (with smart anti-aliased row downsampling for 10k–100k+ regions), while metaplots, ticks, borders, labels, legends, and colorbars stay 100% vector for clean editing in Illustrator/Inkscape.
- **Declarative `+` composition API:** Compose `EnrichedTrack`, `MatrixTrack`, `CategoricalTrack`, `BarTrack`, and `PointTrack` horizontally with deterministic nested `matplotlib.gridspec` layouts.
- **Shared color scales & differential heatmaps:** Group tracks onto shared `vmin`/`vmax` colorbars (`scale_group`) and build diverging differential heatmaps via `mat_a.log2_ratio(mat_b)` or `mat_a.subtract(mat_b)` with automatic `TwoSlopeNorm` centering.

## Installation

```bash
pip install pyenrichedheatmap

# Optional extras
pip install "pyenrichedheatmap[stats]"    # scipy + scikit-learn (clustering, smoothing)
pip install "pyenrichedheatmap[bigwig]"   # pyBigWig loader
pip install "pyenrichedheatmap[pymisha]"  # pymisha loader
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import enriched_heatmap as eh

# Load genomic matrices (from_bigwig, from_pymisha, from_array, from_dataframe)
chec = eh.from_bigwig("tf_chec.bw", peaks_df, upstream=1000, downstream=1000, bin_size=10, name="TF ChEC (<120bp)")
atac = eh.from_bigwig("atac.bw", peaks_df, upstream=1000, downstream=1000, bin_size=10, name="ATAC-seq")
mnase = eh.from_bigwig("mnase.bw", peaks_df, upstream=1000, downstream=1000, bin_size=10, name="MNase (140-180bp)")

# Declarative horizontal composition with `+`
hm = (
    eh.EnrichedTrack(chec, cmap="Reds", vmax_q=0.99, top_profile="mean_ci", downsample_agg="q90")
    + eh.EnrichedTrack(atac, cmap="Blues", vmax_q=0.99, top_profile="mean")
    + eh.EnrichedTrack(mnase, cmap="Purples", top_profile="mean")
    + eh.MatrixTrack(rna_logfc_df, name="RNA log2FC", cmap="RdBu_r", center=0.0, width=1.2)
    + eh.CategoricalTrack(peak_class, name="Class", palette="Set2", width=0.25)
    + eh.BarTrack(motif_score, name="Motif Score", width=0.55)
)

fig = hm.plot(
    order_by="TF ChEC (<120bp)",
    order_metric="center_mean",
    split_by=3,                     # integer k-means, categorical Series, or None
    cluster_palette="tab10",
    row_gap=0.03,
    figsize=(12, 7),
    raster_dpi=300,
)
fig.savefig("figure1.pdf", dpi=300)
```

### Differential & Shared-Scale Example

```python
wt = eh.from_pymisha("atac_wt", peaks_df, upstream=1000, downstream=1000, bin_size=10, name="WT ATAC")
mut = eh.from_pymisha("atac_mut", peaks_df, upstream=1000, downstream=1000, bin_size=10, name="Mut ATAC")
diff = mut.log2_ratio(wt, pseudocount=1.0)  # auto-uses RdBu_r + center=0.0 in EnrichedTrack

hm = (
    eh.EnrichedTrack(wt, cmap="Blues", scale_group="ATAC signal")
    + eh.EnrichedTrack(mut, cmap="Blues", scale_group="ATAC signal")
    + eh.EnrichedTrack(diff, name="log2(Mut/WT)", top_profile="mean_sem")
)
fig = hm.plot(order_by="log2(Mut/WT)", order_metric="center_mean")
```

## Documentation & Examples

- Full documentation (tutorials + API reference): https://vmindel.github.io/pyenrichedheatmap/
- Runnable end-to-end cookbook: [examples/cookbook_multiomics.py](examples/cookbook_multiomics.py)

## Claude Code Skill

`pyenrichedheatmap` ships a bundled Agent Skill for Claude Code covering its API contracts, coordinate conventions, shared scales, and large-scale rendering tips:

```bash
pyenrichedheatmap-install-skills
```

Pass `--dest .claude/skills/pyenrichedheatmap` to install into a single project instead of `~/.claude/skills/pyenrichedheatmap`.
