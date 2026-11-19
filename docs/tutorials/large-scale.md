# Large-scale rendering & clustering

Epigenomic peak sets routinely contain 10,000–100,000 regions. Plotting every
row directly into a vector PDF creates giant files that choke Illustrator, while
naive nearest-neighbor subsampling causes sharp narrow peaks (e.g., TF ChEC-seq
or motif hits) to disappear.

## Smart row downsampling

When a panel contains more rows than its physical pixel height at `raster_dpi`,
`pyenrichedheatmap` aggregates consecutive sorted rows into exact pixel buckets
via [`downsample_rows`](../api/profiles-raster.md):

| `downsample_agg` | Best for | Behavior |
|---|---|---|
| `"mean"` | ATAC-seq, MNase-seq, histone marks | Fast C-level NaN-safe bucket average |
| `"q90"` / `"q95"` | Narrow TF ChEC-seq / ChIP-seq peaks | Preserves sharp peaks without single-outlier blowup |
| `"max"` | Sparse binary motif hits or cut sites | Guarantees narrow bands never vanish |
| `"absmax"` | Differential `log2FC` heatmaps | Retains the signed extreme so `+` and `-` spikes don't cancel |
| `"median"` / `"min"` / callable | Custom summaries | Exact per-bucket aggregation |

Set a global default in `hm.plot(...)` and override per track as needed:

```python
import enriched_heatmap as eh

hm = (
    eh.EnrichedTrack(chec_50k, cmap="Reds", downsample_agg="q90", top_profile="mean_ci")
    + eh.EnrichedTrack(atac_50k, cmap="Blues", downsample_agg="mean", top_profile="mean")
    + eh.EnrichedTrack(diff_50k, downsample_agg="absmax", top_profile="mean")
)

fig = hm.plot(
    order_by="TF ChEC (<120bp)",
    figsize=(11, 7),
    raster_dpi=300,
)
fig.savefig("large_50k.pdf", dpi=300)  # typically < 1–2 MB, vector text + raster body
```

## Hybrid vector / raster PDFs

- **Rasterized (`rasterized=True`):** heatmap `imshow` matrix bodies (and
  `PointTrack` scatter markers when `n > 2,000`).
- **100% Vector:** top metaplot curves and CI/SEM ribbons, dashed anchor guides,
  box borders, ticks, titles, cluster strips, `n = ...` labels, legends, and colorbars.

## Ordering & clustering

```python
# K-means clustering across multiple aligned tracks
clusters = eh.cluster_kmeans([chec, atac, mnase], k=4, scale="zscore", random_state=42)

# Or run directly inside plot() via split_by=int + cluster_tracks
fig = hm.plot(
    order_by="TF ChEC (<120bp)",
    order_metric="enriched_score",      # center-weighted enrichment score
    split_by=4,
    cluster_tracks=["TF ChEC (<120bp)", "ATAC-seq"],
)

# Hierarchical clustering (requires pyenrichedheatmap[stats])
ordered_idx, h_clusters = eh.cluster_hierarchical([chec, atac], k=4, method="ward")
fig = hm.plot(order_by=ordered_idx, split_by=h_clusters)
```
