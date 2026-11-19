# Ordering, clustering, and cluster splitting

Row alignment is guaranteed zero-desync: `eh.HeatmapList.plot()` first aligns
all tracks on their shared `region_index`, then resolves splits and orders rows
**within each split group** uniformly across all tracks.

## `order_by` & `order_metric`

```python
fig = hm.plot(
    order_by="TF ChEC (<120bp)",   # track name, "kmeans", "hierarchical", callable, Series, pd.Index, or 1D array
    order_metric="center_mean",    # "center_mean" | "center_max" | "enriched_score" | "mean" | "median" | "max" | "sum" | callable
    ascending=False,               # False puts strongest signal at the top; NaNs always sort last
)
```

- `eh.compute_order_scores(mat, metric="enriched_score")` computes a
  center-weighted enrichment score (mirroring R's `EnrichedHeatmap::enriched_score`)
  giving linearly increasing weight from flank edges toward the anchor/body.
- Passing a `pd.Index` to `order_by` applies that exact region order directly.
- Passing a `pd.Series` aligns scores by `region_index` before sorting.

## Splitting (`split_by`) & clustering helpers

```python
# Automatic k-means splitting directly in plot()
fig = hm.plot(
    order_by="ATAC-seq",
    split_by=3,                              # runs k-means (k=3)
    cluster_tracks=["TF ChEC", "ATAC-seq"],  # optional multi-track joint clustering
    cluster_palette="tab10",                 # dict, colormap name, or color sequence
    show_cluster_strip=True,
    row_gap=0.03,
)

# Standalone clustering functions (return Series/Index aligned to region_index)
clusters = eh.cluster_kmeans([chec, atac], k=4, scale="zscore", random_state=42)
ordered_idx, h_clusters = eh.cluster_hierarchical([chec, atac], k=4, method="ward")
```

- `eh.cluster_kmeans` orders cluster labels (`C1`, `C2`, ..., `Ck`) by descending
  mean signal of the primary matrix (`order_clusters="descending"`), so `C1` is
  always the strongest cluster. It uses `scikit-learn` when installed and falls
  back to deterministic NumPy k-means++ otherwise.
- `eh.cluster_hierarchical` requires `pyenrichedheatmap[stats]` (`scipy`).
- When `split_by` is a `pd.Series` with an ordered `pd.CategoricalDtype`, the
  figure preserves the exact category order top-to-bottom.

## Pitfalls

- Passing an unindexed list/array to `split_by` assumes positional alignment
  with the aligned `region_index` — always prefer a `pd.Series` indexed by
  `region_index` so shuffling or `join="inner"` subsetting can never desync labels.
- In multi-track clustering (`cluster_tracks=[...]`), each block is globally
  z-scored and weighted by `1 / sqrt(n_features)` so wide matrices do not drown
  out narrower tracks.
