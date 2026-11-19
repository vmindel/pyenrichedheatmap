# Tracks and horizontal `+` composition

Tracks compose left-to-right with `+` into an `eh.HeatmapList`:

```python
import enriched_heatmap as eh

hm = (
    eh.EnrichedTrack(chec, cmap="Reds", vmax_q=0.99, top_profile="mean_ci", downsample_agg="q90")
    + eh.EnrichedTrack(atac, cmap="Blues", vmax_q=0.99, top_profile="mean_sem")
    + eh.MatrixTrack(rna_logfc_df, name="RNA log2FC", cmap="RdBu_r", center=0.0, width=1.1)
    + eh.CategoricalTrack(peak_class, name="Class", palette="Set2", width=0.25)
    + eh.BarTrack(motif_score, name="Motif Score", width=0.6)
    + eh.PointTrack(tss_dist, name="log10 dist", smooth_line=True, width=0.6)
)
fig = hm.plot(order_by="TF ChEC (<120bp)", split_by=3)
```

## Track classes

| class | input | highlights |
|---|---|---|
| `eh.EnrichedTrack` | `eh.NormalizedMatrix` (or interchange) | Genomic heatmap + optional `top_profile` (`"mean"`, `"median"`, `"mean_sem"`, `"mean_ci"`, `"mean_boot_ci"`, `"median_iqr"`, or `None`), vertical dashed guide lines, `anchor_labels` (`"TSS"` or `("TSS", "TES")`) |
| `eh.MatrixTrack` | `pd.DataFrame` or 2D `np.ndarray` | Arbitrary numeric matrix (RNA-seq timepoints, motif scores); supports `center=0.0` (`TwoSlopeNorm`), `cluster_columns=True`, rotated `col_labels` |
| `eh.CategoricalTrack` | `pd.Series`, `pd.Categorical`, or sequence | Discrete annotation strip; `palette` accepts dict, colormap name (`"Set2"`), or color list; bottom vector legend when `show_legend=True` |
| `eh.BarTrack` | 1D `pd.Series` or `np.ndarray` | Horizontal bars (switches cleanly to filled profile above 250 rows); optional `neg_color` for signed values |
| `eh.PointTrack` | 1D `pd.Series` or `np.ndarray` | Scatter track with optional running-mean vector trendline (`smooth_line=True`, `smooth_window`) |

## Shared color scales (`scale_group`)

Give multiple tracks the same `scale_group` string (e.g., WT vs Mutant):

```python
hm = (
    eh.EnrichedTrack(wt, cmap="Blues", scale_group="ATAC (CPM)", top_profile="mean_ci")
    + eh.EnrichedTrack(mut, cmap="Blues", scale_group="ATAC (CPM)", top_profile="mean_ci")
    + eh.EnrichedTrack(mut.log2_ratio(wt), name="log2(Mut/WT)")
)
```

Tracks in a `scale_group` automatically share:
1. One pooled `Normalize` / `TwoSlopeNorm` (`vmin`, `vmax`).
2. One shared colorbar (spanning contiguous group columns).
3. Synchronized top-metaplot y-axis limits (unless overridden via `top_profile_ylim`).

## Differential track auto-defaults

When a `NormalizedMatrix` comes from `.log2_ratio()` or `.subtract()`
(`metadata["is_differential"] == True`), `eh.EnrichedTrack(diff)` automatically
selects `cmap="RdBu_r"` and `center=0.0` (`TwoSlopeNorm`) unless explicitly
overridden.

## Pitfalls

- `eh.HeatmapList` indexes tracks by position (`hm[0]`) or by `track.name` (`hm["ATAC-seq"]`). Ensure track names are unique if you reference them in `order_by` or `cluster_tracks`.
- If you pass a plain NumPy array (without `region_index`) to `eh.MatrixTrack`, `eh.CategoricalTrack`, `eh.BarTrack`, or `eh.PointTrack`, its index defaults to `RangeIndex(n)` — make sure that matches the other tracks' `region_index`, or pass a `pd.Series`/`pd.DataFrame` indexed by the exact region IDs.
