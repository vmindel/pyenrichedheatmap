# Large-scale rendering, downsampling, and hybrid PDFs

Genomic heatmaps routinely contain 10,000–100,000 regions. Plotting every row
as vector rectangles makes PDFs hundreds of megabytes and freezes Illustrator;
naive pixel subsampling makes narrow TF ChEC peaks vanish.

## Smart row downsampling (`downsample_agg`)

When `n_rows` exceeds the panel's physical pixel height at `raster_dpi`,
`eh.HeatmapList.plot()` bins consecutive sorted rows into exact pixel buckets
via `eh.downsample_rows`:

| `downsample_agg` | best for | behavior |
|---|---|---|
| `"mean"` (default) | ATAC-seq, MNase-seq, histone marks | Fast C-level NaN-safe bucket average |
| `"q90"` / `"q95"` | Narrow TF ChEC-seq / ChIP-seq peaks | Preserves sharp peaks without single-outlier blowup |
| `"max"` | Ultra-sparse binary motif hits or cut sites | Preserves every narrow band |
| `"absmax"` | Differential `log2FC` heatmaps | Picks signed value with largest magnitude so `+` and `-` spikes don't cancel |
| `"median"` / `"min"` / callable | Custom summaries | Exact per-bucket aggregation |

You can set a figure-wide `downsample_agg` in `hm.plot(...)` **and** override it
per track:

```python
hm = (
    eh.EnrichedTrack(chec, cmap="Reds", downsample_agg="q90")
    + eh.EnrichedTrack(atac, cmap="Blues", downsample_agg="mean")
    + eh.EnrichedTrack(diff, downsample_agg="absmax")
)
fig = hm.plot(figsize=(11, 7), raster_dpi=300, max_raster_rows=None)
fig.savefig("figure1.pdf", dpi=300)
```

- `eh.CategoricalTrack` downsamples by bucket mode via `eh.downsample_categorical_rows`.
- `eh.BarTrack` switches to a rasterized `fill_betweenx` profile above 250 rows via `eh.downsample_1d_rows`.
- `eh.PointTrack` rasterizes scatter points above 2,000 rows while keeping its optional `smooth_line` 100% vector.

## Vector chrome + raster body

- **Rasterized:** heatmap `imshow` artists (`interpolation="none"`, `rasterized=True`).
- **100% Vector:** top metaplot lines & CI/SEM ribbons (`eh.compute_profile`), vertical guide lines, box borders, x/y ticks, titles, cluster strips, `n = ...` row counts, legends, and colorbars.
- Even 25,000–50,000 regions produce compact (<1–2 MB) Illustrator-friendly PDFs.

## Alignment `join` modes & `return_axes=True`

- `join="inner"` (default): intersects all tracks' `region_index` (raises `ValueError` if empty).
- `join="left"`: uses the first track's `region_index`, filling missing regions in later tracks with `NaN`.
- `join="outer"`: union of all tracks' `region_index`.
- Pass `return_axes=True` to receive `(fig, info)` where `info` contains `axes`, `region_order`, `splits`, `split_slices`, and `norms` for custom matplotlib annotations.

## Pitfalls

- Always pass `dpi=300` (matching `raster_dpi`) to `fig.savefig("figure.pdf", dpi=300)` so matplotlib rasterizes the body panels at full publication resolution.
- Avoid `plt.tight_layout()` on the returned figure — `HeatmapList.plot()` computes explicit deterministic nested `GridSpec` margins and spacer bands so titles, rotated labels, colorbars, and legends already have non-overlapping coordinates.
