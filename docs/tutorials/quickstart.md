# Quick start

Every workflow in `pyenrichedheatmap` follows three steps:

1. Build one or more [`NormalizedMatrix`](../api/matrix.md) objects (from BigWigs, `pymisha`, NumPy arrays, or pandas DataFrames).
2. Wrap matrices and annotations in tracks (`EnrichedTrack`, `MatrixTrack`, `CategoricalTrack`, `BarTrack`, `PointTrack`) and combine them horizontally with `+`.
3. Call `.plot(...)` to render and `.savefig("figure.pdf", dpi=300)` to export a compact vector/raster hybrid PDF.

## Minimal example

```python
import numpy as np
import pandas as pd
import enriched_heatmap as eh

# Create a synthetic ±1 kb signal matrix (200 bins at 10 bp resolution)
rng = np.random.default_rng(0)
n_regions, n_bins = 500, 200
x = np.linspace(-1000, 1000, n_bins)
amps = rng.uniform(1.0, 10.0, size=(n_regions, 1))
signal = amps * np.exp(-0.5 * (x / 120.0) ** 2)[None, :] + rng.gamma(1.0, 0.2, (n_regions, n_bins))

peak_ids = pd.Index([f"peak_{i:04d}" for i in range(n_regions)])
atac = eh.from_array(
    signal.astype(np.float32),
    region_index=peak_ids,
    upstream=1000,
    downstream=1000,
    bin_size=10,
    name="ATAC-seq",
)

# Apply non-destructive signal transforms if desired
atac_smooth = atac.log1p().smooth(sigma=1.0)

# Compose tracks with `+`
hm = (
    eh.EnrichedTrack(atac, cmap="Blues", vmax_q=0.99, top_profile="mean_ci")
    + eh.EnrichedTrack(atac_smooth, name="log1p smoothed", cmap="viridis", top_profile="mean")
)

fig = hm.plot(
    order_by="ATAC-seq",
    order_metric="center_mean",
    figsize=(8, 6),
    raster_dpi=300,
)
fig.savefig("quickstart.pdf", dpi=300)
```

## Zero-desync alignment

Every track carries a unique `region_index`. Even if a downstream DataFrame or
Series was filtered or reordered independently, `HeatmapList.plot()` aligns all
tracks on their shared `region_index` (default `join="inner"`) before sorting
or splitting:

```python
# Shuffled Series stays 100% aligned with `atac`
peak_class = pd.Series(rng.choice(["Promoter", "Enhancer"], n_regions), index=peak_ids).sample(frac=1.0)

hm = eh.EnrichedTrack(atac, cmap="Blues") + eh.CategoricalTrack(peak_class, name="Class", width=0.25)
fig = hm.plot(order_by="ATAC-seq")
```

## Customizing matplotlib axes

Pass `return_axes=True` to access the rendered axes, final region ordering,
cluster slices, and color normalizations:

```python
fig, info = hm.plot(order_by="ATAC-seq", split_by=3, return_axes=True)
ordered_ids = info["region_order"]
top_ax = info["axes"]["top"]["ATAC-seq"]
top_ax.set_ylabel("CPM")
```
