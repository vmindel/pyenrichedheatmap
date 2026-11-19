# pyenrichedheatmap

Enriched heatmaps (signal around a set of regions, one row per region) from pandas
DataFrames, drawn on matplotlib axes. The layout follows R's
[EnrichedHeatmap](https://github.com/jokergoo/EnrichedHeatmap); the data model follows
[pymisha](https://pypi.org/project/pymisha/): everything in and out is a table.

```python
import pymisha as pm
import enriched_heatmap as eh

b = eh.bins(peaks, upstream=1000, downstream=1000, bin_size=20, genome=pm.gintervals_all())
vals = pm.gextract(["chec.msn2", "atac.wt"], intervals=b, iterator=b, colnames=["msn2", "atac"])
mats = eh.matrices(vals, b)

rows = eh.arrange(mats["msn2"], groups=eh.kmeans(mats, k=3))
fig, grid = eh.plot(mats, rows=rows, annotations={"class": peaks["class"]})
```

![Overview](images/overview.png)

- **Tables in.** `eh.bins` turns regions into one row per bin, anchored to each region
  and oriented by strand. Any extraction that returns a value per bin fills it:
  `gextract` with your own expressions and virtual tracks, `eh.read_bigwig`, or your
  own code. `eh.matrices` pivots the result.
- **Functions on axes.** `eh.heatmap`, `eh.profile` and `eh.annotation` each draw one
  panel on an axes you provide, so any matplotlib layout works. `eh.figure` and
  `eh.plot` cover the usual one.
- **One object for alignment.** `eh.arrange` returns the row order and split; pass it
  to every panel. Data are matched to it by region identifier, never by position.
- **Small vector PDFs.** Heatmap bodies are rasterised and averaged to the pixel
  height, everything else stays vector.

## Where next

| | |
|---|---|
| [Quick start](tutorials/quickstart.md) | from a peak table to a clustered figure |
| [Regions to matrices](tutorials/data.md) | anchors, strand, scale-regions, bigWigs, transforms |
| [With pymisha](tutorials/pymisha.md) | track expressions and virtual tracks into matrices |
| [Building figures](tutorials/figures.md) | ordering, splits, custom layouts, shared and diverging scales |
| [Coming from R](tutorials/from-r.md) | EnrichedHeatmap and ComplexHeatmap equivalents |
| [Claude Code skill](claude-skill.md) | the bundled Agent Skill |
| [API reference](api/data.md) | every public function |
