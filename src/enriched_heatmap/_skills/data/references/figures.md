# Drawing and layout

All drawing functions: `fn(data, ax, rows=None, **style)`. Regions in `rows`
missing from the data are drawn blank/NaN.

## eh.heatmap(mat, ax, rows, ...)

`cmap` (default Reds, RdBu_r if `center`), `vmin`, `vmax`, `q=0.99` (quantile
for vmax), `center` (TwoSlopeNorm, symmetric), `norm`, `na_color`,
`max_rows="auto"` (average down to ~1 row/pixel at 300 dpi; `None` = all
rows), `agg` (`mean`, `max` for sparse narrow peaks, `absmax` for signed
differences, `median`, `q90`), `anchor_labels` (`"TSS"` or `("TSS", "TES")`),
`guides`, `xticks`, `frame`, `title`. Returns the image for a colour bar.

## eh.profile(mat, ax, rows, ...)

`stat` (`mean`/`median`), `band` (None, `sem`, `ci`, `boot` with mean; `iqr`
with median), `by_group=True` (one line per group, colours from `colors`,
default tab10), `color` (single line), `ylim`, `legend`, `title`.

## eh.annotation(series_or_df, ax, rows, ...)

Categorical -> colour strip (`colors`, default Set2), returns
`{category: colour}`; numeric -> colour strip (`cmap`, `vmin`, `vmax`, `q`,
`center`), returns the image; `kind="bar"` -> horizontal bars. A DataFrame
draws one strip per column. `legend=True` for categories.

Helpers: `eh.group_labels(ax, rows, counts=True)`, `eh.legend(ax, colours)`,
`eh.colorbar(image, cax, label)`, `eh.limits(mats, q, center)` -> shared
(vmin, vmax), `eh.palette(labels, colors)` -> the mapping profiles use.

## eh.figure(widths, profile=True, colorbar=True, height=5, ...)

Sizes in inches. Returns `(fig, grid)`; `grid.heatmap[i]`, `grid.profile[i]`,
`grid.colorbar[i]` (None where disabled). Heatmaps share y; each profile
shares x with its heatmap. Narrow columns (0.2-0.3) for annotation strips.

## eh.plot(mats, rows, annotations=None, ...)

One column per matrix (dict keys are titles), group strip on the left when
`rows` has groups, annotation strips on the right. Per-column options as a
value or dict keyed by name: `cmap`, `center`, `vmax`. `shared=[["WT",
"KO"]]` for one colour scale; `band`, `share_profile_y`, `colors`, `agg`,
`width`, `height`. Returns `(fig, grid)`; edit axes afterwards as usual.

## Saving

`fig.savefig("x.pdf", bbox_inches="tight")`: bodies are rasterised, the rest
stays vector. Group labels sit left of the axes, so keep `bbox_inches`.
