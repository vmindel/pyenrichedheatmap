# Drawing and layout

All drawing functions: `fn(data, ax, rows=None, **style)`. Regions in `rows`
missing from the data are drawn blank/NaN.

## eh.heatmap(mat, ax, rows, ...)

`cmap` (default Reds, RdBu_r if `center`; matrices from `log2_ratio`/`subtract` get `center=0` automatically), `vmin`, `vmax`, `q=0.99` (quantile
for vmax), `center` (TwoSlopeNorm, symmetric; with only one of `vmin`/`vmax` the other mirrors it
around `center`), `norm`, `na_color`,
`max_rows="auto"` (average down to ~1 row/pixel at 300 dpi; `None` = all
rows), `agg` (`mean`, `max` for sparse narrow peaks, `absmax` for signed
differences, `median`, `q90`), `anchor_labels` (`"TSS"` or `("TSS", "TES")`),
`guides`, `xticks`, `frame`, `title`. Returns the image for a colour bar.

## eh.profile(mat, ax, rows, ...)

`stat` (`mean`/`median`), `band` (None, `sem`, `ci`, `boot` with mean; `iqr`
with median), `by_group=True` (one line per group, colours from `colors`,
default tab10), `color` (single line), `ylim`, `legend`, `title`.

## eh.annotation(series_or_df, ax, rows, ...)

Categorical -> colour strip (`colors`, default tab10 like `eh.profile`), returns
`{category: colour}`; numeric -> colour strip (`cmap`, `vmin`, `vmax`, `q`,
`center`), returns the image; `kind="bar"` -> horizontal bars (unknown `kind`
raises). A DataFrame draws one strip per column, each categorical or numeric on
its own (numeric ones share a scale); with mixed columns it returns
`{column: result}`. `legend=True` adds one legend per categorical column;
`eh.legends(ax, [(title, colours), ...])` stacks legends yourself.

Helpers: `eh.group_labels(ax, rows, counts=True)`, `eh.legend(ax, colours)`,
`eh.colorbar(image, cax, label)`, `eh.limits(mats, q, center)` -> shared
(vmin, vmax), `eh.palette(labels, colors)` -> the mapping profiles use.

## eh.figure(widths, profile=True, colorbar=True, height=5, ...)

Sizes in inches. Returns `(fig, grid)`; `grid.heatmap[i]`, `grid.profile[i]`,
`grid.colorbar[i]` (None where disabled). Heatmaps share y; each profile
shares x with its heatmap. Narrow columns (0.2-0.3) for annotation strips.

## eh.plot(mats, rows, annotations=None, ...)

One column per matrix (dict keys are titles), group strip on the left when
`rows` has groups, annotation strips on the right (`annotations=`: dict of
name -> Series, a DataFrame or a Series, indexed by region; no shared
identifiers with `rows` raises). Per-column options as a
value or dict keyed by name: `cmap`, `center`, `vmax`. `shared=[["WT",
"KO"]]` for one colour scale; `band`, `share_profile_y`, `colors`, `agg`,
`width` (inches) and `widths` (relative factor per column, list in `mats` order
or dict by name, default 1), `height`, `guides=False` (no dashed anchor lines), `pyplot=False` (plain Figure, not registered with pyplot, for
loops), `annotation_colors` (default Set2, so strips differ from the
groups; a dict keyed by strip name gives each categorical strip its own palette),
`annotation_cmap`/`annotation_vmin`/`annotation_vmax`/`annotation_center` (value or
dict keyed by strip name, for numeric strips; colour bar under each, with the
strip name below it, unless `annotation_colorbar=False`), `annotation_legend` (default True: legends for categorical strips in
the right margin). Returns `(fig, grid)`; edit axes afterwards as usual.

## Saving

`fig.savefig("x.pdf", bbox_inches="tight")`: bodies are rasterised, the rest
stays vector. Group labels sit left of the axes, so keep `bbox_inches`.
