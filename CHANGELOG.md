# Changelog

## 0.5.0

### Added

- `plot(vmin=)`: per-column lower colour scale bound (a scalar or a dict keyed by
  column name, matching `vmax=` and `center=`).
- `annotation(kind="categorical" | "numeric")`: explicit discrete or continuous mode to
  override `kind="auto"`, e.g. so integer cluster codes draw as discrete categories
  sorted naturally (`0, 1, ..., 10`). In `plot` it is `annotation_kind=` (a value or a
  dict keyed by strip name).
- `limits(..., rows=)`: aligns each matrix to `rows.index` before computing shared
  `(vmin, vmax)` bounds.
- `NormalizedMatrix.to_dict` and `from_dict` preserve `region_index.name` (`region_index_name`
  key, backwards-compatible with older dicts).
- `plot(widths=)`: relative width of each heatmap column, multiplied by `width`
  (a list in `mats` order or a dict keyed by column name; default all 1), e.g.
  `widths=[1, 1, 0.5]`. Profiles and colour bars follow; annotation strips keep
  `annotation_width`.
- `plot` colour scales per annotation strip: `annotation_cmap`, `annotation_vmin`,
  `annotation_vmax` and `annotation_center` take a value or a dict keyed by strip name
  (like `cmap`/`vmax` for heatmap columns), and every group block of a strip uses them.
  Numeric strips get a colour bar underneath (`annotation_colorbar=False` to drop it),
  with the strip name under the bar instead of under the strip, and the figure grows
  at the bottom and right to fit labels, colour bars, and stacked legends cleanly under
  plain `fig.savefig(...)`.
- `plot(annotation_colors=)` also takes a dict keyed by strip name, one palette per
  categorical strip. A shared palette (the default `"Set2"`) is laid over the labels
  of all categorical strips together: a label has the same colour on every strip, and
  different labels get different colours. With several strips whose labels differ,
  colours therefore change from 0.4.0, where every strip restarted the palette.
- Type annotations on the public API, a `py.typed` marker, and `mypy` in CI (default
  settings, not `--strict`). `kmeans`, `hclust`, `limits` and `plot` accept any
  `Mapping` of matrices. `[project.urls]` metadata.

Review fixes. Behaviour changes are marked **(breaking)**.

### Fixed

- **(breaking)** Region identifiers must be unique, non-missing and not boolean, checked
  once in `NormalizedMatrix` and `bins`. A NaN/None ID cannot be matched reliably across
  panels and `True` hashes equal to `1`, so both used to give silent mismatches.
- **(breaking)** `matrices` and `matrix` reject duplicate or missing `intervalID` values,
  duplicate or missing `(region, bin)` pairs, boolean/fractional/non-finite index dtypes,
  and duplicate value column names instead of silently overwriting bins.
- **(breaking)** `kmeans`, `hclust`, `heatmap`, `profile`, and `limits(..., rows=)`
  raise `ValueError` when a non-empty matrix shares zero `region_index` identifiers with
  the primary matrix or `rows` (matching `annotation`). Lookups never match a
  MultiIndex by prefix or an IntervalIndex by containment.
- **(breaking)** `plot` validates keys in `cmap`, `center`, `vmin`, `vmax`,
  `annotation_cmap`, `annotation_vmin`, `annotation_vmax`, `annotation_center`, and
  `annotation_colors` upfront, aligns `shared` groups to `rows`, honours consistent
  `vmin`/`vmax`/`center` overrides on `shared` groups, and raises on conflicting bounds.
- **(breaking)** `plot` raises on duplicate matrix names in a list of `mats`; the later
  matrix used to replace the earlier one, dropping a column without warning.
- **(breaking)** `bins(snap=)` takes a positive integer only. `snap=0` used to mean no
  snapping and now raises (pass `snap=None`); booleans and floats such as `20.0` also
  raise across integer parameters (`upstream`, `downstream`, `bin_size`, `body_bins`,
  `max_rows`, `k`, `n_boot`).
- **(breaking)** `read_bigwig` clips bins before 0 or past chromosome ends to `NaN`
  instead of crashing `pyBigWig`, and rejects empty path mappings or non-path inputs.
- **(breaking)** `Rows` and `arrange` validate finite `0 <= gap < inf` upfront and
  require `groups` to be a `pandas.Series` (`TypeError` otherwise), raising on duplicate
  region identifiers. Empty grouped `Rows` and 0-row `kmeans`/`hclust` run warning-free.
- **(breaking)** `profile` and `compute_profile` validate `0 < ci < 1` and positive
  integer `n_boot`/`n_bootstrap` upfront (so `ci=95` raises instead of inverting bounds).
- **(breaking)** `agg="qXX"` (`downsample_rows`, `heatmap`, `plot`) is always a
  percentile in [0, 100]: `"q1"` is the 1st percentile (it used to be the maximum),
  `"q99.9"` the 99.9th and `"q0.5"` the 0.5th. Anything else raises.
- `legends` measures each legend's bounding box so stacked legends never overlap on
  compact figure heights, and `heatmap(frame=True)` / `annotation(kind="bar")` hide ticks
  on shared y-axes without wiping `group_labels`.
- `plot(widths=)` rejects NaN and infinite factors; `heatmap` and `annotation` validate
  finite `vmin`, `vmax`, `center`, and `q` bounds.
- `plot(annotation_colors=)`: a colormap name that is also a colour name (`"pink"`,
  `"gray"`) counts as a per-strip palette, and a misspelt colormap or strip name raises
  instead of turning the dict into a shared `{category: colour}` mapping.
- `m.loc[bool_series]` aligns the mask to `region_index` by label (raising on NA,
  duplicate labels, wrong length, or mismatched index; pass `.to_numpy()` for positional
  masks). `m.loc[a:b]` follows pandas label semantics; `m.iloc` rejects boolean scalars
  and wrong-length masks; `smooth` validates positive finite `sigma`.
- **(breaking)** `bins` raises on missing chromosome names or 0-bin layouts. `matrices`
  raises when `values["bin"]` is outside the layout and when positional `values`
  chromosomes differ from `bins`.
- `bins(snap=)`: a zero-length region with `anchor="start"` on - or `"end"` on + no
  longer anchors before its own start.
- `enriched_score` on a centred matrix tapers the flank weights to just below the
  middle bin instead of repeating 1.0 on three bins.
- `subtract` and `log2_ratio` also compare the column count.
- With `center=` and only one of `vmin` or `vmax`, the other bound mirrors it around
  the centre instead of coming from the data quantile. A bound on the wrong side of an
  explicit `center` raises (it used to be clamped to the centre).
- The automatic `center=0` of differential matrices gives way to a one-sided bound:
  `vmin >= 0` or `vmax <= 0` draws a plain linear scale, so `plot(..., vmin=0)` works
  with a ratio column next to signal columns.
- `palette` takes a Colormap object and raises on an empty list. `limits` takes a single
  matrix and raises on none. `plot` raises on empty `mats` or malformed `shared`.
  `hclust(k=0)` raises. An unnamed Series annotation no longer gets the label "0".
- Bundled skill `check_regions.py` supports gzipped `.narrowPeak.gz`/`.broadPeak.gz`/
  `.gappedPeak[.gz]`/`.csv.gz`/`.tsv.gz`, catches missing/non-numeric coordinates,
  validates `--id` and window geometry cleanly, and accepts `"1.0"`/`"-1.0"` strands.
- All-NaN rows no longer emit RuntimeWarnings from scoring, clustering, `zscore` or
  `downsample_rows`.

## 0.4.0

### Fixed

- `plot(annotations=...)` raised "truth value of a DataFrame is ambiguous" on a
  DataFrame. It now accepts a dict of name -> Series, a DataFrame (one strip per
  column) or a single Series, and raises a clear TypeError on anything else.
- **(breaking)** `annotation` (and so `plot`) raises when the data shares no region
  identifiers with `rows`, e.g. a Series built from `.values` with a RangeIndex. It
  used to draw a silent blank strip. Partial overlap still draws missing regions blank.
- **(breaking)** `bins(snap=)` put the true anchor half a stored bin right of the
  dashed line on average, and stranded regions one bin apart: the anchor's stored bin
  landed right of 0 on + rows but left of 0 on - rows. `anchor="start"` on - and
  `anchor="end"` on + also used the `end` boundary instead of the last base. With
  `snap` the layout is now centred: `upstream/bin_size + 1 + downstream/bin_size`
  bins, the middle one holding the stored bin with the anchor base on both strands,
  x = 0 at its middle. `bin_size` must be an odd multiple of `snap`, and `snap` with
  `body_bins` raises.

### Added

- `NormalizedMatrix(centered=True)`: reference-point geometry with a middle anchor
  bin, kept through every transform, `subtract`/`log2_ratio` checks and `to_dict`.
- `plot(guides=False)` turns off the dashed anchor lines in heatmaps and profiles.

## 0.3.0

Fixes from a full review of the package. Behaviour changes are marked **(breaking)**.

### Fixed

- `arrange(groups=...)` crashed on boolean group labels and could misorder integer
  ones; group ranks no longer go through `Series.loc`.
- `NormalizedMatrix.smooth` filled NaN bins from their neighbours, so off-chromosome
  bins were drawn as data. NaN bins now stay NaN. Unknown boundary modes raise, and the
  NumPy fallback (no scipy) now agrees with scipy for every mode.
- `annotation` drew only the first column of a multi-column DataFrame when it was
  categorical, raised on mixed numeric and categorical columns, and ignored a mistyped
  `kind`. Every column now picks categorical or numeric on its own, and `kind` is
  validated. An empty categorical strip no longer raises.
- The `eh.bins` layout in `DataFrame.attrs` made `DataFrame.merge` raise on pandas 3
  (it compared the stored region Index). It is now compared as a plain bool and shared
  on `copy`/`deepcopy`. pandas 2.x still drops `attrs` in `merge`/`concat`; the pivot
  then reports "lost its layout".
- `eh.bins` silently dropped regions whose start or end was NaN; it now raises.
- `matrices` always marked matrices as strand-oriented; the flag follows the `strand`
  argument of `bins`.
- `NormalizedMatrix == NormalizedMatrix` raised; matrices now compare by identity.
- `example_peaks(n=...)` failed with an obscure NumPy error for large `n`.
- `example_values(bins)` silently generated signal for the default peaks when the bins
  came from another table; it now raises and asks for `peaks=`.
- Zero-row heatmaps and annotations no longer warn about identical axis limits.
- The `legend` of categorical annotation strips in `plot` is not clipped by the axes.

### Changed

- **(breaking)** `NormalizedMatrix` validates that the column count equals
  `upstream/bin_size + target_bins + downstream/bin_size`, and that the flanks are
  multiples of `bin_size`. A mismatch used to give silently wrong tick positions.
- **(breaking)** `subtract` and `log2_ratio` raise on matrices with different mode,
  flanks, bin size or `target_bins` instead of combining them.
- **(breaking)** The default `colors` of `annotation` is `"tab10"` (was `"Set2"`), the
  same as `profile`, so a group strip and the profile lines agree.
- **(breaking)** The modules `enriched_heatmap.bins` and `enriched_heatmap.matrix` are
  now private (`_bins`, `_matrix`): the names belong to the functions `eh.bins` and
  `eh.matrix`. Import from `enriched_heatmap`.
- **(breaking)** `compute_target_raster_rows` is removed (it was unused).
- `heatmap` and `limits` centre differential matrices (from `subtract` and
  `log2_ratio`) on 0 with `RdBu_r` unless `center=` or `cmap=` says otherwise.
- `annotation` returns, for several numeric columns, the image of the last one (all
  share one norm); with any categorical column it returns `{column: result}`.
- `hclust` raises a `ResourceWarning` above 20,000 regions (distance matrix memory).
- `Rows` validates its groups (aligned to the index, contiguous, no missing labels)
  instead of drawing non-contiguous groups wrongly; `spans()` runs in linear time.
- `heatmap(max_rows="auto")` runs a constrained or tight layout before measuring the
  axes; it undersampled by about 20% before.
- The bootstrap band resamples through multinomial counts and matrix products: same
  statistics, much faster and flat in memory (50,000 x 150 in under a second).
- The NumPy k-means fallback no longer builds an n x k x d array.
- Empty-bin RuntimeWarnings from the summary profiles are suppressed.

### Added

- `hclust` returns an `HClust` named tuple (`order`, `clusters`), still unpackable;
  `arrange(hc)` takes it directly and uses its clusters as groups.
- `figure(..., pyplot=False)` and `plot(..., pyplot=False)` return a plain `Figure`
  that is not registered with pyplot, so loops do not leak figures. Importing the
  package no longer imports `matplotlib.pyplot`.
- `profile(..., n_boot=)` for the bootstrap band.

- `plot(..., annotation_legend=True, annotation_colors="Set2")`: categorical
  annotation strips get a legend in the right margin and their own palette.
- `eh.legends(ax, [(title, colours), ...])` stacks legends beside an axes.
- `matrices(..., dtype=)` and `matrix(..., dtype=)`; `np.float64` keeps raw counts
  above 2**24 exact.
- `py.typed`, a docs build on every pull request, and required pymisha and pyBigWig
  tests on Linux CI.

## 0.2.0

Table-based API: `bins`, `matrices`, axes-level drawing functions, `arrange`.
