# Changelog

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
