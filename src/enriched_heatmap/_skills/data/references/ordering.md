# Row order and splits

```python
rows = eh.arrange(by=None, groups=None, metric="center_mean", ascending=False,
                  gap=0.02, index=None)   # -> eh.Rows
```

- `by`: a NormalizedMatrix (scored with `eh.score(by, metric)`), a Series of
  scores, or a `pd.Index` giving the order. `None` keeps input order.
- Highest score on top by default; NaN scores last.
- `groups`: Series of labels indexed by region (any source). Categorical ->
  category order; otherwise the group with the top-scoring row first. Rows are
  sorted by `by` within groups. Regions without a group raise KeyError.
- `index=`: restrict to a subset of regions.
- `gap`: blank space between groups, as a fraction of row count.
- `Rows` attributes: `index`, `groups`, `gap`; methods `labels`, `spans()`,
  `sizes`, `height`.

## Scores and clusters

- `eh.score(mat, metric)` -> Series. Metrics: `center_mean` (bins within
  `min(150, 15% of window)` bp of the anchor, or the body in scale-regions),
  `center_max`, `mean`, `median`, `max`, `sum`, `enriched_score` (R's
  distance-weighted score; on a `centered` matrix the flank weights stay below
  the middle bin's 1.0), or `callable(mat) -> array`.
- `eh.kmeans(mats, k, scale="zscore", random_state=42)` -> ordered categorical
  Series `C1..Ck`, numbered by mean of the first matrix, strongest first.
  `mats` can be one matrix, a list or a dict. `scale="row_zscore"` clusters
  on shape. Uses scikit-learn when installed, NumPy otherwise.
- `tree = eh.hclust(mats, k=3, method="ward")` -> `HClust(order, clusters)`:
  leaf order Index and cluster Series (None without k); unpacks as
  `order, clusters = ...`. Needs scipy and O(n^2) memory (warns above 20,000
  regions; use kmeans there). Use `eh.arrange(tree)`.
