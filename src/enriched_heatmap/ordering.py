"""Per-region scores and clusterings. Every function returns a pandas object indexed by region."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from enriched_heatmap._matrix import NormalizedMatrix

OrderMetric = (
    Literal["mean", "median", "max", "sum", "center_mean", "center_max", "enriched_score"]
    | Callable[[NormalizedMatrix], np.ndarray]
)


_HCLUST_WARN_ROWS = 20_000


class HClust(NamedTuple):
    """Result of :func:`hclust`; unpacks as ``order, clusters`` and can be given to
    :func:`arrange` directly.

    Attributes
    ----------
    order : pandas.Index
        Region identifiers in dendrogram leaf order.
    clusters : pandas.Series or None
        Ordered categorical cluster labels, ``None`` when ``k`` was not given.
    """

    order: pd.Index
    clusters: pd.Series | None


def _center_bin_mask(
    matrix: NormalizedMatrix, center_window: float | None = None
) -> np.ndarray:
    """Return boolean mask selecting central / target bins."""
    centers = matrix.bin_centers()
    if matrix.n_bins == 0:
        return np.zeros(0, dtype=bool)

    if matrix.mode == "scale_regions":
        mask = (centers >= 0.0) & (centers <= matrix.body_length)
        if np.any(mask):
            return mask
        return np.ones(matrix.n_bins, dtype=bool)

    if center_window is None:
        span = float(matrix.upstream + matrix.downstream)
        half_win = min(150.0, 0.15 * span) if span > 0 else float(matrix.bin_size)
    else:
        half_win = float(center_window)

    mask = np.abs(centers) <= half_win
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(centers)))
        mask[nearest] = True
    return mask


def _enriched_score(matrix: NormalizedMatrix) -> np.ndarray:
    """Compute center-weighted enrichment score inspired by R's EnrichedHeatmap."""
    vals = np.asarray(matrix.values, dtype=np.float64)
    n_bins = matrix.n_bins
    if n_bins == 0:
        return np.zeros(matrix.n_regions, dtype=np.float64)

    centers = matrix.bin_centers()
    weights = np.zeros(n_bins, dtype=np.float64)

    if matrix.mode == "reference_point":
        up_mask = centers < 0.0
        down_mask = centers >= 0.0
        n_up = int(up_mask.sum())
        n_down = int(down_mask.sum())
        if n_up > 0:
            weights[up_mask] = np.linspace(1.0 / n_up, 1.0, n_up)
        if n_down > 0:
            weights[down_mask] = np.linspace(1.0, 1.0 / n_down, n_down)
    else:
        up_mask = centers < 0.0
        body_mask = (centers >= 0.0) & (centers <= matrix.body_length)
        down_mask = centers > matrix.body_length
        n_up = int(up_mask.sum())
        n_down = int(down_mask.sum())
        if n_up > 0:
            weights[up_mask] = np.linspace(1.0 / n_up, 1.0, n_up)
        if np.any(body_mask):
            weights[body_mask] = 1.0
        if n_down > 0:
            weights[down_mask] = np.linspace(1.0, 1.0 / n_down, n_down)

    finite = np.isfinite(vals)
    num = np.where(finite, vals * weights[None, :], 0.0).sum(axis=1)
    den = np.where(finite, weights[None, :], 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _scores(
    matrix: NormalizedMatrix,
    metric: OrderMetric = "center_mean",
    center_window: float | None = None,
) -> np.ndarray:
    """Compute a 1D per-region ordering score for a :class:`NormalizedMatrix`.

    Parameters
    ----------
    matrix : NormalizedMatrix
        Input signal matrix.
    metric : str | Callable, default "center_mean"
        Scoring strategy: ``"center_mean"``, ``"center_max"``, ``"mean"``,
        ``"median"``, ``"max"``, ``"sum"``, ``"enriched_score"``, or a callable
        ``fn(matrix) -> 1D np.ndarray``.
    center_window : float | None, default None
        Half-window in bp around 0 for ``"center_mean"`` / ``"center_max"`` in
        reference-point mode (defaults to ``min(150, 15% of span)``).
    """
    if callable(metric):
        scores = np.asarray(metric(matrix), dtype=np.float64)
        if scores.shape != (matrix.n_regions,):
            raise ValueError(
                f"Custom `order_metric` must return shape ({matrix.n_regions},), "
                f"got {scores.shape}."
            )
        return scores

    vals = np.asarray(matrix.values, dtype=np.float64)
    if matrix.n_regions == 0:
        return np.empty(0, dtype=np.float64)

    with np.errstate(invalid="ignore", all="ignore"):
        if metric == "mean":
            return np.nanmean(vals, axis=1)
        if metric == "median":
            return np.nanmedian(vals, axis=1)
        if metric == "max":
            out = np.full(matrix.n_regions, np.nan, dtype=np.float64)
            valid = np.any(np.isfinite(vals), axis=1)
            if np.any(valid):
                out[valid] = np.nanmax(vals[valid], axis=1)
            return out
        if metric == "sum":
            valid = np.any(np.isfinite(vals), axis=1)
            s = np.nansum(vals, axis=1)
            return np.where(valid, s, np.nan)
        if metric == "center_mean":
            mask = _center_bin_mask(matrix, center_window=center_window)
            return np.nanmean(vals[:, mask], axis=1)
        if metric == "center_max":
            mask = _center_bin_mask(matrix, center_window=center_window)
            sub = vals[:, mask]
            out = np.full(matrix.n_regions, np.nan, dtype=np.float64)
            valid = np.any(np.isfinite(sub), axis=1)
            if np.any(valid):
                out[valid] = np.nanmax(sub[valid], axis=1)
            return out
        if metric == "enriched_score":
            return _enriched_score(matrix)

    raise ValueError(
        f"Unsupported `order_metric` {metric!r}. Choose from "
        "'center_mean', 'center_max', 'mean', 'median', 'max', 'sum', "
        "'enriched_score', or a callable."
    )


def _extract_feature_block(
    item: NormalizedMatrix | pd.DataFrame | np.ndarray,
    region_index: pd.Index | None = None,
    scale: Literal["zscore", "row_zscore", "none"] = "zscore",
) -> tuple[np.ndarray, pd.Index]:
    """Convert a matrix/DataFrame/ndarray into a NaN-imputed, scaled 2D feature block."""
    if isinstance(item, NormalizedMatrix):
        aligned = item.reindex(region_index) if region_index is not None else item
        vals = aligned.values.astype(np.float64, copy=True)
        idx = aligned.region_index
    elif isinstance(item, pd.DataFrame):
        aligned_df = item.reindex(region_index) if region_index is not None else item
        vals = aligned_df.to_numpy(dtype=np.float64, copy=True)
        idx = aligned_df.index
    else:
        arr = np.asarray(item, dtype=np.float64)
        vals = arr[:, None].copy() if arr.ndim == 1 else arr.copy()
        idx = region_index if region_index is not None else pd.RangeIndex(vals.shape[0])

    # Impute NaNs column-wise with median (fallback 0.0)
    if np.any(~np.isfinite(vals)):
        with np.errstate(invalid="ignore", all="ignore"):
            col_med = np.nanmedian(vals, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad_r, bad_c = np.where(~np.isfinite(vals))
        vals[bad_r, bad_c] = col_med[bad_c]

    if scale == "zscore":
        mu = vals.mean()
        sd = vals.std()
        if sd > 0:
            vals = (vals - mu) / sd
    elif scale == "row_zscore":
        mu = vals.mean(axis=1, keepdims=True)
        sd = vals.std(axis=1, keepdims=True)
        vals = np.where(sd > 0, (vals - mu) / sd, 0.0)
    elif scale != "none":
        raise ValueError(f"`scale` must be 'zscore', 'row_zscore', or 'none', got {scale!r}.")

    return vals, idx


def _build_joint_features(
    matrices: NormalizedMatrix | pd.DataFrame | np.ndarray | Sequence[Any],
    scale: Literal["zscore", "row_zscore", "none"] = "zscore",
) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    """Stack one or more matrices horizontally into a joint clustering feature matrix."""
    if isinstance(matrices, (NormalizedMatrix, pd.DataFrame, np.ndarray)):
        items = [matrices]
    elif isinstance(matrices, dict):
        items = list(matrices.values())
    else:
        items = list(matrices)
    if not items:
        raise ValueError("`matrices` must contain at least one matrix.")

    first_vals, base_index = _extract_feature_block(items[0], region_index=None, scale=scale)
    # Also keep unscaled primary mean per row for deterministic cluster ordering
    if isinstance(items[0], NormalizedMatrix):
        primary_raw = np.nanmean(items[0].values, axis=1)
    elif isinstance(items[0], pd.DataFrame):
        primary_raw = np.nanmean(items[0].to_numpy(dtype=np.float64), axis=1)
    else:
        arr0 = np.asarray(items[0], dtype=np.float64)
        primary_raw = np.nanmean(arr0 if arr0.ndim == 2 else arr0[:, None], axis=1)

    blocks = [first_vals]
    for item in items[1:]:
        blk, _ = _extract_feature_block(item, region_index=base_index, scale=scale)
        # Weight each matrix equally by 1 / sqrt(n_features) so wide matrices don't dominate
        blocks.append(blk)

    if len(blocks) > 1:
        blocks = [b / np.sqrt(max(b.shape[1], 1)) for b in blocks]

    X = np.hstack(blocks)
    return X, base_index, primary_raw


def _numpy_kmeans(
    X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, random_state: int = 42
) -> np.ndarray:
    """Lightweight k-means++ fallback when scikit-learn is not installed.

    Seeded and deterministic, but it draws random numbers differently from
    scikit-learn, so the two give different (equally valid) clusterings for one seed.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    x2 = np.einsum("ij,ij->i", X, X)
    best_labels = np.zeros(n_samples, dtype=int)
    best_inertia = np.inf

    for _ in range(max(1, n_init)):
        # k-means++ initialization
        centers = np.empty((k, X.shape[1]), dtype=np.float64)
        first = int(rng.integers(0, n_samples))
        centers[0] = X[first]
        min_d2 = np.sum((X - centers[0]) ** 2, axis=1)
        for c in range(1, k):
            total = min_d2.sum()
            if total <= 0:
                idx = int(rng.integers(0, n_samples))
            else:
                probs = min_d2 / total
                idx = int(rng.choice(n_samples, p=probs))
            centers[c] = X[idx]
            d2 = np.sum((X - centers[c]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        labels = np.zeros(n_samples, dtype=int)
        for _it in range(max_iter):
            # |x|^2 - 2 x.c + |c|^2 keeps memory at n_samples x k, not n_samples x k x d
            dists = x2[:, None] - 2.0 * (X @ centers.T) + np.einsum("ij,ij->i", centers, centers)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centers[c] = X[mask].mean(axis=0)
                else:
                    centers[c] = X[int(rng.integers(0, n_samples))]

        inertia = float(np.sum((X - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def kmeans(
    matrices: NormalizedMatrix | pd.DataFrame | np.ndarray | Sequence[Any],
    k: int,
    random_state: int = 42,
    scale: Literal["zscore", "row_zscore", "none"] = "zscore",
    n_init: int = 10,
    order_clusters: Literal["descending", "ascending", "none"] = "descending",
    label_prefix: str = "C",
) -> pd.Series:
    """k-means on one or several matrices, joined on region.

    Parameters
    ----------
    matrices : NormalizedMatrix, DataFrame, array, or a list/dict of them
        Features are the bins of each matrix side by side. Later matrices are aligned to
        the regions of the first; NaNs are imputed with the bin median.
    k : int
        Number of clusters.
    scale : {"zscore", "row_zscore", "none"}
        ``"zscore"`` standardises each matrix as a whole, so tracks with different
        units weigh equally. ``"row_zscore"`` clusters on profile shape only.
    order_clusters : {"descending", "ascending", "none"}
        Number clusters by the mean of the first matrix, strongest first by default.
    random_state : int
        Seed. Results are reproducible within one install, but scikit-learn and the
        NumPy fallback (used when scikit-learn is missing) differ for the same seed.
    label_prefix : str
        Labels are ``C1``, ``C2``, ... by default.

    Returns
    -------
    pandas.Series
        Ordered categorical, indexed by region. Pass it as ``groups=`` to :func:`arrange`.
    """
    if k < 1:
        raise ValueError("`k` must be >= 1.")

    X, region_index, primary_raw = _build_joint_features(matrices, scale=scale)
    n_regions = X.shape[0]
    k_eff = min(int(k), max(1, n_regions))

    if k_eff == 1 or n_regions <= 1:
        labels = np.zeros(n_regions, dtype=int)
    else:
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k_eff, n_init=n_init, random_state=random_state)
            labels = km.fit_predict(X)
        except ImportError:
            labels = _numpy_kmeans(X, k=k_eff, n_init=n_init, random_state=random_state)

    unique_raw = np.unique(labels)
    if order_clusters in ("descending", "ascending"):
        with np.errstate(invalid="ignore", all="ignore"):
            cluster_means = {
                c: float(np.nanmean(primary_raw[labels == c])) for c in unique_raw
            }
        ordered_raw = sorted(
            unique_raw.tolist(),
            key=lambda c: cluster_means[c] if np.isfinite(cluster_means[c]) else -np.inf,
            reverse=(order_clusters == "descending"),
        )
    else:
        ordered_raw = unique_raw.tolist()

    remap = {old: f"{label_prefix}{new_i + 1}" for new_i, old in enumerate(ordered_raw)}
    categories = [f"{label_prefix}{i + 1}" for i in range(len(ordered_raw))]
    mapped = [remap[int(c)] for c in labels]
    cat = pd.Categorical(mapped, categories=categories, ordered=True)
    return pd.Series(cat, index=region_index, name="cluster")


def hclust(
    matrices: NormalizedMatrix | pd.DataFrame | np.ndarray | Sequence[Any],
    k: int | None = None,
    method: str = "ward",
    metric: str = "euclidean",
    scale: Literal["zscore", "row_zscore", "none"] = "zscore",
    optimal_ordering: bool = False,
    label_prefix: str = "C",
) -> HClust:
    """Perform hierarchical clustering across one or multiple aligned tracks.

    Memory grows with the square of the number of regions (all pairwise distances);
    beyond about 20,000 regions a ``ResourceWarning`` is raised. Prefer :func:`kmeans`
    there.

    Returns
    -------
    HClust
        ``order``: region IDs in dendrogram leaf order. ``clusters``: ordered
        categorical Series if ``k`` is given, else ``None``. Unpack it
        (``order, clusters = hclust(...)``) or pass it as ``eh.arrange(hc)``, which
        uses the leaf order and, when present, the clusters as groups.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for `hclust`. "
            "Install it with: pip install 'pyenrichedheatmap[stats]'"
        ) from exc

    X, region_index, _ = _build_joint_features(matrices, scale=scale)
    n_regions = X.shape[0]
    if n_regions <= 1:
        clusters = (
            pd.Series(
                pd.Categorical([f"{label_prefix}1"] * n_regions, ordered=True),
                index=region_index,
                name="cluster",
            )
            if k is not None
            else None
        )
        return HClust(region_index.copy(), clusters)

    if n_regions > _HCLUST_WARN_ROWS:
        gb = n_regions * (n_regions - 1) / 2 * 8 / 1e9
        warnings.warn(
            f"hclust on {n_regions:,} regions builds a distance matrix of about {gb:.0f} GB. "
            "Cluster a subsample, or use kmeans for large tables.",
            ResourceWarning,
            stacklevel=2,
        )

    Z = linkage(X, method=method, metric=metric, optimal_ordering=optimal_ordering)
    leaf_order = leaves_list(Z)
    ordered_index = region_index[leaf_order]

    if k is None:
        return HClust(ordered_index, None)

    raw_clusters = fcluster(Z, t=min(int(k), n_regions), criterion="maxclust")
    # Preserve visual top-to-bottom appearance order along the dendrogram
    seen_order: list[int] = []
    for c in raw_clusters[leaf_order]:
        ci = int(c)
        if ci not in seen_order:
            seen_order.append(ci)
    remap = {old: f"{label_prefix}{i + 1}" for i, old in enumerate(seen_order)}
    categories = [f"{label_prefix}{i + 1}" for i in range(len(seen_order))]
    mapped = [remap[int(c)] for c in raw_clusters]
    cluster_series = pd.Series(
        pd.Categorical(mapped, categories=categories, ordered=True),
        index=region_index,
        name="cluster",
    )
    return HClust(ordered_index, cluster_series)


def score(
    matrix: NormalizedMatrix,
    metric: OrderMetric = "center_mean",
    center_window: float | None = None,
) -> pd.Series:
    """One number per region, for sorting rows.

    Parameters
    ----------
    matrix : NormalizedMatrix
    metric : str or callable
        ``"center_mean"`` and ``"center_max"`` use bins within ``center_window`` bp of
        the anchor (reference-point) or the body bins (scale-regions). ``"mean"``,
        ``"median"``, ``"max"`` and ``"sum"`` use every bin. ``"enriched_score"`` is
        R EnrichedHeatmap's weighting: flank bins count less the further they are from
        the anchor. A callable gets the matrix and returns one value per region.
    center_window : float, optional
        Half-width in bp for the ``center_*`` metrics. Default: ``min(150, 15%`` of the
        window``)``.

    Returns
    -------
    pandas.Series
        Indexed by region, named after the matrix. NaN where a region has no data.
    """
    vals = _scores(matrix, metric=metric, center_window=center_window)
    return pd.Series(vals, index=matrix.region_index, name=matrix.name or "score")
