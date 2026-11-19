"""Row ordering, k-means, hierarchical clustering, and cluster splitting utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from enriched_heatmap.matrix import NormalizedMatrix

OrderMetric = (
    Literal["mean", "median", "max", "sum", "center_mean", "center_max", "enriched_score"]
    | Callable[[NormalizedMatrix], np.ndarray]
)


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


def compute_order_scores(
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
    """Lightweight k-means++ fallback when scikit-learn is not installed."""
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
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
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
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


def cluster_kmeans(
    matrices: NormalizedMatrix | pd.DataFrame | np.ndarray | Sequence[Any],
    k: int,
    random_state: int = 42,
    scale: Literal["zscore", "row_zscore", "none"] = "zscore",
    n_init: int = 10,
    order_clusters: Literal["descending", "ascending", "none"] = "descending",
    label_prefix: str = "C",
) -> pd.Series:
    """Cluster regions with k-means across one or multiple aligned tracks.

    Returns an ordered Categorical :class:`pandas.Series` indexed by ``region_index``
    with labels ``f"{label_prefix}1"`` .. ``f"{label_prefix}{k}"``.
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


def cluster_hierarchical(
    matrices: NormalizedMatrix | pd.DataFrame | np.ndarray | Sequence[Any],
    k: int | None = None,
    method: str = "ward",
    metric: str = "euclidean",
    scale: Literal["zscore", "row_zscore", "none"] = "zscore",
    optimal_ordering: bool = False,
    label_prefix: str = "C",
) -> tuple[pd.Index, pd.Series | None]:
    """Perform hierarchical clustering across one or multiple aligned tracks.

    Returns
    -------
    ordered_index : pd.Index
        Region IDs ordered according to the dendrogram leaves.
    clusters : pd.Series | None
        Ordered Categorical cluster Series if ``k`` is provided, otherwise ``None``.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for `cluster_hierarchical`. "
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
        return region_index.copy(), clusters

    Z = linkage(X, method=method, metric=metric, optimal_ordering=optimal_ordering)
    leaf_order = leaves_list(Z)
    ordered_index = region_index[leaf_order]

    if k is None:
        return ordered_index, None

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
    return ordered_index, cluster_series


def _sort_positions_by_scores(
    scores: np.ndarray, ascending: bool = False
) -> np.ndarray:
    """Stable sort pushing NaNs to the bottom regardless of sort direction."""
    vals = np.asarray(scores, dtype=np.float64)
    if ascending:
        # NaNs sort last by default with np.argsort
        return np.argsort(vals, kind="mergesort")
    # For descending, negate finite values and place NaNs at +inf (bottom)
    key = np.where(np.isfinite(vals), -vals, np.inf)
    return np.argsort(key, kind="mergesort")


def resolve_ordering_and_splits(
    tracks: list[Any],
    region_index: pd.Index,
    order_by: str | Callable | pd.Series | pd.Index | np.ndarray | None = None,
    order_metric: OrderMetric = "center_mean",
    ascending: bool = False,
    split_by: pd.Series | pd.Categorical | Sequence[Any] | int | None = None,
    cluster_tracks: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Series | None, list[tuple[str, slice]]]:
    """Resolve global row order and cluster splits across a track list.

    Returns
    -------
    ordered_index : pd.Index
        Final ordered region index applied uniformly to every track.
    split_series : pd.Series | None
        Cluster membership aligned to ``ordered_index`` (``None`` when unsplit).
    split_slices : list[tuple[str, slice]]
        Contiguous ``(group_label, slice)`` partitions of ``ordered_index``.
    """
    n_regions = len(region_index)
    if n_regions == 0:
        return region_index.copy(), None, []

    track_map = {t.name: t for t in tracks if t.name}

    def _pick_cluster_matrices() -> list[Any]:
        if cluster_tracks is not None:
            chosen = []
            for name in cluster_tracks:
                if name not in track_map:
                    raise KeyError(
                        f"Track {name!r} in `cluster_tracks` not found. "
                        f"Available tracks: {list(track_map.keys())}"
                    )
                chosen.append(track_map[name]._clustering_data(region_index))
            return chosen
        if isinstance(order_by, str) and order_by in track_map:
            return [track_map[order_by]._clustering_data(region_index)]
        for t in tracks:
            data = t._clustering_data(region_index)
            if data is not None:
                return [data]
        raise ValueError("No numeric tracks available for clustering.")

    # 1. Resolve split_by -> Series aligned to region_index
    split_series_raw: pd.Series | None = None
    group_order: list[Any] = []

    if split_by is not None:
        if isinstance(split_by, (int, np.integer)):
            split_series_raw = cluster_kmeans(
                _pick_cluster_matrices(), k=int(split_by), random_state=random_state
            )
            group_order = list(split_series_raw.cat.categories)
        elif isinstance(split_by, pd.Series):
            aligned = split_by.reindex(region_index)
            if isinstance(aligned.dtype, pd.CategoricalDtype):
                present = set(aligned.dropna().unique())
                group_order = [c for c in aligned.cat.categories if c in present]
            else:
                group_order = list(pd.unique(aligned.dropna()))
            split_series_raw = aligned
        elif isinstance(split_by, pd.Categorical):
            if len(split_by) != n_regions:
                raise ValueError(
                    f"`split_by` length ({len(split_by)}) does not match n_regions ({n_regions})."
                )
            s = pd.Series(split_by, index=region_index)
            present = set(s.dropna().unique())
            group_order = [c for c in split_by.categories if c in present]
            split_series_raw = s
        else:
            arr = list(split_by)
            if len(arr) != n_regions:
                raise ValueError(
                    f"`split_by` length ({len(arr)}) does not match n_regions ({n_regions})."
                )
            s = pd.Series(arr, index=region_index)
            group_order = list(pd.unique(s.dropna()))
            split_series_raw = s

    # 2. Compute ordering scores OR explicit order
    explicit_order_index: pd.Index | None = None
    global_scores: np.ndarray | None = None
    use_hierarchical_within_split = False

    if order_by is None:
        pass
    elif isinstance(order_by, str):
        if order_by in track_map:
            global_scores = track_map[order_by]._order_scores(region_index, metric=order_metric)
        elif order_by == "kmeans":
            if split_series_raw is None:
                split_series_raw = cluster_kmeans(
                    _pick_cluster_matrices(), k=3, random_state=random_state
                )
                group_order = list(split_series_raw.cat.categories)
            # Also order within each cluster by the primary numeric track
            for t in tracks:
                sc = t._order_scores(region_index, metric=order_metric)
                if sc is not None:
                    global_scores = sc
                    break
        elif order_by in ("hierarchical", "hclust"):
            use_hierarchical_within_split = True
        else:
            raise KeyError(
                f"`order_by` {order_by!r} does not match any track name "
                f"({list(track_map.keys())}) or built-in method ('kmeans', 'hierarchical')."
            )
    elif callable(order_by):
        res = order_by(track_map) if len(track_map) > 0 else order_by(region_index)
        if isinstance(res, pd.Index):
            explicit_order_index = res
        elif isinstance(res, pd.Series):
            global_scores = res.reindex(region_index).to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(res)
            if arr.shape != (n_regions,):
                raise ValueError(
                    f"Callable `order_by` returned shape {arr.shape}, expected ({n_regions},)."
                )
            global_scores = arr.astype(np.float64)
    elif isinstance(order_by, pd.Index):
        explicit_order_index = order_by
    elif isinstance(order_by, pd.Series):
        global_scores = order_by.reindex(region_index).to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(order_by)
        if arr.ndim != 1 or len(arr) != n_regions:
            raise ValueError(
                f"`order_by` must be 1D of length {n_regions}, got shape {arr.shape}."
            )
        if np.issubdtype(arr.dtype, np.number):
            # Distinguish explicit integer region_index IDs vs numeric scores:
            # treat numeric array as ordering scores unless it's explicitly pd.Index
            global_scores = arr.astype(np.float64)
        else:
            explicit_order_index = pd.Index(arr)

    # 3. Apply ordering within each split group (or globally if unsplit)
    def _order_subset(sub_pos: np.ndarray) -> np.ndarray:
        if len(sub_pos) <= 1:
            return sub_pos
        if explicit_order_index is not None:
            sub_ids = region_index[sub_pos]
            rank_map = {rid: i for i, rid in enumerate(explicit_order_index)}
            sub_ranks = np.array([rank_map.get(rid, len(rank_map)) for rid in sub_ids])
            return sub_pos[np.argsort(sub_ranks, kind="mergesort")]
        if use_hierarchical_within_split:
            sub_idx = region_index[sub_pos]
            mats = []
            for m in _pick_cluster_matrices():
                if isinstance(m, (NormalizedMatrix, pd.DataFrame)):
                    mats.append(m.reindex(sub_idx))
                else:
                    mats.append(np.asarray(m)[sub_pos])
            h_idx, _ = cluster_hierarchical(mats, k=None)
            pos_lookup = region_index.get_indexer(h_idx)
            return pos_lookup
        if global_scores is not None:
            sub_scores = global_scores[sub_pos]
            local_order = _sort_positions_by_scores(sub_scores, ascending=ascending)
            return sub_pos[local_order]
        return sub_pos

    if split_series_raw is None:
        ordered_pos = _order_subset(np.arange(n_regions, dtype=int))
        ordered_index = region_index[ordered_pos]
        return ordered_index, None, [("", slice(0, n_regions))]

    ordered_pos_parts: list[np.ndarray] = []
    split_slices: list[tuple[str, slice]] = []
    cursor = 0

    for grp in group_order:
        mask = (split_series_raw == grp).to_numpy(dtype=bool, na_value=False)
        sub_pos = np.flatnonzero(mask)
        if sub_pos.size == 0:
            continue
        sorted_sub = _order_subset(sub_pos)
        ordered_pos_parts.append(sorted_sub)
        split_slices.append((str(grp), slice(cursor, cursor + len(sorted_sub))))
        cursor += len(sorted_sub)

    # Append any NaN split rows at the bottom under "NA" if present
    na_mask = split_series_raw.isna().to_numpy(dtype=bool)
    if np.any(na_mask):
        sub_pos = np.flatnonzero(na_mask)
        sorted_sub = _order_subset(sub_pos)
        ordered_pos_parts.append(sorted_sub)
        split_slices.append(("NA", slice(cursor, cursor + len(sorted_sub))))
        cursor += len(sorted_sub)

    ordered_pos = np.concatenate(ordered_pos_parts) if ordered_pos_parts else np.array([], dtype=int)
    ordered_index = region_index[ordered_pos]
    aligned_splits = split_series_raw.reindex(ordered_index)
    return ordered_index, aligned_splits, split_slices
