"""Tests for row ordering metrics, k-means, hierarchical clustering, and split resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd

import enriched_heatmap as eh
from enriched_heatmap.ordering import resolve_ordering_and_splits


def _make_clustered_matrices():
    """Create 3 cleanly separated synthetic clusters (high center, flank, low)."""
    rng = np.random.default_rng(7)
    n_per = 20
    n_bins = 40
    x = np.linspace(-1000, 1000, n_bins)

    c1 = 10.0 * np.exp(-0.5 * (x / 120.0) ** 2)[None, :] + rng.normal(0, 0.2, (n_per, n_bins))
    c2 = 6.0 * np.exp(-0.5 * ((x - 450.0) / 140.0) ** 2)[None, :] + rng.normal(0, 0.2, (n_per, n_bins))
    c3 = rng.normal(0.5, 0.15, (n_per, n_bins))

    vals = np.vstack([c1, c2, c3]).astype(np.float32)
    idx = pd.Index([f"r_{i}" for i in range(3 * n_per)])
    mat1 = eh.from_array(vals, region_index=idx, upstream=1000, downstream=1000, bin_size=50, name="ChEC")

    # Second track correlated with cluster identity
    vals2 = np.vstack([
        np.full((n_per, n_bins), 5.0),
        np.full((n_per, n_bins), 2.0),
        np.full((n_per, n_bins), 0.1),
    ]) + rng.normal(0, 0.1, (3 * n_per, n_bins))
    mat2 = eh.from_array(vals2, region_index=idx, upstream=1000, downstream=1000, bin_size=50, name="ATAC")
    return mat1, mat2


def test_compute_order_scores_metrics():
    mat1, _ = _make_clustered_matrices()

    cm = eh.compute_order_scores(mat1, metric="center_mean")
    es = eh.compute_order_scores(mat1, metric="enriched_score")
    assert cm.shape == (60,)
    assert es.shape == (60,)

    # Cluster 1 (rows 0..19) has strong central peak -> highest center_mean and enriched_score
    assert cm[:20].mean() > cm[20:40].mean()
    assert cm[:20].mean() > cm[40:].mean()
    assert es[:20].mean() > es[40:].mean()

    # Custom callable metric
    custom = eh.compute_order_scores(mat1, metric=lambda m: m.values[:, 0])
    np.testing.assert_allclose(custom, mat1.values[:, 0])


def test_kmeans_single_and_joint_clustering():
    mat1, mat2 = _make_clustered_matrices()

    cl = eh.cluster_kmeans(mat1, k=3, random_state=42)
    assert isinstance(cl, pd.Series)
    assert list(cl.index) == list(mat1.region_index)
    assert list(cl.cat.categories) == ["C1", "C2", "C3"]
    # Because clusters are cleanly separated, each true block should map almost purely to one label
    assert cl.iloc[:20].nunique() == 1
    assert cl.iloc[20:40].nunique() == 1
    assert cl.iloc[40:].nunique() == 1
    assert cl.iloc[0] == "C1"  # highest mean signal

    # Joint clustering on both tracks (with mat2 permuted to verify region_index alignment!)
    mat2_shuf = mat2.iloc[::-1]
    cl_joint = eh.cluster_kmeans([mat1, mat2_shuf], k=3, random_state=42)
    assert cl_joint.iloc[:20].nunique() == 1
    assert cl_joint.iloc[20:40].nunique() == 1
    assert cl_joint.iloc[40:].nunique() == 1


def test_hierarchical_clustering():
    mat1, mat2 = _make_clustered_matrices()
    ordered_idx, clusters = eh.cluster_hierarchical([mat1, mat2], k=3)
    assert len(ordered_idx) == 60
    assert set(ordered_idx) == set(mat1.region_index)
    assert clusters is not None
    assert clusters.nunique() == 3


def test_resolve_ordering_and_splits_zero_desync():
    mat1, mat2 = _make_clustered_matrices()
    tracks = [eh.EnrichedTrack(mat1), eh.EnrichedTrack(mat2)]

    # Shuffled categorical Series for split_by -> must align by region_index!
    raw_groups = pd.Series(
        (["Low"] * 20) + (["Mid"] * 20) + (["High"] * 20),
        index=mat1.region_index[::-1],
    )
    ordered_idx, split_s, slices = resolve_ordering_and_splits(
        tracks=tracks,
        region_index=mat1.region_index,
        order_by="ChEC",
        order_metric="center_mean",
        split_by=raw_groups,
    )
    assert len(ordered_idx) == 60
    assert split_s is not None
    assert [lbl for lbl, _ in slices] == ["High", "Mid", "Low"]
    # Verify High slice actually contains rows 0..19
    high_ids = set(ordered_idx[slices[0][1]])
    assert high_ids == {f"r_{i}" for i in range(20)}

    # Explicit pd.Index ordering
    explicit = mat1.region_index[::-1]
    ordered_idx2, split_s2, _ = resolve_ordering_and_splits(
        tracks=tracks,
        region_index=mat1.region_index,
        order_by=explicit,
    )
    assert split_s2 is None
    assert list(ordered_idx2) == list(explicit)
