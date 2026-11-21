"""Tests for row ordering metrics, k-means, hierarchical clustering, and split resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import enriched_heatmap as eh


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
    mat1 = eh.NormalizedMatrix(vals, region_index=idx, upstream=1000, downstream=1000, bin_size=50, name="ChEC")

    # Second track correlated with cluster identity
    vals2 = np.vstack([
        np.full((n_per, n_bins), 5.0),
        np.full((n_per, n_bins), 2.0),
        np.full((n_per, n_bins), 0.1),
    ]) + rng.normal(0, 0.1, (3 * n_per, n_bins))
    mat2 = eh.NormalizedMatrix(vals2, region_index=idx, upstream=1000, downstream=1000, bin_size=50, name="ATAC")
    return mat1, mat2


def test_score_metrics():
    mat1, _ = _make_clustered_matrices()

    cm = eh.score(mat1, metric="center_mean")
    es = eh.score(mat1, metric="enriched_score")
    assert cm.shape == (60,)
    assert es.shape == (60,)

    # Cluster 1 (rows 0..19) has strong central peak -> highest center_mean and enriched_score
    assert cm[:20].mean() > cm[20:40].mean()
    assert cm[:20].mean() > cm[40:].mean()
    assert es[:20].mean() > es[40:].mean()

    # Custom callable metric
    custom = eh.score(mat1, metric=lambda m: m.values[:, 0])
    np.testing.assert_allclose(custom.to_numpy(), mat1.values[:, 0])
    assert custom.index.equals(mat1.region_index)


def test_kmeans_single_and_joint_clustering():
    mat1, mat2 = _make_clustered_matrices()

    cl = eh.kmeans(mat1, k=3, random_state=42)
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
    cl_joint = eh.kmeans([mat1, mat2_shuf], k=3, random_state=42)
    assert cl_joint.iloc[:20].nunique() == 1
    assert cl_joint.iloc[20:40].nunique() == 1
    assert cl_joint.iloc[40:].nunique() == 1


def test_hierarchical_clustering():
    mat1, mat2 = _make_clustered_matrices()
    ordered_idx, clusters = eh.hclust([mat1, mat2], k=3)
    assert len(ordered_idx) == 60
    assert set(ordered_idx) == set(mat1.region_index)
    assert clusters is not None
    assert clusters.nunique() == 3


def test_arrange_groups_and_explicit_order():
    mat1, _ = _make_clustered_matrices()
    # Groups given in shuffled order must still be matched by region id.
    raw_groups = pd.Series(
        (["Low"] * 20) + (["Mid"] * 20) + (["High"] * 20),
        index=mat1.region_index[::-1],
    )
    rows = eh.arrange(mat1, groups=raw_groups, gap=0.0)
    assert len(rows) == 60
    # Uncategorised groups come in order of their best-scoring row.
    assert rows.labels[0] == "High"
    assert set(rows.labels) == {"High", "Mid", "Low"}
    spans = rows.spans()
    assert set(rows.index[spans[0][1]:spans[0][2]]) == {f"r_{i}" for i in range(20)}
    # Within a group rows are sorted by score, highest first.
    sc = eh.score(mat1).reindex(rows.index[:20]).to_numpy()
    assert np.all(np.diff(sc) <= 0)

    cat = raw_groups.astype(pd.CategoricalDtype(["Low", "Mid", "High"]))
    assert eh.arrange(mat1, groups=cat).labels == ["Low", "Mid", "High"]

    explicit = mat1.region_index[::-1]
    rows2 = eh.arrange(explicit)
    assert rows2.groups is None
    assert list(rows2.index) == list(explicit)


def test_arrange_gap_and_missing_groups():
    mat1, _ = _make_clustered_matrices()
    groups = pd.Series(["a"] * 30 + ["b"] * 30, index=mat1.region_index)
    rows = eh.arrange(mat1, groups=groups, gap=0.1)
    assert rows.gap == 6.0
    assert rows.height == 66.0
    assert rows.sizes.to_dict() == {"a": 30, "b": 30}
    with pytest.raises(KeyError):
        eh.arrange(mat1, groups=groups.iloc[:50])


# ---------------------------------------------------------------------------
# Regressions: group labels that look like masks or positions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "labels, expected",
    [
        ([True, False, True, False, True], [True, False]),
        ([2, 1, 2, 1, 1], [2, 1]),
        ([0, 1, 0, 1, 1], [0, 1]),
        (["b", "a", "b", "a", "a"], ["b", "a"]),
    ],
)
def test_arrange_accepts_bool_and_integer_groups(labels, expected):
    idx = pd.Index(list("abcde"))
    rows = eh.arrange(index=idx, groups=pd.Series(labels, index=idx))
    assert rows.labels == expected
    assert rows.sizes.sum() == 5
    # members of each group are contiguous and nothing is lost or duplicated
    assert sorted(rows.index) == list("abcde")
    assert rows.groups.loc[rows.index].tolist() == rows.groups.tolist()
    assert [rows.groups.iloc[i0] for _, i0, _, _, _ in rows.spans()] == expected


def test_arrange_bool_groups_with_scores_orders_within_group():
    mat1, _ = _make_clustered_matrices()
    bound = pd.Series(np.arange(mat1.n_regions) % 2 == 0, index=mat1.region_index)
    rows = eh.arrange(mat1, groups=bound)
    scores = eh.score(mat1)
    for _, i0, i1, _, _ in rows.spans():
        block = scores.loc[rows.index[i0:i1]].to_numpy()
        assert np.all(np.diff(block) <= 0)  # descending inside every group


def test_arrange_categorical_groups_drop_unused_categories_in_order():
    idx = pd.Index(list("abcd"))
    g = pd.Series(pd.Categorical(["z", "x", "z", "x"], categories=["z", "y", "x"]), index=idx)
    assert eh.arrange(index=idx, groups=g).labels == ["z", "x"]


def test_hclust_warns_when_the_distance_matrix_would_be_huge(monkeypatch):
    pytest.importorskip("scipy")
    from enriched_heatmap import ordering

    mat1, _ = _make_clustered_matrices()
    monkeypatch.setattr(ordering, "_HCLUST_WARN_ROWS", 10)
    with pytest.warns(ResourceWarning, match="kmeans"):
        eh.hclust(mat1, k=3)
