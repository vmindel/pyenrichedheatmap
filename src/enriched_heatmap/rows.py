"""Row order and grouping shared by every panel of a figure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enriched_heatmap._matrix import NormalizedMatrix
from enriched_heatmap.ordering import HClust, OrderMetric, score


@dataclass(frozen=True)
class Rows:
    """Which regions are drawn, top to bottom, and how they are split into groups.

    Build it with :func:`arrange` and pass the same object to every drawing function;
    that is what keeps heatmaps, profiles and annotations aligned row for row.

    Attributes
    ----------
    index : pandas.Index
        Region identifiers, top row first.
    groups : pandas.Series or None
        Group label of each row, aligned to ``index``. Groups must be contiguous and
        free of missing labels; construction raises otherwise.
    gap : float
        Blank space between groups, in rows.
    """

    index: pd.Index
    groups: pd.Series | None = None
    gap: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.index, pd.Index):
            object.__setattr__(self, "index", pd.Index(self.index))
        if not self.index.is_unique:
            raise ValueError("`index` must contain unique region identifiers.")
        if self.gap < 0:
            raise ValueError("`gap` must be non-negative.")
        if self.groups is None:
            return
        if len(self.groups) != len(self.index) or not self.groups.index.equals(self.index):
            raise ValueError("`groups` must be a Series aligned to `index`, in the same order.")
        if self.groups.isna().any():
            raise ValueError("`groups` must not contain missing labels.")
        codes, uniques = pd.factorize(self.groups.to_numpy())
        runs = int(np.count_nonzero(np.diff(codes))) + 1 if len(codes) else 0
        if runs != len(uniques):
            raise ValueError(
                "Groups must be contiguous (all rows of a group together); build `Rows` "
                "with `arrange`, which sorts them."
            )

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> list:
        """Group labels in drawing order (``[None]`` when ungrouped)."""
        if self.groups is None:
            return [None]
        return [lbl for lbl, *_ in self.spans()]

    def spans(self) -> list[tuple[object, int, int, float, float]]:
        """``(label, first_row, stop_row, y_top, y_bottom)`` for each group.

        Rows occupy y in ``[0, height)`` with y increasing downwards, as drawn.
        """
        if self.groups is None:
            return [(None, 0, len(self), 0.0, float(len(self)))]
        vals = self.groups.to_numpy()
        if len(vals) == 0:
            return []
        codes, _ = pd.factorize(vals)
        starts = np.concatenate([[0], np.flatnonzero(np.diff(codes)) + 1])
        stops = list(starts[1:]) + [len(vals)]
        out, y = [], 0.0
        for i0, i1 in zip(starts, stops, strict=True):
            n = int(i1 - i0)
            out.append((vals[i0], int(i0), int(i1), y, y + n))
            y += n + self.gap
        return out

    @property
    def height(self) -> float:
        return self.spans()[-1][4] if len(self) else 0.0

    @property
    def sizes(self) -> pd.Series:
        """Number of rows per group."""
        return pd.Series(
            {lbl: stop - start for lbl, start, stop, _, _ in self.spans()}, dtype=np.int64
        )


def _scores_for(by, metric: OrderMetric) -> pd.Series | None:
    if by is None:
        return None
    if isinstance(by, NormalizedMatrix):
        return score(by, metric=metric)
    if isinstance(by, pd.Series):
        return by
    raise TypeError(
        "`by` must be a NormalizedMatrix, a Series of scores, or an Index giving the order; "
        f"got {type(by).__name__}."
    )


def arrange(
    by: NormalizedMatrix | pd.Series | pd.Index | HClust | None = None,
    groups: pd.Series | None = None,
    metric: OrderMetric = "center_mean",
    ascending: bool = False,
    gap: float = 0.02,
    index: pd.Index | None = None,
) -> Rows:
    """Decide the row order (and optional split) for a figure.

    Parameters
    ----------
    by : NormalizedMatrix, Series, Index or HClust, optional
        A matrix is scored with :func:`score` using ``metric``. A Series is taken as
        per-region scores. An Index is taken as the order itself. The result of
        :func:`hclust` gives its leaf order, and its clusters as ``groups`` unless you
        pass your own. ``None`` keeps the input order.
    groups : Series, optional
        Group label per region (e.g. from :func:`kmeans`, or a column of your peak
        table). Groups are drawn in category order for a categorical Series; otherwise
        the group holding the top-scoring row comes first. Rows are sorted by ``by``
        within each group.
    metric : str or callable
        Scoring metric when ``by`` is a matrix.
    ascending : bool
        Sort direction; the default puts the highest score on top. NaN scores go last.
    gap : float
        Space between groups as a fraction of the number of rows.
    index : Index, optional
        The regions to draw. Default: all regions in ``by`` (or ``groups``).

    Returns
    -------
    Rows
    """
    if isinstance(by, HClust):
        if groups is None:
            groups = by.clusters
        by = by.order
    if index is None:
        if isinstance(by, NormalizedMatrix):
            index = by.region_index
        elif isinstance(by, (pd.Series, pd.Index)):
            index = by.index if isinstance(by, pd.Series) else by
        elif groups is not None:
            index = groups.index
        else:
            raise ValueError("Pass `by`, `groups` or `index` so the regions are known.")
    index = pd.Index(index)
    if not index.is_unique:
        raise ValueError("`index` must contain unique region identifiers.")

    if isinstance(by, pd.Index):
        if not by.is_unique:
            raise ValueError("`by` order Index must contain unique region identifiers.")
        missing = index.difference(by)
        if len(missing):
            raise KeyError(f"{len(missing)} regions are not in the `by` order, e.g. {list(missing[:3])}.")
        ordered = by[by.isin(index)]
    else:
        scores = _scores_for(by, metric)
        if scores is None:
            ordered = index
        else:
            vals = scores.reindex(index).to_numpy(dtype=np.float64)
            key = np.where(np.isfinite(vals), vals if ascending else -vals, np.inf)
            ordered = index[np.argsort(key, kind="stable")]

    if groups is None:
        return Rows(index=ordered, groups=None, gap=0.0)

    g = groups.reindex(ordered)
    if g.isna().any():
        missing = ordered[g.isna().to_numpy()]
        raise KeyError(f"{len(missing)} regions have no group, e.g. {list(missing[:3])}.")
    if isinstance(g.dtype, pd.CategoricalDtype):
        present = set(g.unique())
        levels = [c for c in g.cat.categories if c in present]
    else:
        levels = list(pd.unique(g.to_numpy()))
    # Rank through a plain dict: a bool or integer-labelled Series would make
    # ``Series.loc[array]`` read the labels as a mask or as positions.
    rank = {lbl: i for i, lbl in enumerate(levels)}
    codes = np.fromiter((rank[v] for v in g.to_numpy()), dtype=np.int64, count=len(g))
    pos = np.argsort(codes, kind="stable")
    ordered = ordered[pos]
    g = g.iloc[pos]
    g.index = ordered
    return Rows(index=ordered, groups=g, gap=float(gap) * len(ordered))
