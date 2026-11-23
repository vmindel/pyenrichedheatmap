"""Figure grids, and a one-call figure built from the drawing functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
from matplotlib.figure import Figure

from enriched_heatmap import draw
from enriched_heatmap._matrix import NormalizedMatrix
from enriched_heatmap.rows import Rows, arrange


@dataclass
class Grid:
    """Axes of a figure made by :func:`figure`, one entry per column.

    ``profile[i]`` and ``colorbar[i]`` are ``None`` for columns created without them.
    """

    fig: Figure
    heatmap: list = field(default_factory=list)
    profile: list = field(default_factory=list)
    colorbar: list = field(default_factory=list)


def _per_column(value, n: int, name: str) -> list:
    if isinstance(value, bool) or value is None:
        return [bool(value)] * n
    out = list(value)
    if len(out) != n:
        raise ValueError(f"`{name}` has {len(out)} entries for {n} columns.")
    return [bool(v) for v in out]


def figure(
    widths: Sequence[float],
    profile: bool | Sequence[bool] = True,
    colorbar: bool | Sequence[bool] = True,
    height: float = 5.0,
    profile_height: float = 1.0,
    colorbar_height: float = 0.1,
    gap: float = 0.25,
    margins: tuple[float, float, float, float] = (0.7, 0.3, 0.35, 0.55),
    pyplot: bool = True,
) -> tuple[Figure, Grid]:
    """Create aligned columns: profile on top, heatmap body, colour bar below.

    All sizes are in inches. Heatmap axes share their y-axis, and each profile shares
    its x-axis with the heatmap beneath it.

    Parameters
    ----------
    widths : list of float
        Width of each column. Use narrow columns (0.15-0.3) for annotation strips.
    profile, colorbar : bool or list of bool
        Whether each column gets a profile axes / a colour-bar axes.
    height : float
        Height of the heatmap body.
    profile_height, colorbar_height : float
    gap : float
        Horizontal space between columns.
    margins : (left, right, top, bottom)
        Space around the grid, for group labels, titles and tick labels.
    pyplot : bool
        ``True`` creates the figure with ``plt.figure``, so ``plt.show()`` and notebook
        display work; it also stays registered until ``plt.close(fig)``, which matters
        when making many figures in a loop. ``False`` returns a plain
        ``matplotlib.figure.Figure`` that is freed when you drop it; save it with
        ``fig.savefig``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    grid : Grid
    """
    n = len(widths)
    if n == 0:
        raise ValueError("`widths` must not be empty.")
    prof = _per_column(profile, n, "profile")
    cbar = _per_column(colorbar, n, "colorbar")
    left, right, top, bottom = margins
    ph = profile_height if any(prof) else 0.0
    ch = colorbar_height if any(cbar) else 0.0
    vgap = 0.12  # profile to heatmap
    tick_room = 0.4  # heatmap x tick labels above the colour bar
    W = left + right + sum(widths) + gap * (n - 1)
    H = top + bottom + ph + (vgap if ph else 0) + height + (ch + tick_room if ch else 0)
    if pyplot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(W, H))
    else:
        fig = Figure(figsize=(W, H))
    grid = Grid(fig=fig)

    x = left
    first = None
    for j, w in enumerate(widths):
        y = bottom + (ch + tick_room if ch else 0)
        hm = fig.add_axes((x / W, y / H, w / W, height / H), sharey=first)
        first = first or hm
        grid.heatmap.append(hm)
        if prof[j]:
            yp = y + height + vgap
            grid.profile.append(fig.add_axes((x / W, yp / H, w / W, ph / H), sharex=hm))
        else:
            grid.profile.append(None)
        if cbar[j]:
            grid.colorbar.append(fig.add_axes((x / W, bottom / H, w / W, ch / H)))
        else:
            grid.colorbar.append(None)
        x += w + gap
    return fig, grid


def _as_dict(mats) -> dict[str, NormalizedMatrix]:
    if isinstance(mats, NormalizedMatrix):
        return {mats.name or "signal": mats}
    if isinstance(mats, Mapping):
        return dict(mats)
    return {m.name or f"m{i}": m for i, m in enumerate(mats)}


def _pick(value, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return default if value is None else value


def _as_annotations(annotations) -> dict[str, pd.Series]:
    """Normalise ``plot(annotations=...)`` to a dict of name -> Series."""
    if annotations is None:
        return {}
    if isinstance(annotations, pd.DataFrame):
        return {str(c): annotations[c] for c in annotations.columns}
    if isinstance(annotations, pd.Series):
        return {str(annotations.name) if annotations.name is not None else "": annotations}
    if not isinstance(annotations, Mapping):
        raise TypeError("`annotations` must be a dict of name -> Series, a Series or a "
                        f"DataFrame indexed by region, got {type(annotations).__name__}.")
    return dict(annotations)


def plot(
    mats: NormalizedMatrix | Sequence[NormalizedMatrix] | Mapping[str, NormalizedMatrix],
    rows: Rows | None = None,
    annotations: Mapping[str, pd.Series] | pd.DataFrame | pd.Series | None = None,
    cmap: str | Mapping[str, str] | None = None,
    center: float | Mapping[str, float] | None = None,
    vmax: float | Mapping[str, float] | None = None,
    shared: Sequence[Sequence[str]] = (),
    q: float = 0.99,
    profile: bool = True,
    band: str | None = None,
    share_profile_y: bool = False,
    colors: str | Mapping | Sequence = "tab10",
    width: float = 1.6,
    annotation_width: float = 0.22,
    height: float = 5.0,
    group_strip: bool = True,
    agg: str = "mean",
    annotation_colors: str | Mapping | Sequence = "Set2",
    annotation_legend: bool = True,
    guides: bool = True,
    pyplot: bool = True,
) -> tuple[Figure, Grid]:
    """The standard figure in one call: one heatmap column per matrix.

    Everything here is :func:`figure` plus :func:`heatmap`, :func:`profile`,
    :func:`annotation` and :func:`colorbar`; drop to those when you need a layout this
    does not cover.

    Parameters
    ----------
    mats : matrix, list or dict of name to matrix
        Columns, left to right. Dict keys are the column titles.
    rows : Rows, optional
        From :func:`arrange`. Default: sorted by the first matrix, no groups.
    annotations : dict of name to Series, DataFrame or Series, optional
        Narrow strips to the right of the heatmaps, one per entry (or per DataFrame
        column), indexed by region identifier like the matrices.
    cmap, center, vmax : value or dict keyed by column name
        Passed to :func:`heatmap`.
    shared : list of lists of names
        Columns that share one colour scale, e.g. ``[["WT", "KO"]]``.
    q : float
        Quantile for ``vmax`` when it is not given.
    profile : bool
        Draw profiles above the heatmaps.
    band : str, optional
        Profile band, see :func:`profile`.
    share_profile_y : bool
        Same y-range for every profile.
    colors : colormap name, list or dict
        Group colours, used by the group strip and the profile lines.
    group_strip : bool
        With groups, add a labelled colour strip on the left.
    annotation_colors : colormap name, list or dict
        Category colours of the annotation strips. Differs from ``colors`` by default
        so a strip is not mistaken for the groups.
    annotation_legend : bool
        Legend in the right margin for every categorical annotation strip.
    guides : bool
        Dashed reference lines at the anchor (and body end) in heatmaps and profiles.
    pyplot : bool
        See :func:`figure`; pass ``False`` to build many figures without leaking them.

    Returns
    -------
    fig, grid
    """
    mats = _as_dict(mats)
    names = list(mats)
    if not names:
        raise ValueError("`mats` must not be empty.")
    if rows is None:
        rows = arrange(mats[names[0]])
    annotations = _as_annotations(annotations)
    grouped = rows.groups is not None and group_strip

    widths = ([annotation_width] if grouped else []) + [width] * len(names) + \
        [annotation_width] * len(annotations)
    is_hm = ([False] if grouped else []) + [True] * len(names) + [False] * len(annotations)
    categorical = [
        n for n, sr in annotations.items() if draw._is_categorical(sr)
    ] if annotation_legend else []
    margins = (0.7, 1.3 if categorical else 0.3, 0.35, 0.55)
    fig, grid = figure(widths, profile=[profile and h for h in is_hm], colorbar=is_hm,
                       height=height, margins=margins, pyplot=pyplot)
    off = 1 if grouped else 0

    limits = {}
    if isinstance(shared, str):
        raise TypeError(
            "`shared` must be a list of groups of column names, e.g. shared=[['WT', 'KO']]."
        )
    for group in shared:
        if isinstance(group, str) or not isinstance(group, Sequence):
            raise TypeError(
                "`shared` must be a list of groups of column names, e.g. shared=[['WT', 'KO']]."
            )
        if not group:
            continue
        lim = draw.limits([mats[n] for n in group], q=q, center=_pick(center, group[0]))
        limits.update({n: lim for n in group})

    for j, name in enumerate(names):
        ax = grid.heatmap[off + j]
        lo, hi = limits.get(name, (None, _pick(vmax, name)))
        im = draw.heatmap(mats[name], ax, rows, cmap=_pick(cmap, name), center=_pick(center, name),
                          vmin=lo, vmax=hi, q=q, agg=agg, guides=guides,
                          title=None if profile else name)
        draw.colorbar(im, grid.colorbar[off + j])
        if profile:
            draw.profile(mats[name], grid.profile[off + j], rows, band=band, colors=colors,
                         guides=guides, title=name)

    if profile and share_profile_y:
        axes = [a for a in grid.profile if a is not None]
        lo = min(a.get_ylim()[0] for a in axes)
        hi = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(lo, hi)

    if grouped:
        draw.annotation(rows.groups.rename(""), grid.heatmap[0], rows, colors=colors, labels=False)
        draw.group_labels(grid.heatmap[0], rows)

    items = []
    for k, (name, series) in enumerate(annotations.items()):
        res = draw.annotation(series.rename(name), grid.heatmap[off + len(names) + k], rows,
                              colors=annotation_colors)
        if isinstance(res, dict):
            items.append((name, res))
    if items and annotation_legend:
        draw.legends(grid.heatmap[-1], items, x=1.15)
    return fig, grid
