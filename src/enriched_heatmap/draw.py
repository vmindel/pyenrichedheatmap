"""Axes-level drawing: every function has the shape ``fn(data, ax, rows, **style)``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from enriched_heatmap.matrix import NormalizedMatrix
from enriched_heatmap.profiles import compute_profile
from enriched_heatmap.raster import (
    downsample_1d_rows,
    downsample_categorical_rows,
    downsample_rows,
)
from enriched_heatmap.rows import Rows

_FRAME = {"color": "#333333", "lw": 0.6}
_GUIDE = {"color": "0.3", "lw": 0.6, "ls": "--", "alpha": 0.6}
_PROFILE_MODES = {
    ("mean", None): "mean",
    ("median", None): "median",
    ("mean", "sem"): "mean_sem",
    ("mean", "ci"): "mean_ci",
    ("mean", "boot"): "mean_boot_ci",
    ("median", "iqr"): "median_iqr",
}


def palette(categories: Sequence[Any], colors: str | Mapping | Sequence = "tab10") -> dict:
    """Map categories to colours.

    ``colors`` is a matplotlib colormap name, a list of colours (cycled), or a dict;
    categories missing from a dict fall back to ``tab10``.
    """
    cats = list(categories)
    if isinstance(colors, Mapping):
        fallback = mpl.colormaps["tab10"]
        return {c: colors.get(c, fallback(i % 10)) for i, c in enumerate(cats)}
    if isinstance(colors, str):
        cmap = mpl.colormaps[colors]
        if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= 20:
            return {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
        n = max(len(cats) - 1, 1)
        return {c: cmap(i / n) for i, c in enumerate(cats)}
    cols = list(colors)
    return {c: cols[i % len(cols)] for i, c in enumerate(cats)}


def _rows_for(data_index: pd.Index, rows: Rows | None) -> Rows:
    return Rows(index=pd.Index(data_index)) if rows is None else rows


def _target_rows(ax, n_rows: int, max_rows: int | str | None, dpi: float = 300.0) -> int | None:
    if max_rows is None:
        return None
    if max_rows == "auto":
        bbox = ax.get_window_extent()
        return max(16, int(round(bbox.height / ax.figure.dpi * dpi)))
    return int(max_rows)


def _norm(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    q: float | None,
    center: float | None,
) -> mcolors.Normalize:
    finite = values[np.isfinite(values)]
    if center is not None:
        c = float(center)
        if vmin is None or vmax is None:
            span = float(np.quantile(np.abs(finite - c), q if q is not None else 1.0)) if finite.size else 1.0
            span = span if span > 0 else 1.0
            lo = c - span if vmin is None else float(vmin)
            hi = c + span if vmax is None else float(vmax)
        else:
            lo, hi = float(vmin), float(vmax)
        return mcolors.TwoSlopeNorm(vmin=min(lo, c - 1e-9), vcenter=c, vmax=max(hi, c + 1e-9))
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(min(0.0, finite.min())) if vmin is None else float(vmin)
        hi = float(np.quantile(finite, q) if q is not None else finite.max()) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1.0
    return mcolors.Normalize(vmin=lo, vmax=hi)


def limits(
    matrices: Sequence[NormalizedMatrix] | Mapping[str, NormalizedMatrix],
    q: float = 0.99,
    center: float | None = None,
) -> tuple[float, float]:
    """``(vmin, vmax)`` over several matrices, to put them on one colour scale.

    With ``center`` the range is symmetric around it; otherwise it runs from
    ``min(0, data)`` to the ``q`` quantile of all values.
    """
    mats = list(matrices.values()) if isinstance(matrices, Mapping) else list(matrices)
    vals = np.concatenate([m.values.ravel() for m in mats])
    n = _norm(vals, None, None, q, center)
    return float(n.vmin), float(n.vmax)


def _xaxis(mat: NormalizedMatrix, ax, labels, guides: bool, ticks: bool) -> None:
    x0, x1 = mat.x_extent
    ax.set_xlim(x0, x1)
    pos, lab = mat.default_xticks(anchor_labels=labels)
    ax.set_xticks(pos, lab)
    # Hide rather than clear: a profile shares its x-axis with the heatmap below.
    ax.tick_params(axis="x", labelsize=7, length=2 if ticks else 0, labelbottom=ticks)
    # Edge labels point inwards so neighbouring columns do not collide.
    tl = ax.get_xticklabels()
    if len(tl) > 1:
        if pos[0] <= x0:
            tl[0].set_horizontalalignment("left")
        if pos[-1] >= x1:
            tl[-1].set_horizontalalignment("right")
    if guides:
        for g in mat.guide_positions():
            ax.axvline(g, **_GUIDE)


def _frame_groups(ax, rows: Rows, x0: float, x1: float, frame: bool) -> None:
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylim(rows.height, 0)
    ax.set_yticks([])
    if frame:
        for _, _, _, y0, y1 in rows.spans():
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, zorder=3,
                                   edgecolor=_FRAME["color"], lw=_FRAME["lw"], clip_on=False))


def heatmap(
    mat: NormalizedMatrix,
    ax,
    rows: Rows | None = None,
    cmap: str | mcolors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    q: float | None = 0.99,
    center: float | None = None,
    norm: mcolors.Normalize | None = None,
    na_color: str = "#eeeeee",
    max_rows: int | str | None = "auto",
    agg: str = "mean",
    anchor_labels: str | tuple[str, str] | None = None,
    guides: bool = True,
    xticks: bool = True,
    frame: bool = True,
    title: str | None = None,
):
    """Draw a region x bin matrix as a heatmap.

    Parameters
    ----------
    mat : NormalizedMatrix
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Order and groups from :func:`arrange`. Regions in ``rows`` but not in ``mat``
        are drawn as NaN. Default: the matrix order, one group.
    cmap : colormap, optional
        Default ``"Reds"``, or ``"RdBu_r"`` when ``center`` is set.
    vmin, vmax : float, optional
        Colour limits. Pass the same values (see :func:`limits`) to several heatmaps to
        put them on one scale.
    q : float or None
        Quantile used for ``vmax`` when it is not given, so a few hot bins do not wash
        out the rest. ``None`` uses the maximum.
    center : float, optional
        Diverging scale around this value (e.g. 0 for a log2 ratio).
    norm : matplotlib.colors.Normalize, optional
        Overrides ``vmin``/``vmax``/``q``/``center``.
    max_rows : "auto", int or None
        More rows than this are averaged into this many before drawing. ``"auto"``
        is one row per pixel at 300 dpi, which keeps files small without visible loss.
        ``None`` draws every row.
    agg : str
        How rows are combined when downsampling: ``"mean"``, ``"max"``, ``"absmax"``,
        ``"median"``, ``"q90"``. Use ``"max"`` for sparse, narrow peaks.
    anchor_labels : str or (str, str), optional
        Tick label for the anchor (default ``"0"``), or the body start/end labels in
        scale-regions mode (default ``("Start", "End")``).
    guides : bool
        Dashed lines at the anchor (or body edges).
    xticks : bool
        Label the x-axis in bp/kb.
    frame : bool
        Box around each group.
    title : str, optional

    Returns
    -------
    matplotlib.image.AxesImage
        For ``fig.colorbar`` or :func:`colorbar`. Every group shares its norm.
    """
    rows = _rows_for(mat.region_index, rows)
    vals = mat.reindex(rows.index).values if not mat.region_index.equals(rows.index) else mat.values
    if norm is None:
        norm = _norm(vals, vmin, vmax, q, center)
    if cmap is None:
        cmap = "RdBu_r" if center is not None else "Reds"
    cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad=na_color)

    x0, x1 = mat.x_extent
    target = _target_rows(ax, len(rows), max_rows)
    image = None
    for _, i0, i1, y0, y1 in rows.spans():
        block = vals[i0:i1]
        if block.shape[0] == 0:
            continue
        if target is not None:
            share = max(1, int(round(target * (i1 - i0) / max(len(rows), 1))))
            block = downsample_rows(block, share, agg=agg)
        image = ax.imshow(block, extent=(x0, x1, y1, y0), aspect="auto", cmap=cm, norm=norm,
                          interpolation="nearest", rasterized=True)
    _frame_groups(ax, rows, x0, x1, frame)
    _xaxis(mat, ax, anchor_labels, guides, xticks)
    if title is not None:
        ax.set_title(title, fontsize=9)
    if image is None:
        image = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    return image


def profile(
    mat: NormalizedMatrix,
    ax,
    rows: Rows | None = None,
    stat: str = "mean",
    band: str | None = None,
    by_group: bool = True,
    colors: str | Mapping | Sequence = "tab10",
    color: str = "0.15",
    ci: float = 0.95,
    lw: float = 1.2,
    alpha: float = 0.25,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    guides: bool = True,
    xticks: bool = False,
    title: str | None = None,
):
    """Draw the average profile over regions (the metaplot above a heatmap).

    Parameters
    ----------
    mat : NormalizedMatrix
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Restricts to these regions; with groups and ``by_group`` draws one line each.
    stat : {"mean", "median"}
    band : {None, "sem", "ci", "boot", "iqr"}
        Shaded band: standard error, normal ``ci`` interval, bootstrap ``ci`` interval
        (with mean), or interquartile range (with median).
    by_group : bool
        One line per group (coloured as in :func:`annotation` with the same
        ``colors``), or a single line over all rows.
    colors : colormap name, list or dict
        Group colours.
    color : colour
        Line colour when there is a single line.
    ylim : (float, float), optional
        Pass the same limits to several profiles to compare them.
    legend : bool
        Add a legend with group names.

    Returns
    -------
    matplotlib.axes.Axes
    """
    key = (stat, band)
    if key not in _PROFILE_MODES:
        raise ValueError(f"Unsupported stat/band combination {key}; choose from {list(_PROFILE_MODES)}.")
    rows = _rows_for(mat.region_index, rows)
    vals = mat.reindex(rows.index).values if not mat.region_index.equals(rows.index) else mat.values
    x = mat.bin_centers()

    if by_group and rows.groups is not None:
        cols = palette(rows.labels, colors)
        parts = [(lbl, vals[i0:i1], cols[lbl]) for lbl, i0, i1, _, _ in rows.spans()]
    else:
        parts = [(None, vals, color)]
    for lbl, block, c in parts:
        res = compute_profile(block, mode=_PROFILE_MODES[key], ci=ci)
        ax.plot(x, res.center, color=c, lw=lw, label=None if lbl is None else str(lbl))
        if res.lower is not None:
            ax.fill_between(x, res.lower, res.upper, color=c, alpha=alpha, lw=0)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=7, length=2)
    _xaxis(mat, ax, None, guides, xticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend and len(parts) > 1:
        ax.legend(frameon=False, fontsize=7)
    if title is not None:
        ax.set_title(title, fontsize=9)
    return ax


def _is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool \
        or pd.api.types.is_string_dtype(s.dtype)


def annotation(
    data: pd.Series | pd.DataFrame,
    ax,
    rows: Rows | None = None,
    kind: str = "auto",
    colors: str | Mapping | Sequence = "Set2",
    cmap: str | mcolors.Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    q: float | None = 0.99,
    center: float | None = None,
    bar_color: str = "0.35",
    max_rows: int | str | None = "auto",
    agg: str = "mean",
    legend: bool = False,
    frame: bool = True,
    labels: bool = True,
):
    """Draw per-region values beside a heatmap: a colour strip or bars.

    Parameters
    ----------
    data : Series or DataFrame
        Indexed by region. A DataFrame draws one strip per column (numeric columns
        share one colour scale).
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Same object as for the heatmaps. Regions missing from ``data`` are blank.
    kind : {"auto", "color", "bar"}
        ``"auto"``: categorical data as a colour strip, numeric data as a colour strip
        too. ``"bar"`` draws numeric values as horizontal bars.
    colors : colormap name, list or dict
        Category colours.
    cmap, vmin, vmax, q, center
        Colour scale for numeric strips, as in :func:`heatmap`.
    legend : bool
        Legend of categories (categorical data).
    labels : bool
        Column names under the strip.

    Returns
    -------
    dict, AxesImage or Axes
        ``{category: colour}`` for categorical data (use with :func:`legend`), the image
        for numeric strips (use with :func:`colorbar`), the axes for bars.
    """
    df = data.to_frame() if isinstance(data, pd.Series) else data
    rows = _rows_for(df.index, rows)
    df = df.reindex(rows.index)
    target = _target_rows(ax, len(rows), max_rows)
    ncol = df.shape[1]
    first = df.iloc[:, 0]

    if kind == "bar":
        if ncol != 1:
            raise ValueError("kind='bar' takes a single column.")
        v = first.to_numpy(dtype=np.float64)
        for _, i0, i1, y0, y1 in rows.spans():
            block = v[i0:i1]
            if target is not None:
                block = downsample_1d_rows(block, max(1, int(round(target * (i1 - i0) / len(rows)))), agg=agg)
            h = (y1 - y0) / max(len(block), 1)
            y = y0 + h * np.arange(len(block))
            ax.barh(y, np.nan_to_num(block), height=h, align="edge", color=bar_color, lw=0)
        ax.set_ylim(rows.height, 0)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.axvline(0, color="0.3", lw=0.5)
        if labels:
            ax.set_xlabel(str(first.name), fontsize=8)
        return ax

    if _is_categorical(first):
        cats = (list(first.cat.categories) if isinstance(first.dtype, pd.CategoricalDtype)
                else sorted(pd.unique(first.dropna()), key=str))
        cmap_ = palette(cats, colors)
        codes = pd.Categorical(first, categories=cats).codes.astype(np.int64)
        lut = mcolors.ListedColormap([cmap_[c] for c in cats] or ["white"])
        result: Any = cmap_
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(cats) + 0.5), max(len(cats), 1))
        for _, i0, i1, y0, y1 in rows.spans():
            block = codes[i0:i1]
            if target is not None:
                block = downsample_categorical_rows(block, max(1, int(round(target * (i1 - i0) / len(rows)))))
            img = np.where(block < 0, np.nan, block).astype(float)[:, None]
            ax.imshow(img, extent=(0, 1, y1, y0), aspect="auto", cmap=lut.with_extremes(bad="white"),
                      norm=norm, interpolation="nearest", rasterized=True)
        x1 = 1
        if legend:
            ax.legend(handles=[Patch(color=cmap_[c], label=str(c)) for c in cats], frameon=False,
                      fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1))
    else:
        vals = df.to_numpy(dtype=np.float64)
        norm = _norm(vals, vmin, vmax, q, center)
        cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad="#eeeeee")
        for _, i0, i1, y0, y1 in rows.spans():
            block = vals[i0:i1]
            if target is not None:
                block = downsample_rows(block, max(1, int(round(target * (i1 - i0) / len(rows)))), agg=agg)
            result = ax.imshow(block, extent=(0, ncol, y1, y0), aspect="auto", cmap=cm, norm=norm,
                               interpolation="nearest", rasterized=True)
        x1 = ncol

    _frame_groups(ax, rows, 0, x1, frame)
    ax.set_xlim(0, x1)
    if labels:
        ax.set_xticks(np.arange(ncol) + 0.5, [str(c) for c in df.columns], rotation=90, fontsize=8)
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
    return result


def group_labels(ax, rows: Rows, counts: bool = True, side: str = "left", fontsize: float = 8) -> None:
    """Write each group's label (and size) beside its block of rows."""
    if rows.groups is None:
        return
    x, ha = (-0.15, "right") if side == "left" else (1.15, "left")
    for lbl, i0, i1, y0, y1 in rows.spans():
        text = f"{lbl}\nn={i1 - i0:,}" if counts else str(lbl)
        ax.text(x, (y0 + y1) / 2, text, transform=ax.get_yaxis_transform(), ha=ha, va="center",
                fontsize=fontsize)


def legend(ax, colors: Mapping, title: str | None = None, loc: str = "upper left", **kwargs) -> None:
    """Legend from a ``{label: colour}`` dict, such as :func:`annotation` returns."""
    handles = [Patch(color=c, label=str(k)) for k, c in colors.items()]
    ax.legend(handles=handles, title=title, frameon=False, fontsize=7, title_fontsize=8, loc=loc, **kwargs)


def colorbar(mappable, cax, label: str | None = None, orientation: str = "horizontal"):
    """A thin colour bar with small ticks; returns the matplotlib Colorbar."""
    cb = cax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    if label:
        cb.set_label(label, fontsize=8)
    return cb
