"""Axes-level drawing: every function has the shape ``fn(data, ax, rows, **style)``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.patches import Patch, Rectangle

from enriched_heatmap._matrix import NormalizedMatrix
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


def palette(
    categories: Sequence[Any], colors: str | mcolors.Colormap | Mapping | Sequence = "tab10"
) -> dict:
    """Map categories to colours.

    ``colors`` is a matplotlib colormap name or object, a list of colours (cycled), or a
    dict; categories missing from a dict fall back to ``tab10``.
    """
    cats = list(categories)
    if isinstance(colors, Mapping):
        fallback = mpl.colormaps["tab10"]
        return {c: colors.get(c, fallback(i % 10)) for i, c in enumerate(cats)}
    if isinstance(colors, (str, mcolors.Colormap)):
        cmap = mpl.colormaps[colors] if isinstance(colors, str) else colors
        if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= 20:
            return {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
        n = max(len(cats) - 1, 1)
        return {c: cmap(i / n) for i, c in enumerate(cats)}
    cols = list(colors)
    if not cols and cats:
        raise ValueError("`colors` must not be empty.")
    return {c: cols[i % len(cols)] for i, c in enumerate(cats)}


def _rows_for(data_index: pd.Index, rows: Rows | None) -> Rows:
    return Rows(index=pd.Index(data_index)) if rows is None else rows


def _target_rows(ax, n_rows: int, max_rows: int | str | None, dpi: float = 300.0) -> int | None:
    if max_rows is None:
        return None
    if max_rows == "auto":
        fig = ax.figure
        if fig.get_layout_engine() is not None:
            # constrained/tight layout only positions axes at draw time; run it first
            fig.draw_without_rendering()
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
        if vmin is None and vmax is None:
            span = float(np.quantile(np.abs(finite - c), q if q is not None else 1.0)) if finite.size else 1.0
            span = span if span > 0 else 1.0
            lo, hi = c - span, c + span
        elif vmin is None:
            hi = float(vmax)  # type: ignore[arg-type]
            span = max(hi - c, 1e-9)
            lo = c - span
        elif vmax is None:
            lo = float(vmin)
            span = max(c - lo, 1e-9)
            hi = c + span
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


def _center_for(mats: Sequence[NormalizedMatrix], center: float | None) -> float | None:
    """Differential matrices (``subtract``, ``log2_ratio``) are centred on 0 unless told otherwise."""
    if center is None and mats and all(m.metadata.get("is_differential") for m in mats):
        return 0.0
    return center


def _set_ylim(ax, rows: Rows) -> None:
    ax.set_ylim(rows.height or 1.0, 0)


def limits(
    matrices: Sequence[NormalizedMatrix] | Mapping[str, NormalizedMatrix],
    q: float = 0.99,
    center: float | None = None,
) -> tuple[float, float]:
    """``(vmin, vmax)`` over several matrices, to put them on one colour scale.

    With ``center`` the range is symmetric around it; otherwise it runs from
    ``min(0, data)`` to the ``q`` quantile of all values. Differential matrices
    (from ``subtract`` or ``log2_ratio``) are centred on 0 when ``center`` is not given.
    """
    if isinstance(matrices, NormalizedMatrix):
        mats = [matrices]
    elif isinstance(matrices, Mapping):
        mats = list(matrices.values())
    else:
        mats = list(matrices)
    if not mats:
        raise ValueError("`matrices` must not be empty.")
    center = _center_for(mats, center)
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
        # Centred matrices put the flank ticks on the outer bin centres, half a bin inside.
        slack = mat.bin_size / 2.0 if getattr(mat, "centered", False) else 0.0
        if pos[0] <= x0 + slack:
            tl[0].set_horizontalalignment("left")
        if pos[-1] >= x1 - slack:
            tl[-1].set_horizontalalignment("right")
    if guides:
        for g in mat.guide_positions():
            ax.axvline(g, **_GUIDE)


def _frame_groups(ax, rows: Rows, x0: float, x1: float, frame: bool) -> None:
    for s in ax.spines.values():
        s.set_visible(False)
    _set_ylim(ax, rows)
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
        Diverging scale around this value (e.g. 0 for a log2 ratio). Differential
        matrices (from ``subtract`` or ``log2_ratio``) default to ``center=0``.
    norm : matplotlib.colors.Normalize, optional
        Overrides ``vmin``/``vmax``/``q``/``center``.
    max_rows : "auto", int or None
        More rows than this are averaged into this many before drawing. ``"auto"``
        is one row per pixel at 300 dpi, which keeps files small without visible loss.
        It measures the axes when called (running a constrained or tight layout first),
        so after a later resize of the figure pass an explicit number instead. ``None``
        draws every row.
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
    center = _center_for([mat], center)
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
    n_boot: int = 200,
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
    ci : float
        Confidence level for the ``"ci"`` and ``"boot"`` bands.
    n_boot : int
        Bootstrap resamples for ``band="boot"``.
    by_group : bool
        One line per group (coloured as in :func:`annotation`, which has the same
        default ``colors``), or a single line over all rows.
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
        res = compute_profile(block, mode=_PROFILE_MODES[key], ci=ci, n_bootstrap=n_boot)
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
    return (
        isinstance(s.dtype, pd.CategoricalDtype)
        or s.dtype == object
        or pd.api.types.is_bool_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
    )


def legends(ax, items: Sequence[tuple[str | None, Mapping]], x: float = 1.05, y: float = 1.0) -> None:
    """Stack ``(title, {label: colour})`` legends down the right-hand side of ``ax``.

    ``x`` and ``y`` are axes coordinates of the first legend's upper-left corner.
    """
    for title, colors in items:
        handles = [Patch(color=c, label=str(k)) for k, c in colors.items()]
        leg = Legend(ax, handles, [str(k) for k in colors], title=title, frameon=False,
                     fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(x, y))
        ax.add_artist(leg)
        leg.set_clip_on(False)  # add_artist clips to the axes, and these sit outside
        y -= 0.028 * (len(handles) + (2.2 if title else 0.6)) + 0.03


_ANNOTATION_KINDS = ("auto", "color", "bar")


def annotation(
    data: pd.Series | pd.DataFrame,
    ax,
    rows: Rows | None = None,
    kind: str = "auto",
    colors: str | Mapping | Sequence = "tab10",
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
        Indexed by region. A DataFrame draws one strip per column. Each column is
        categorical or numeric on its own; numeric columns share one colour scale.
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Same object as for the heatmaps. Regions missing from ``data`` are blank.
    kind : {"auto", "color", "bar"}
        ``"auto"`` and ``"color"`` draw colour strips (categorical or numeric).
        ``"bar"`` draws one numeric column as horizontal bars.
    colors : colormap name, list or dict
        Category colours. The default matches :func:`profile`, so a group strip and the
        profile lines agree.
    cmap, vmin, vmax, q, center
        Colour scale for numeric strips, as in :func:`heatmap`.
    legend : bool
        Legend of categories, one per categorical column, to the right of the axes.
    labels : bool
        Column names under the strip.

    Returns
    -------
    dict, AxesImage or Axes
        For a single column: ``{category: colour}`` for categorical data (use with
        :func:`legend`), the image for numeric strips (use with :func:`colorbar`), the
        axes for bars. For several numeric columns, the image of the last one (all share
        one norm). For several columns of which some are categorical:
        ``{column: <that column's return value>}``.
    """
    if kind not in _ANNOTATION_KINDS:
        raise ValueError(f"`kind` must be one of {_ANNOTATION_KINDS}, got {kind!r}.")
    if isinstance(data, pd.Series):
        df = data.to_frame("" if data.name is None else data.name)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"`data` must be a pandas Series or DataFrame, got {type(data).__name__}.")
    rows = _rows_for(df.index, rows)
    if len(df) and len(rows) and not df.index.isin(rows.index).any():
        raise ValueError(
            "`data` shares no region identifiers with `rows` (data index starts "
            f"{list(df.index[:3])}, rows start {list(rows.index[:3])}). Annotations are "
            "matched by index, not position: pass a Series indexed by region, not "
            "`.values` or a reset index.")
    df = df.reindex(rows.index)
    target = _target_rows(ax, len(rows), max_rows)
    ncol = df.shape[1]
    if ncol == 0:
        raise ValueError("`data` has no columns.")
    first = df.iloc[:, 0]

    def share(i0: int, i1: int) -> int:
        return max(1, int(round(target * (i1 - i0) / max(len(rows), 1))))

    if kind == "bar":
        if ncol != 1:
            raise ValueError("kind='bar' takes a single column.")
        v = first.to_numpy(dtype=np.float64, na_value=np.nan)
        for _, i0, i1, y0, y1 in rows.spans():
            block = v[i0:i1]
            if target is not None:
                block = downsample_1d_rows(block, share(i0, i1), agg=agg)
            h = (y1 - y0) / max(len(block), 1)
            y = y0 + h * np.arange(len(block))
            ax.barh(
                y,
                np.where(np.isfinite(block), block, 0.0),
                height=h,
                align="edge",
                color=bar_color,
                lw=0,
            )
        _set_ylim(ax, rows)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.axvline(0, color="0.3", lw=0.5)
        if labels and str(first.name):
            ax.set_xlabel(str(first.name), fontsize=8)
        return ax

    categorical = [_is_categorical(df.iloc[:, j]) for j in range(ncol)]
    numeric = {
        j: df.iloc[:, j].to_numpy(dtype=np.float64, na_value=np.nan)
        for j in range(ncol) if not categorical[j]
    }
    num_norm = None
    if numeric:
        num_norm = _norm(np.concatenate(list(numeric.values())), vmin, vmax, q, center)
        num_cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad="#eeeeee")

    results: dict[Any, Any] = {}
    legend_items: list[tuple[str | None, Mapping]] = []
    for j, col in enumerate(df.columns):
        series = df.iloc[:, j]
        if categorical[j]:
            cats = (list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype)
                    else sorted(pd.unique(series.dropna()), key=str))
            pal = palette(cats, colors)
            codes = pd.Categorical(series, categories=cats).codes.astype(np.int64)
            lut = mcolors.ListedColormap([pal[c] for c in cats] or ["white"]).with_extremes(bad="white")
            cat_norm = mcolors.BoundaryNorm(np.arange(-0.5, max(len(cats), 1) + 0.5), max(len(cats), 1))
            for _, i0, i1, y0, y1 in rows.spans():
                block = codes[i0:i1]
                if block.shape[0] == 0:
                    continue
                if target is not None:
                    block = downsample_categorical_rows(block, share(i0, i1))
                img = np.where(block < 0, np.nan, block).astype(float)[:, None]
                ax.imshow(img, extent=(j, j + 1, y1, y0), aspect="auto", cmap=lut,
                          norm=cat_norm, interpolation="nearest", rasterized=True)
            results[col] = pal
            legend_items.append((str(col) if ncol > 1 else None, pal))
        else:
            vals = numeric[j][:, None]
            image = None
            for _, i0, i1, y0, y1 in rows.spans():
                block = vals[i0:i1]
                if block.shape[0] == 0:
                    continue
                if target is not None:
                    block = downsample_rows(block, share(i0, i1), agg=agg)
                image = ax.imshow(block, extent=(j, j + 1, y1, y0), aspect="auto", cmap=num_cm,
                                  norm=num_norm, interpolation="nearest", rasterized=True)
            results[col] = image if image is not None else mpl.cm.ScalarMappable(norm=num_norm, cmap=num_cm)

    _frame_groups(ax, rows, 0, ncol, frame)
    ax.set_xlim(0, ncol)
    if labels and any(str(c) for c in df.columns):
        ax.set_xticks(np.arange(ncol) + 0.5, [str(c) for c in df.columns], rotation=90, fontsize=8)
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
    if legend and legend_items:
        legends(ax, legend_items)
    if ncol == 1:
        return results[df.columns[0]]
    if not any(categorical):
        return results[df.columns[-1]]  # every strip shares this image's norm
    return results


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
