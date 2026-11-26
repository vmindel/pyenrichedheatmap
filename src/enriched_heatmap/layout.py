"""Figure grids, and a one-call figure built from the drawing functions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from enriched_heatmap import draw
from enriched_heatmap._matrix import NormalizedMatrix
from enriched_heatmap.raster import DownsampleAgg
from enriched_heatmap.rows import Rows, arrange


@dataclass
class Grid:
    """Axes of a figure made by :func:`figure`, one entry per column.

    ``profile[i]`` and ``colorbar[i]`` are ``None`` for columns created without them.
    """

    fig: Figure
    heatmap: list[Axes] = field(default_factory=list)
    profile: list[Axes | None] = field(default_factory=list)
    colorbar: list[Axes | None] = field(default_factory=list)


def _per_column(value: bool | Sequence[bool] | None, n: int, name: str) -> list[bool]:
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
    first: Axes | None = None
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


def _as_dict(
    mats: NormalizedMatrix | Sequence[NormalizedMatrix] | Mapping[str, NormalizedMatrix],
) -> dict[str, NormalizedMatrix]:
    if isinstance(mats, NormalizedMatrix):
        return {mats.name or "signal": mats}
    if isinstance(mats, Mapping):
        return dict(mats)
    out: dict[str, NormalizedMatrix] = {}
    for i, m in enumerate(mats):
        name = m.name or f"m{i}"
        if name in out:
            raise ValueError(
                f"Duplicate matrix name {name!r} in `mats`; pass a dict with distinct column names."
            )
        out[name] = m
    return out


def _validate_dict_keys(value: Any, allowed: Sequence[str], param_name: str) -> None:
    if isinstance(value, Mapping):
        unknown = [k for k in value if k not in allowed]
        if unknown:
            raise KeyError(
                f"`{param_name}` has unknown keys {unknown} that are not in {list(allowed)}."
            )


def _pick(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return default if value is None else value


def _as_annotations(
    annotations: Mapping[str, pd.Series] | pd.DataFrame | pd.Series | None,
) -> dict[str, pd.Series]:
    """Normalise ``plot(annotations=...)`` to a dict of name -> Series."""
    if annotations is None:
        return {}
    if isinstance(annotations, pd.DataFrame):
        return {str(c): annotations[c] for c in annotations.columns}
    if isinstance(annotations, pd.Series):
        return {str(annotations.name) if annotations.name is not None else "": annotations}
    if not isinstance(annotations, Mapping):
        raise TypeError(
            "`annotations` must be a dict of name -> Series, a Series or a "
            f"DataFrame indexed by region, got {type(annotations).__name__}."
        )
    return dict(annotations)


def _column_factors(
    widths: Sequence[float] | Mapping[str, float] | None, names: list[str]
) -> list[float]:
    """``plot(widths=)``: relative width factor per heatmap column, default 1."""
    if widths is None:
        return [1.0] * len(names)
    if isinstance(widths, Mapping):
        unknown = [k for k in widths if k not in names]
        if unknown:
            raise KeyError(f"`widths` has keys {unknown} that are not columns {names}.")
        out = [float(widths.get(n, 1.0)) for n in names]
    else:
        if isinstance(widths, (str, bytes)) or not hasattr(widths, "__len__"):
            raise TypeError(
                "`widths` must be a list of factors, one per matrix, or a dict "
                "keyed by column name."
            )
        out = [float(w) for w in widths]
        if len(out) != len(names):
            raise ValueError(
                f"`widths` has {len(out)} entries for {len(names)} matrices ({names})."
            )
    if any(not (0 < w < math.inf) for w in out):
        raise ValueError(f"`widths` factors must be positive and finite, got {out}.")
    return out


def _is_strict_palette_spec(v: Any) -> bool:
    """True when ``v`` cannot be a single colour (it is a dict, Colormap, list, or non-colour string)."""
    if isinstance(v, (Mapping, mcolors.Colormap)):
        return True
    return not mcolors.is_color_like(v)


def _is_cmap_name(v: Any) -> bool:
    return isinstance(v, str) and v in mpl.colormaps


def _resolve_annotation_palettes(
    colors: Any,
    annotations: dict[str, pd.Series],
    rows: Rows,
) -> dict[str, Any]:
    """Resolve per-strip or shared categorical palettes, validating keys and offsetting shared qualitative palettes."""
    ann_names = list(annotations)
    cat_by_strip: dict[str, list[Any]] = {
        n: draw._categories_for_series(sr.reindex(rows.index))
        for n, sr in annotations.items()
        if draw._is_categorical(sr)
    }
    all_cat_values: set[Any] = {c for cats in cat_by_strip.values() for c in cats}

    if isinstance(colors, Mapping):
        if not colors:
            return {n: colors for n in ann_names}
        is_per_strip = any(_is_strict_palette_spec(v) for v in colors.values()) or (
            all(_is_cmap_name(v) for v in colors.values())
            and (set(colors) <= set(ann_names) or not (set(colors) & all_cat_values))
        )

        if is_per_strip:
            unknown = [k for k in colors if k not in ann_names]
            if unknown:
                raise KeyError(
                    f"`annotation_colors` has unknown strip keys {unknown}; "
                    f"expected subset of {ann_names}."
                )
            for v in colors.values():
                if isinstance(v, str):
                    _ = mpl.colormaps[v]
            return {n: colors.get(n, "Set2") for n in ann_names}
        return {n: colors for n in ann_names}

    # Shared non-mapping palette: offset qualitative ListedColormap (N <= 20) or colour sequences across strips
    if isinstance(colors, (str, mcolors.Colormap)):
        cm = mpl.colormaps[colors] if isinstance(colors, str) else colors
        if isinstance(cm, mcolors.ListedColormap) and cm.N <= 20:
            out_cm: dict[str, Any] = {}
            offset = 0
            for n in ann_names:
                if n in cat_by_strip:
                    cats = cat_by_strip[n]
                    k_len = len(cats)
                    out_cm[n] = [cm((offset + i) % cm.N) for i in range(max(k_len, 1))]
                    offset += k_len
                else:
                    out_cm[n] = colors
            return out_cm
        return {n: cm for n in ann_names}

    seq_list = list(colors)
    out_seq: dict[str, Any] = {}
    offset = 0
    for n in ann_names:
        if n in cat_by_strip:
            cats = cat_by_strip[n]
            if not seq_list and cats:
                raise ValueError("`annotation_colors` must not be empty.")
            k_len = len(cats)
            out_seq[n] = (
                [seq_list[(offset + i) % len(seq_list)] for i in range(k_len)]
                if (seq_list and k_len > 0)
                else seq_list
            )
            offset += k_len
        else:
            out_seq[n] = seq_list
    return out_seq


def _resolve_shared_group_bound(
    param: Any, group: Sequence[str], param_name: str
) -> float | None:
    if param is None:
        return None
    if isinstance(param, Mapping):
        specified = [param[n] for n in group if n in param and param[n] is not None]
        if not specified:
            return None
        first_val = draw._validate_finite_bound(specified[0], param_name)
        for v in specified[1:]:
            vf = draw._validate_finite_bound(v, param_name)
            if vf != first_val:
                raise ValueError(
                    f"Conflicting `{param_name}` values {specified} within shared group {list(group)}."
                )
        return first_val
    return draw._validate_finite_bound(param, param_name)


def _strip_colorbar(cb: Colorbar, norm: Any, name: str) -> None:
    """Colour bar under a narrow strip: only the ends (and centre), rotated, and the strip
    name below them, so the name cannot run into the bar as it would under the strip."""
    assert norm.vmin is not None and norm.vmax is not None
    ticks = [float(norm.vmin), float(norm.vmax)]
    if isinstance(norm, mcolors.TwoSlopeNorm):
        ticks = [float(norm.vmin), float(norm.vcenter), float(norm.vmax)]
    cb.set_ticks(ticks, labels=[f"{t:.3g}" for t in ticks])
    cb.ax.tick_params(labelsize=6, length=2, labelrotation=90)
    if name:
        cb.set_label(name, fontsize=8, rotation=90, va="top", labelpad=2)


def _fit_margins(fig: Figure, pad: float = 0.08) -> None:
    """Expand bottom and/or right margins using public Matplotlib APIs while preserving
    exact physical column widths and heights in inches."""
    if not fig.axes:
        return
    fig.draw_without_rendering()
    min_y0_px = math.inf
    max_x1_px = -math.inf
    for ax in fig.axes:
        tb = ax.get_tightbbox()
        if tb is not None:
            min_y0_px = min(min_y0_px, float(tb.y0))
            max_x1_px = max(max_x1_px, float(tb.x1))
        for art in ax.artists:
            ab = art.get_window_extent()
            min_y0_px = min(min_y0_px, float(ab.y0))
            max_x1_px = max(max_x1_px, float(ab.x1))

    if not math.isfinite(min_y0_px) or not math.isfinite(max_x1_px):
        return

    W, H = fig.get_size_inches()
    low_in = min_y0_px / fig.dpi
    right_clear_in = float(W) - (max_x1_px / fig.dpi)

    d_bottom = (pad - low_in) if low_in < pad else 0.0
    d_right = (pad - right_clear_in) if right_clear_in < pad else 0.0
    if d_bottom <= 0.0 and d_right <= 0.0:
        return

    W_new = float(W) + d_right
    H_new = float(H) + d_bottom
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position().bounds
        ax.set_position(
            (
                (x0 * W) / W_new,
                (y0 * H + d_bottom) / H_new,
                (w * W) / W_new,
                (h * H) / H_new,
            )
        )
    fig.set_size_inches(W_new, H_new)


def plot(
    mats: NormalizedMatrix | Sequence[NormalizedMatrix] | Mapping[str, NormalizedMatrix],
    rows: Rows | None = None,
    annotations: Mapping[str, pd.Series] | pd.DataFrame | pd.Series | None = None,
    cmap: str | mcolors.Colormap | Mapping[str, str | mcolors.Colormap] | None = None,
    center: float | Mapping[str, float] | None = None,
    vmin: float | Mapping[str, float] | None = None,
    vmax: float | Mapping[str, float] | None = None,
    shared: Sequence[Sequence[str]] = (),
    q: float | None = 0.99,
    profile: bool = True,
    band: str | None = None,
    share_profile_y: bool = False,
    colors: str | mcolors.Colormap | Mapping[Any, Any] | Sequence[Any] = "tab10",
    width: float = 1.6,
    widths: Sequence[float] | Mapping[str, float] | None = None,
    annotation_width: float = 0.22,
    height: float = 5.0,
    group_strip: bool = True,
    agg: DownsampleAgg = "mean",
    annotation_colors: str | mcolors.Colormap | Mapping[Any, Any] | Sequence[Any] = "Set2",
    annotation_legend: bool = True,
    annotation_cmap: str | mcolors.Colormap | Mapping[str, str | mcolors.Colormap] = "viridis",
    annotation_vmin: float | Mapping[str, float] | None = None,
    annotation_vmax: float | Mapping[str, float] | None = None,
    annotation_center: float | Mapping[str, float] | None = None,
    annotation_colorbar: bool = True,
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
    cmap, center, vmin, vmax : value or dict keyed by column name
        Passed to :func:`heatmap`. Unknown column keys raise ``KeyError``.
    shared : list of lists of names
        Columns that share one colour scale, e.g. ``[["WT", "KO"]]``, computed over
        ``rows.index`` and honouring ``vmin``/``vmax``/``center``.
    q : float or None
        Quantile for ``vmax`` when it is not given.
    profile : bool
        Draw profiles above the heatmaps.
    band : str, optional
        Profile band, see :func:`profile`.
    share_profile_y : bool
        Same y-range for every profile.
    colors : colormap name, list or dict
        Group colours, used by the group strip and the profile lines.
    width : float
        Width of a heatmap column in inches.
    widths : list of float or dict keyed by column name, optional
        Relative width of each heatmap column, multiplied by ``width``: one factor per
        matrix in ``mats`` order (default all 1), e.g. ``widths=[1, 1, 0.5]`` makes the
        third column half as wide. A dict sets some columns and leaves the rest at 1.
        Annotation strips keep ``annotation_width``.
    group_strip : bool
        With groups, add a labelled colour strip on the left.
    annotation_colors : colormap name, list or dict
        Category colours of the categorical annotation strips. Differs from ``colors``
        by default so a strip is not mistaken for the groups; when several categorical
        strips share one qualitative palette (such as the default ``"Set2"``), strip
        offsets advance automatically so adjacent strips use distinct colours. A dict
        keyed by strip name gives each strip its own palette; any other dict is one
        ``{category: colour}`` mapping shared by every strip.
    annotation_legend : bool
        Legend in the right margin for every categorical annotation strip.
    annotation_cmap : colormap name or object, or dict keyed by strip name
        Colour map of numeric annotation strips, e.g. ``{"GC%": "YlGnBu"}``; strips not
        in a dict use ``"viridis"``.
    annotation_vmin, annotation_vmax, annotation_center : value or dict keyed by strip name
        Colour scale of numeric strips, as ``vmin``/``vmax``/``center`` of
        :func:`heatmap`; an unset ``annotation_vmax`` comes from the ``q`` quantile.
    annotation_colorbar : bool
        A colour bar under every numeric annotation strip, with the strip name below
        it. The figure grows at the bottom and right so long names and legends are not
        cut off.
    guides : bool
        Dashed reference lines at the anchor (and body end) in heatmaps and profiles.
    pyplot : bool
        See :func:`figure`; pass ``False`` to build many figures without leaking them.

    Returns
    -------
    fig, grid
    """
    mats_dict = _as_dict(mats)
    names = list(mats_dict)
    if not names:
        raise ValueError("`mats` must not be empty.")
    if rows is None:
        rows = arrange(mats_dict[names[0]])
    ann_dict = _as_annotations(annotations)
    ann_names = list(ann_dict)

    # Upfront validation before creating any figure so pyplot=True never leaks on error
    factors = _column_factors(widths, names)
    _validate_dict_keys(cmap, names, "cmap")
    _validate_dict_keys(center, names, "center")
    _validate_dict_keys(vmin, names, "vmin")
    _validate_dict_keys(vmax, names, "vmax")
    _validate_dict_keys(annotation_cmap, ann_names, "annotation_cmap")
    _validate_dict_keys(annotation_vmin, ann_names, "annotation_vmin")
    _validate_dict_keys(annotation_vmax, ann_names, "annotation_vmax")
    _validate_dict_keys(annotation_center, ann_names, "annotation_center")
    draw._validate_quantile_q(q)

    for n in names:
        draw._ensure_overlap(mats_dict[n].region_index, rows.index, f"mats[{n!r}]")
        _ = draw._validate_finite_bound(_pick(vmin, n), "vmin")
        _ = draw._validate_finite_bound(_pick(vmax, n), "vmax")
        _ = draw._validate_finite_bound(_pick(center, n), "center")
    for n, sr in ann_dict.items():
        draw._ensure_overlap(sr.index, rows.index, f"annotations[{n!r}]")
        _ = draw._validate_finite_bound(_pick(annotation_vmin, n), "annotation_vmin")
        _ = draw._validate_finite_bound(_pick(annotation_vmax, n), "annotation_vmax")
        _ = draw._validate_finite_bound(_pick(annotation_center, n), "annotation_center")

    palettes = _resolve_annotation_palettes(annotation_colors, ann_dict, rows)

    if isinstance(shared, (str, bytes)) or not isinstance(shared, Sequence):
        raise TypeError(
            "`shared` must be a list of groups of column names, e.g. shared=[['WT', 'KO']]."
        )
    shared_scales: dict[str, tuple[float, float, float | None]] = {}
    for group in shared:
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise TypeError(
                "`shared` must be a list of groups of column names, e.g. shared=[['WT', 'KO']]."
            )
        if not group:
            continue
        group_mats: list[NormalizedMatrix] = []
        for n in group:
            m_col = mats_dict[n]  # raises KeyError on unknown column name
            group_mats.append(
                m_col.reindex(rows.index)
                if not m_col.region_index.equals(rows.index)
                else m_col
            )
        grp_vmin = _resolve_shared_group_bound(vmin, group, "vmin")
        grp_vmax = _resolve_shared_group_bound(vmax, group, "vmax")
        grp_center_explicit = _resolve_shared_group_bound(center, group, "center")
        grp_center = draw._center_for(group_mats, grp_center_explicit)
        grp_vals = np.concatenate([gm.values.ravel() for gm in group_mats])
        grp_norm = draw._norm(grp_vals, grp_vmin, grp_vmax, q, grp_center)
        assert grp_norm.vmin is not None and grp_norm.vmax is not None
        for n in group:
            shared_scales[n] = (float(grp_norm.vmin), float(grp_norm.vmax), grp_center)

    grouped = rows.groups is not None and group_strip
    col_widths = (
        ([annotation_width] if grouped else [])
        + [width * f for f in factors]
        + [annotation_width] * len(ann_dict)
    )
    is_hm = ([False] if grouped else []) + [True] * len(names) + [False] * len(ann_dict)
    categorical = (
        [n for n, sr in ann_dict.items() if draw._is_categorical(sr)]
        if annotation_legend
        else []
    )
    numeric_cbar = [
        annotation_colorbar and not draw._is_categorical(sr) for sr in ann_dict.values()
    ]
    has_cbar = is_hm[: len(is_hm) - len(ann_dict)] + numeric_cbar
    margins = (0.7, 1.3 if categorical else 0.3, 0.35, 0.75 if any(numeric_cbar) else 0.55)
    fig, grid = figure(
        col_widths,
        profile=[profile and h for h in is_hm],
        colorbar=has_cbar,
        height=height,
        margins=margins,
        pyplot=pyplot,
    )
    off = 1 if grouped else 0

    try:
        for j, name in enumerate(names):
            ax = grid.heatmap[off + j]
            if name in shared_scales:
                lo_s, hi_s, cen_s = shared_scales[name]
                lo: float | None = lo_s
                hi: float | None = hi_s
                cen: float | None = cen_s
            else:
                lo = _pick(vmin, name)
                hi = _pick(vmax, name)
                cen = _pick(center, name)
            im = draw.heatmap(
                mats_dict[name],
                ax,
                rows,
                cmap=_pick(cmap, name),
                center=cen,
                vmin=lo,
                vmax=hi,
                q=q,
                agg=agg,
                guides=guides,
                title=None if profile else name,
            )
            cax_hm = grid.colorbar[off + j]
            assert cax_hm is not None
            draw.colorbar(im, cax_hm)
            if profile:
                pax = grid.profile[off + j]
                assert pax is not None
                draw.profile(
                    mats_dict[name],
                    pax,
                    rows,
                    band=band,
                    colors=colors,
                    guides=guides,
                    title=name,
                )

        if profile and share_profile_y:
            p_axes = [a for a in grid.profile if a is not None]
            lo_y = min(a.get_ylim()[0] for a in p_axes)
            hi_y = max(a.get_ylim()[1] for a in p_axes)
            for a in p_axes:
                a.set_ylim(lo_y, hi_y)

        if grouped:
            assert rows.groups is not None
            draw.annotation(
                rows.groups.rename(""), grid.heatmap[0], rows, colors=colors, labels=False
            )
            draw.group_labels(grid.heatmap[0], rows)

        items: list[tuple[str | None, Mapping[Any, Any]]] = []
        for k, (name, series) in enumerate(ann_dict.items()):
            col = off + len(names) + k
            cax = grid.colorbar[col]
            res = draw.annotation(
                series.rename(name),
                grid.heatmap[col],
                rows,
                colors=palettes[name],
                labels=cax is None,
                cmap=_pick(annotation_cmap, name, "viridis"),
                vmin=_pick(annotation_vmin, name),
                vmax=_pick(annotation_vmax, name),
                center=_pick(annotation_center, name),
                q=q,
                agg=agg,
            )
            if isinstance(res, dict):
                items.append((name, res))
            elif cax is not None and isinstance(res, mcm.ScalarMappable):
                _strip_colorbar(draw.colorbar(res, cax), res.norm, name)
        if items and annotation_legend:
            draw.legends(grid.heatmap[-1], items, x=1.15)
        _fit_margins(fig)
    except Exception:
        if pyplot:
            import matplotlib.pyplot as plt

            plt.close(fig)
        raise
    return fig, grid
