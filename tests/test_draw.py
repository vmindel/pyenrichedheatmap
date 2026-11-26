"""Drawing functions: alignment through Rows, downsampling, scales, layout."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import enriched_heatmap as eh  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _mat(n=40, seed=0, name="sig"):
    rng = np.random.default_rng(seed)
    vals = rng.gamma(2.0, 1.0, size=(n, 20))
    vals[:, 9:11] += np.linspace(10, 0, n)[:, None]
    idx = pd.Index([f"r{i}" for i in range(n)])
    return eh.NormalizedMatrix(vals, region_index=idx, upstream=500, downstream=500, bin_size=50, name=name)


def test_heatmap_follows_rows_order_and_groups():
    m = _mat()
    groups = pd.Series(["a"] * 20 + ["b"] * 20, index=m.region_index)
    rows = eh.arrange(m.region_index[::-1], groups=groups, gap=0.1)
    fig, ax = plt.subplots()
    eh.heatmap(m, ax, rows, max_rows=None)
    images = ax.get_images()
    assert len(images) == 2
    # First group drawn is "b" (it holds the first row of the explicit order), reversed.
    top = images[0].get_array()
    np.testing.assert_allclose(top[0], m.values[39])
    assert ax.get_ylim() == (rows.height, 0)
    assert images[1].get_extent()[3] == pytest.approx(20 + rows.gap)


def test_missing_regions_draw_as_nan():
    m = _mat()
    rows = eh.arrange(pd.Index(["r0", "nope", "r1"]))
    fig, ax = plt.subplots()
    eh.heatmap(m, ax, rows, max_rows=None)
    arr = ax.get_images()[0].get_array()
    assert np.isnan(np.asarray(arr[1], dtype=float)).all()


def test_downsampling_caps_drawn_rows():
    m = _mat(n=5000)
    fig, ax = plt.subplots(figsize=(2, 1))
    eh.heatmap(m, ax, max_rows=100)
    assert ax.get_images()[0].get_array().shape[0] == 100
    eh.heatmap(m, ax, max_rows="auto")
    assert ax.get_images()[-1].get_array().shape[0] < 5000


def test_scales_center_and_limits():
    a, b = _mat(seed=1), _mat(seed=2)
    lo, hi = eh.limits([a, b], q=1.0)
    assert lo == 0.0 and hi == pytest.approx(max(a.values.max(), b.values.max()))
    fig, ax = plt.subplots()
    im = eh.heatmap(a.log2_ratio(b), ax, center=0)
    assert im.norm.vcenter == 0
    assert -im.norm.vmin == pytest.approx(im.norm.vmax)


def test_profile_one_line_per_group_and_bands():
    m = _mat()
    rows = eh.arrange(m, groups=pd.Series(["x"] * 10 + ["y"] * 30, index=m.region_index))
    fig, ax = plt.subplots()
    eh.profile(m, ax, rows, band="sem", legend=True)
    assert len(ax.lines) - 1 == 2  # plus the guide line
    assert len(ax.collections) == 2
    with pytest.raises(ValueError):
        eh.profile(m, ax, stat="median", band="sem")


def test_annotation_kinds():
    m = _mat()
    rows = eh.arrange(m)
    cats = pd.Series(np.where(np.arange(40) % 2, "odd", "even"), index=m.region_index)
    fig, axes = plt.subplots(1, 3)
    colors = eh.annotation(cats, axes[0], rows)
    assert set(colors) == {"even", "odd"}
    im = eh.annotation(pd.DataFrame({"u": np.arange(40.0), "v": 1.0}, index=m.region_index),
                       axes[1], rows)
    assert len(axes[1].images) == 2  # one strip per column, on one shared scale
    assert axes[1].images[0].norm is axes[1].images[1].norm is im.norm
    eh.annotation(pd.Series(np.arange(40.0), index=m.region_index, name="n"), axes[2], rows, kind="bar")
    assert len(axes[2].patches) == 40


def test_figure_grid_and_plot():
    fig, grid = eh.figure([2, 2, 0.3], profile=[True, True, False], colorbar=[True, True, False])
    assert grid.profile[2] is None and grid.colorbar[2] is None
    assert grid.heatmap[1].get_shared_y_axes().joined(grid.heatmap[0], grid.heatmap[1])

    a, b = _mat(seed=1, name="A"), _mat(seed=2, name="B")
    rows = eh.arrange(a, groups=eh.kmeans([a, b], k=2))
    fig, grid = eh.plot({"A": a, "B": b}, rows=rows, shared=[["A", "B"]],
                        annotations={"s": eh.score(a)})
    ims = [grid.heatmap[i].get_images()[0] for i in (1, 2)]
    assert ims[0].norm.vmax == ims[1].norm.vmax
    assert len(grid.heatmap) == 4  # group strip, A, B, annotation


# ---------------------------------------------------------------------------
# Regressions: annotation dispatch, colour defaults, differential scale, legends
# ---------------------------------------------------------------------------
def _labels(m):
    return pd.Series(np.where(np.arange(m.n_regions) % 3 == 0, "a", "b"), index=m.region_index)


def test_annotation_draws_every_categorical_column():
    m = _mat()
    df = pd.DataFrame({"x": _labels(m), "y": pd.Categorical(np.where(np.arange(40) < 20, "u", "v"),
                                                             categories=["v", "u"])},
                      index=m.region_index)
    fig, ax = plt.subplots()
    res = eh.annotation(df, ax)
    assert set(res) == {"x", "y"}
    assert list(res["y"]) == ["v", "u"]  # category order kept per column
    assert len(ax.images) == 2
    assert ax.get_xlim() == (0.0, 2.0)  # both strips inside the axes
    assert [t.get_text() for t in ax.get_xticklabels()] == ["x", "y"]


def test_annotation_mixes_numeric_and_categorical_columns():
    m = _mat()
    df = pd.DataFrame({"score": np.arange(40.0), "cls": _labels(m), "s2": np.arange(40.0)[::-1]},
                      index=m.region_index)
    fig, ax = plt.subplots()
    res = eh.annotation(df, ax)  # used to raise "could not convert string to float"
    assert set(res) == {"score", "cls", "s2"}
    assert isinstance(res["cls"], dict)
    assert res["score"].norm is res["s2"].norm  # numeric columns share a scale
    assert len(ax.images) == 3
    # the strip of each column sits in its own unit-wide slot
    assert sorted(im.get_extent()[0] for im in ax.images) == [0, 1, 2]


def test_annotation_handles_nullable_and_missing_values():
    m = _mat()
    s = pd.Series(pd.array([1, None, 3] * 13 + [4], dtype="Int64"), index=m.region_index, name="n")
    fig, ax = plt.subplots()
    im = eh.annotation(s, ax, max_rows=None)
    assert np.ma.getmaskarray(im.get_array())[1, 0]  # missing value is masked, not 0


def test_annotation_rejects_unknown_kind_and_bar_with_several_columns():
    m = _mat()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="kind"):
        eh.annotation(_labels(m), ax, kind="bars")
    with pytest.raises(ValueError, match="single column"):
        eh.annotation(pd.DataFrame({"a": 1.0, "b": 2.0}, index=m.region_index), ax, kind="bar")


def test_group_strip_and_profile_lines_share_default_colours():
    m = _mat()
    clusters = pd.Series(np.where(np.arange(40) < 15, "C1", np.where(np.arange(40) < 30, "C2", "C3")),
                         index=m.region_index)
    rows = eh.arrange(m, groups=clusters)
    fig, (a1, a2) = plt.subplots(1, 2)
    strip = eh.annotation(clusters, a1, rows)
    eh.profile(m, a2, rows)
    lines = {ln.get_label(): ln.get_color() for ln in a2.lines if ln.get_label() in strip}
    assert {k: mpl_hex(v) for k, v in lines.items()} == {k: mpl_hex(v) for k, v in strip.items()}


def mpl_hex(c):
    from matplotlib.colors import to_hex

    return to_hex(c)


def test_differential_matrices_default_to_a_centred_diverging_scale():
    a, b = _mat(seed=1, name="A"), _mat(seed=2, name="B")
    diff = a.log2_ratio(b)
    fig, ax = plt.subplots()
    im = eh.heatmap(diff, ax)
    assert im.cmap.name == "RdBu_r"
    assert im.norm.vcenter == 0.0
    assert eh.limits([diff]) == pytest.approx((-eh.limits([diff])[1], eh.limits([diff])[1]))

    # an explicit choice still wins, and ordinary matrices are untouched
    im2 = eh.heatmap(diff, plt.subplots()[1], cmap="viridis", center=1.0)
    assert (im2.cmap.name, im2.norm.vcenter) == ("viridis", 1.0)
    assert eh.heatmap(a, plt.subplots()[1]).cmap.name == "Reds"
    # a plain matrix mixed with a differential one does not get centred
    assert eh.limits([a, diff])[0] == pytest.approx(min(0.0, float(np.nanmin(np.r_[a.values.ravel(), diff.values.ravel()]))))


def test_plot_adds_legends_for_categorical_annotations():
    m = _mat()
    fig, grid = eh.plot(m, annotations={"class": _labels(m), "score": pd.Series(
        np.arange(40.0), index=m.region_index)})
    last = grid.heatmap[-1]
    legends = [a for a in last.artists if a.__class__.__name__ == "Legend"]
    assert len(legends) == 1  # categorical strip only
    assert legends[0].get_title().get_text() == "class"
    assert {t.get_text() for t in legends[0].get_texts()} == {"a", "b"}
    # legends sit right of the strips and are not clipped away by the axes
    assert not legends[0].get_clip_on()
    fig.canvas.draw()
    assert legends[0].get_window_extent().x0 > last.get_window_extent().x1
    assert legends[0].get_window_extent().x1 <= fig.bbox.x1 + 1  # inside the figure margin

    fig2, grid2 = eh.plot(m, annotations={"class": _labels(m)}, annotation_legend=False)
    assert not [a for a in grid2.heatmap[-1].artists if a.__class__.__name__ == "Legend"]


def test_plot_annotation_colours_differ_from_group_colours_by_default():
    m = _mat()
    rows = eh.arrange(m, groups=_labels(m))
    fig, grid = eh.plot(m, rows=rows, annotations={"class": _labels(m)})
    group_cmap = grid.heatmap[0].images[0].cmap
    annot_cmap = grid.heatmap[-1].images[0].cmap
    assert list(group_cmap.colors) != list(annot_cmap.colors)


def test_zero_row_figures_do_not_warn(recwarn):
    import warnings

    m = _mat()
    empty = m.iloc[:0]
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eh.heatmap(empty, ax)
        eh.annotation(pd.Series([], dtype=object, index=empty.region_index, name="x"),
                      plt.subplots()[1], eh.arrange(empty))


def test_heatmap_rows_missing_from_matrix_are_nan():
    m = _mat()
    rows = eh.Rows(index=pd.Index(["r0", "unknown"]))
    fig, ax = plt.subplots()
    im = eh.heatmap(m, ax, rows, max_rows=None)
    arr = np.asarray(im.get_array())
    assert arr.shape[0] == 2
    assert np.isnan(arr[1]).all() and np.isfinite(arr[0]).all()


# ---------------------------------------------------------------------------
# max_rows="auto" under a layout engine, and figures without pyplot
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["constrained", "tight"])
def test_auto_max_rows_measures_the_laid_out_axes(layout):
    rng = np.random.default_rng(0)
    big = eh.NormalizedMatrix(rng.random((20000, 20)), upstream=500, downstream=500, bin_size=50)
    fig, axes = plt.subplots(2, 1, figsize=(3, 6), height_ratios=[1, 4], layout=layout)
    im = eh.heatmap(big, axes[1], max_rows="auto")
    drawn = np.asarray(im.get_array()).shape[0]
    fig.canvas.draw()  # the layout engine moves the axes
    final = round(axes[1].get_window_extent().height / fig.dpi * 300)
    # the heatmap's own tick labels nudge the layout a little afterwards (~3%); unmeasured
    # axes were 20% off
    assert drawn == pytest.approx(final, rel=0.05)


def test_figure_without_pyplot_is_not_registered(tmp_path):
    from matplotlib.figure import Figure

    before = set(plt.get_fignums())
    fig, grid = eh.figure([1.0, 1.0], pyplot=False)
    assert isinstance(fig, Figure) and set(plt.get_fignums()) == before
    m = _mat()
    fig2, grid2 = eh.plot(m, rows=eh.arrange(m), pyplot=False)
    assert set(plt.get_fignums()) == before
    fig2.savefig(tmp_path / "x.png")  # a plain Figure can still be saved
    assert (tmp_path / "x.png").stat().st_size > 0
    fig3, _ = eh.figure([1.0])
    assert set(plt.get_fignums()) - before == {fig3.number}  # default stays pyplot-managed


def test_importing_the_package_does_not_import_pyplot():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, enriched_heatmap; print('matplotlib.pyplot' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"


def test_profile_bootstrap_band_is_exposed_and_reproducible():
    m = _mat()
    fig, ax = plt.subplots()
    eh.profile(m, ax, band="boot", n_boot=50)
    assert len(ax.collections) == 1
    f2, a2 = plt.subplots()
    eh.profile(m, a2, band="boot", n_boot=50)
    np.testing.assert_array_equal(ax.collections[0].get_paths()[0].vertices,
                                  a2.collections[0].get_paths()[0].vertices)


def test_plot_accepts_annotations_as_dataframe_or_series():
    m = _mat()
    labels = _labels(m)
    df = pd.DataFrame({"class": labels, "score": np.arange(40.0)}, index=m.region_index)
    fig, grid = eh.plot(m, annotations=df)
    assert len(grid.heatmap) == 3  # heatmap + one strip per column
    legends = [a for a in grid.heatmap[-1].artists if a.__class__.__name__ == "Legend"]
    assert [lg.get_title().get_text() for lg in legends] == ["class"]
    fig2, grid2 = eh.plot(m, annotations=labels.rename("class"))
    assert len(grid2.heatmap) == 2
    with pytest.raises(TypeError, match="annotations"):
        eh.plot(m, annotations=[labels])


def test_annotation_without_shared_identifiers_raises():
    m = _mat()
    rows = eh.arrange(m)
    positional = pd.Series(_labels(m).values)  # RangeIndex: matched by position by mistake
    with pytest.raises(ValueError, match="no region identifiers"):
        eh.annotation(positional, plt.subplots()[1], rows)
    with pytest.raises(ValueError, match="no region identifiers"):
        eh.plot(m, rows=rows, annotations=pd.DataFrame({"cCRE": _labels(m).values}))
    # partial overlap stays allowed: missing regions are blank
    eh.annotation(_labels(m).iloc[:5], plt.subplots()[1], rows)


def test_plot_guides_switch():
    m = _mat()
    def dashed(ax):
        return [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    fig, grid = eh.plot(m)
    assert dashed(grid.heatmap[0]) and dashed(grid.profile[0])
    fig, grid = eh.plot(m, guides=False)
    assert not dashed(grid.heatmap[0]) and not dashed(grid.profile[0])


def test_centred_matrix_edge_labels_point_inwards():
    vals = np.ones((10, 2000 // 20 * 2 + 1))
    m = eh.NormalizedMatrix(vals, upstream=2000, downstream=2000, bin_size=20, centered=True)
    fig, ax = plt.subplots()
    eh.heatmap(m, ax)
    tl = ax.get_xticklabels()
    assert tl[0].get_horizontalalignment() == "left"
    assert tl[-1].get_horizontalalignment() == "right"
    assert [t.get_text() for t in tl][1] == "0"


def test_centered_heatmap_mirrors_single_bound_and_plot_validates_shared():
    a, b = _mat(seed=1, name="A"), _mat(seed=2, name="B")
    diff = a.log2_ratio(b)
    fig, ax = plt.subplots()
    im = eh.heatmap(diff, ax, vmax=5.0)
    assert (im.norm.vmin, im.norm.vcenter, im.norm.vmax) == (-5.0, 0.0, 5.0)
    im_lo = eh.heatmap(diff, plt.subplots()[1], vmin=-4.0)
    assert (im_lo.norm.vmin, im_lo.norm.vcenter, im_lo.norm.vmax) == (-4.0, 0.0, 4.0)

    with pytest.raises(TypeError, match="shared"):
        eh.plot({"A": a, "B": b}, shared=["A", "B"], pyplot=False)
    with pytest.raises(ValueError, match="empty"):
        eh.plot({}, pyplot=False)
    with pytest.raises(ValueError, match="empty"):
        eh.figure([], pyplot=False)
    with pytest.raises(ValueError, match="empty"):
        eh.limits([])


def test_annotation_nullable_boolean_unnamed_series_and_palette_colormap():
    m = _mat()
    nullable = pd.Series(
        [True, False] * 20, index=m.region_index, dtype="boolean", name="bound"
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    res = eh.annotation(nullable, ax1)
    assert isinstance(res, dict) and set(res) == {False, True}

    unnamed = pd.Series(np.arange(40.0), index=m.region_index)
    eh.annotation(unnamed, ax2)
    assert [t.get_text() for t in ax2.get_xticklabels()] == []
    eh.annotation(unnamed, ax3, kind="bar")
    assert ax3.get_xlabel() == ""

    assert eh.palette(["a", "b"], matplotlib.colormaps["Set2"]) == eh.palette(["a", "b"], "Set2")
    with pytest.raises(ValueError, match="empty"):
        eh.palette(["a"], [])


def test_plot_per_annotation_numeric_scales_and_colorbars():
    m = _mat()
    gc = pd.Series(np.linspace(30, 60, 40), index=m.region_index)
    lfc = pd.Series(np.linspace(-2, 3, 40), index=m.region_index)
    rows = eh.arrange(m, groups=_labels(m))
    fig, grid = eh.plot(m, rows=rows, annotations={"class": _labels(m), "GC%": gc, "lfc": lfc},
                        annotation_cmap={"GC%": "YlGnBu", "lfc": "RdBu_r"},
                        annotation_vmin={"GC%": 35}, annotation_vmax={"GC%": 55},
                        annotation_center={"lfc": 0})
    ax_class, ax_gc, ax_lfc = grid.heatmap[-3:]
    # every group block of a numeric strip is recoloured, not only the first
    assert len(ax_gc.images) == 2
    for im in ax_gc.images:
        assert im.cmap.name.startswith("YlGnBu")
        assert (im.norm.vmin, im.norm.vmax) == (35, 55)
    for im in ax_lfc.images:
        assert im.cmap.name.startswith("RdBu_r") and im.norm.vcenter == 0
    # colour bars under numeric strips only
    assert grid.colorbar[-3] is None
    assert grid.colorbar[-2] is not None and grid.colorbar[-1] is not None
    assert [t.get_text() for t in grid.colorbar[-2].get_xticklabels()] == ["35", "55"]
    assert len(grid.colorbar[-1].get_xticks()) == 3  # vmin, centre, vmax
    fig2, grid2 = eh.plot(m, annotations={"GC%": gc}, annotation_colorbar=False)
    assert grid2.colorbar[-1] is None
    assert grid2.heatmap[-1].images[0].cmap.name.startswith("viridis")


def test_plot_annotation_colors_per_strip_or_shared():
    m = _mat()
    a, b = _labels(m), _labels(m).map({"a": "x", "b": "y"})
    fig, grid = eh.plot(m, annotations={"A": a, "B": b},
                        annotation_colors={"A": {"a": "red", "b": "blue"}, "B": ["green", "black"]})
    lut_a = grid.heatmap[-2].images[0].cmap
    lut_b = grid.heatmap[-1].images[0].cmap
    import matplotlib.colors as mc
    assert mc.to_hex(lut_a(0)) == "#ff0000" and mc.to_hex(lut_b(0)) == "#008000"
    # a colormap name that is also a CSS colour name (e.g. "pink" or "gray") still works per-strip
    fig, grid = eh.plot(m, annotations={"A": a}, annotation_colors={"A": "pink"}, pyplot=False)
    expected_pink = eh.palette(["a", "b"], "pink")
    assert mc.to_hex(grid.heatmap[-1].images[0].cmap(0)) == mc.to_hex(expected_pink["a"])
    # a misspelt colormap name raises instead of silently making the dict a shared mapping
    with pytest.raises(KeyError, match="Set2_typo"):
        eh.plot(m, annotations={"A": a, "B": b},
                annotation_colors={"A": "Set2_typo", "B": "Dark2"}, pyplot=False)
    # a {category: colour} dict is still shared across strips
    fig, grid = eh.plot(m, annotations={"A": a}, annotation_colors={"a": "red", "b": "blue"})
    assert mc.to_hex(grid.heatmap[-1].images[0].cmap(1)) == "#0000ff"


def test_plot_relative_column_widths():
    a, b, c = _mat(name="a"), _mat(seed=1, name="b"), _mat(seed=2, name="c")
    mats = {"a": a, "b": b, "c": c}
    def w(ax):
        return ax.get_position().width * ax.figure.get_figwidth()
    fig, grid = eh.plot(mats, width=2.0)
    assert [round(w(ax), 3) for ax in grid.heatmap] == [2.0, 2.0, 2.0]
    fig, grid = eh.plot(mats, width=2.0, widths=[1, 1, 0.5])
    assert [round(w(ax), 3) for ax in grid.heatmap] == [2.0, 2.0, 1.0]
    # profile and colour bar follow their heatmap
    assert round(w(grid.profile[2]), 3) == 1.0 and round(w(grid.colorbar[2]), 3) == 1.0
    fig, grid = eh.plot(mats, width=2.0, widths={"b": 1.5}, annotations={"s": eh.score(a)})
    assert [round(w(ax), 3) for ax in grid.heatmap[:3]] == [2.0, 3.0, 2.0]
    assert round(w(grid.heatmap[3]), 3) == 0.22  # annotation strip unchanged
    fig, grid = eh.plot(mats, widths=np.array([1, 2, 1]))
    with pytest.raises(ValueError, match="3 matrices"):
        eh.plot(mats, widths=[1, 0.5])
    with pytest.raises(KeyError):
        eh.plot(mats, widths={"zzz": 0.5})
    for bad_w in ([1, 0, 1], [1, np.nan, 1], [1, np.inf, 1]):
        with pytest.raises(ValueError, match="positive"):
            eh.plot(mats, widths=bad_w)
    with pytest.raises(ValueError, match="Duplicate matrix name"):
        eh.plot([_mat(name="dup"), _mat(name="dup")], pyplot=False)


@pytest.mark.parametrize("pyplot", [True, False])
def test_plot_numeric_strip_name_clears_its_colorbar(pyplot):
    m = _mat()
    gc = pd.Series(np.linspace(-1e4, 2e4, 40), index=m.region_index)
    fig, grid = eh.plot(m, annotations={"GC content (%)": gc, "class": _labels(m)},
                        pyplot=pyplot)
    ax_gc, ax_class = grid.heatmap[-2:]
    cax = grid.colorbar[-2]
    # the name moves from under the strip to under the colour bar's tick labels
    assert [t.get_text() for t in ax_gc.get_xticklabels()] == []
    assert cax.get_xlabel() == "GC content (%)"
    assert [t.get_text() for t in ax_class.get_xticklabels()] == ["class"]
    # measure at the dpi it was drawn at: label positions are stored in pixels
    fig.savefig(io.BytesIO(), format="png", dpi=fig.dpi)
    name = cax.xaxis.label.get_window_extent()
    assert name.y1 <= cax.get_window_extent().y0
    assert all(name.y1 <= t.get_window_extent().y0 + 1e-6
               for t in cax.get_xticklabels() if t.get_text())
    assert name.y0 > 0  # inside the figure without bbox_inches="tight"
    # no colour bar: the name stays under the strip
    fig, grid = eh.plot(m, annotations={"GC%": gc}, annotation_colorbar=False, pyplot=pyplot)
    assert [t.get_text() for t in grid.heatmap[-1].get_xticklabels()] == ["GC%"]


def test_plot_vmin_shared_overrides_multi_strip_palettes_and_annotation_kinds():
    from matplotlib.legend import Legend

    a, b = _mat(name="WT"), _mat(seed=2, name="KO")
    fig, grid = eh.plot(
        {"WT": a, "KO": b},
        shared=[["WT", "KO"]],
        vmin={"WT": -1.5, "KO": -1.5},
        vmax=8.0,
        annotations={
            "cls_a": pd.Series(["x", "y"] * 20, index=a.region_index),
            "cls_b": pd.Series(["u", "v"] * 20, index=a.region_index),
        },
        height=2.0,
        pyplot=False,
    )
    assert grid.heatmap[0].images[0].norm.vmin == pytest.approx(-1.5)
    assert grid.heatmap[1].images[0].norm.vmax == pytest.approx(8.0)

    legs = [art for art in grid.heatmap[-1].artists if isinstance(art, Legend)]
    assert len(legs) == 2
    fig.draw_without_rendering()
    assert legs[0].get_window_extent().y0 >= legs[1].get_window_extent().y1
    c_a = legs[0].get_patches()[0].get_facecolor()
    c_b = legs[1].get_patches()[0].get_facecolor()
    assert c_a != c_b

    with pytest.raises(ValueError, match="Conflicting"):
        eh.plot({"WT": a, "KO": b}, shared=[["WT", "KO"]], vmax={"WT": 5.0, "KO": 9.0}, pyplot=False)
    with pytest.raises(KeyError):
        eh.plot({"WT": a}, vmin={"typo": 0.0}, pyplot=False)

    int_codes = pd.Series([10, 2, 1, 0] * 10, index=a.region_index)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 2))
    cat_map = eh.annotation(int_codes, ax1, kind="categorical", max_rows=None)
    assert list(cat_map.keys()) == [0, 1, 2, 10]
    num_im = eh.annotation(int_codes, ax2, kind="numeric", q=None, max_rows=None)
    assert num_im.norm.vmin == 0.0 and num_im.norm.vmax == 10.0
    plt.close(fig2)



def test_differential_auto_centre_yields_to_one_sided_bounds():
    import matplotlib.colors as mcolors

    a, b = _mat(name="WT"), _mat(seed=2, name="KO")
    d = b.log2_ratio(a)
    fig, ax = plt.subplots()
    assert isinstance(eh.heatmap(d, ax).norm, mcolors.TwoSlopeNorm)
    # vmax alone still mirrors around the automatic centre
    norm = eh.heatmap(d, ax, vmax=2.0).norm
    assert isinstance(norm, mcolors.TwoSlopeNorm) and norm.vmin == pytest.approx(-2.0)
    # A bound 0 cannot sit inside asks for a one-sided scale instead of raising
    norm = eh.heatmap(d, ax, vmin=0.1, vmax=2.0).norm
    assert not isinstance(norm, mcolors.TwoSlopeNorm)
    assert (norm.vmin, norm.vmax) == (pytest.approx(0.1), pytest.approx(2.0))
    # An explicit centre still has to lie between the bounds
    with pytest.raises(ValueError, match="strictly less than `center`"):
        eh.heatmap(d, ax, center=0.0, vmin=0.0)
    plt.close(fig)

    # A scalar vmin=0 for signal columns no longer breaks a ratio column next to them
    fig, grid = eh.plot({"WT": a, "KO": b, "KO/WT": d}, vmin=0.0, pyplot=False)
    assert grid.heatmap[2].images[0].norm.vmin == 0.0


def test_plot_annotation_palettes_follow_labels_and_kinds():
    from matplotlib.legend import Legend

    a = _mat()
    s1 = pd.Series(["bound", "unbound"] * 20, index=a.region_index)
    s2 = pd.Series(["unbound", "bound", "weak", "bound"] * 10, index=a.region_index)
    s3 = pd.Series(["p", "q"] * 20, index=a.region_index)
    fig, grid = eh.plot(a, annotations={"s1": s1, "s2": s2, "s3": s3}, pyplot=False)
    legs = [art for art in grid.heatmap[-1].artists if isinstance(art, Legend)]
    pals = [
        {t.get_text(): p.get_facecolor() for t, p in zip(L.get_texts(), L.get_patches(), strict=True)}
        for L in legs
    ]
    # Same label, same colour on every strip; different labels never share one here
    assert pals[0]["bound"] == pals[1]["bound"] and pals[0]["unbound"] == pals[1]["unbound"]
    every = [c for p in pals for lbl, c in p.items() if lbl not in ("bound", "unbound")]
    every += [pals[0]["bound"], pals[0]["unbound"]]
    assert len(set(every)) == len(every)

    # A typo in a shared {category: colour} mapping is reported as a colour error
    with pytest.raises(ValueError, match="RGBA"):
        eh.plot(a, annotations={"s1": s1, "s3": s3},
                annotation_colors={"bound": "red", "unbound": "bleu"}, pyplot=False)
    with pytest.raises(KeyError, match="unknown strip keys"):
        eh.plot(a, annotations={"s1": s1}, annotation_colors={"s9": "Dark2"}, pyplot=False)

    # Integer cluster codes as discrete categories through plot()
    codes = pd.Series([0, 1, 2, 10] * 10, index=a.region_index)
    fig, grid = eh.plot(a, annotations={"cl": codes}, annotation_kind={"cl": "categorical"},
                        pyplot=False)
    legs = [art for art in grid.heatmap[-1].artists if isinstance(art, Legend)]
    assert [t.get_text() for t in legs[0].get_texts()] == ["0", "1", "2", "10"]
    assert grid.colorbar[-1] is None
    fig, grid = eh.plot(a, annotations={"cl": codes}, pyplot=False)
    assert grid.colorbar[-1] is not None  # default: numeric strip with a colour bar
    with pytest.raises(ValueError, match="annotation_kind"):
        eh.plot(a, annotations={"cl": codes}, annotation_kind="bar", pyplot=False)
    with pytest.raises(KeyError):
        eh.plot(a, annotations={"cl": codes}, annotation_kind={"typo": "numeric"}, pyplot=False)
