"""Drawing functions: alignment through Rows, downsampling, scales, layout."""

from __future__ import annotations

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
