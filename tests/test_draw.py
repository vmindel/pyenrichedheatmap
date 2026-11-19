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
    assert im.get_array().shape[1] == 2
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
