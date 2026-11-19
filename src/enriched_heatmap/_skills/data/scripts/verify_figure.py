"""Verify enriched_heatmap alignment, rasterization, and compact PDF generation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import enriched_heatmap as eh


def run_check(n_regions: int = 3000, n_bins: int = 100, out_pdf: Path | None = None) -> int:
    rng = np.random.default_rng(0)
    idx = pd.Index([f"reg_{i:06d}" for i in range(n_regions)])
    x = np.linspace(-1000, 1000, n_bins)

    amps = np.linspace(10.0, 0.5, n_regions, dtype=np.float32)[:, None]
    vals_a = amps * np.exp(-0.5 * (x / 100.0) ** 2)[None, :].astype(np.float32) + rng.uniform(
        0, 0.3, (n_regions, n_bins)
    ).astype(np.float32)
    vals_b = amps * 0.7 * np.exp(-0.5 * (x / 200.0) ** 2)[None, :].astype(np.float32) + rng.uniform(
        0, 0.2, (n_regions, n_bins)
    ).astype(np.float32)

    mat_a = eh.from_array(vals_a, region_index=idx, upstream=1000, downstream=1000, bin_size=20, name="ChEC")
    mat_b = eh.from_array(vals_b, region_index=idx, upstream=1000, downstream=1000, bin_size=20, name="ATAC")
    # Shuffle annotation index to test zero-desync alignment
    cls = pd.Series(rng.choice(["Promoter", "Enhancer"], size=n_regions), index=idx).sample(
        frac=1.0, random_state=1
    )

    hm = (
        eh.EnrichedTrack(mat_a, cmap="Reds", top_profile="mean_ci", downsample_agg="q90")
        + eh.EnrichedTrack(mat_b, cmap="Blues", top_profile="mean")
        + eh.CategoricalTrack(cls, name="Class", width=0.25)
    )
    fig, info = hm.plot(order_by="ChEC", split_by=3, figsize=(9, 6), raster_dpi=200, return_axes=True)

    # 1. Exact vertical cluster alignment across all tracks and cluster strip
    strip_axes = info["axes"]["cluster_strip"]
    for c_i, ref_ax in enumerate(strip_axes):
        ref_box = ref_ax.get_position()
        for t_key, ax_list in info["axes"]["body"].items():
            box = ax_list[c_i].get_position()
            if not (np.isclose(box.y0, ref_box.y0, atol=1e-6) and np.isclose(box.y1, ref_box.y1, atol=1e-6)):
                print(f"FAIL: vertical alignment mismatch on {t_key} cluster {c_i}")
                return 1

    # 2. Hybrid raster/vector verification
    body_img = info["axes"]["body"]["ChEC"][0].images[0]
    top_lines = info["axes"]["top"]["ChEC"].lines
    if not body_img.get_rasterized():
        print("FAIL: heatmap body is not rasterized")
        return 1
    if any(ln.get_rasterized() for ln in top_lines):
        print("FAIL: top metaplot line was unexpectedly rasterized")
        return 1

    # 3. Compact PDF output check
    with tempfile.TemporaryDirectory() as td:
        target = out_pdf if out_pdf is not None else Path(td) / "verify.pdf"
        fig.savefig(target, dpi=200)
        size_kb = target.stat().st_size / 1024.0
        plt.close(fig)
        print(
            f"OK: n_regions={n_regions}, clusters={len(strip_axes)}, "
            f"raster_rows={sum(ax.images[0].get_array().shape[0] for ax in info['axes']['body']['ChEC'])}, "
            f"pdf_size={size_kb:.1f} KB"
        )
        if size_kb > 2048:
            print(f"FAIL: PDF unexpectedly large ({size_kb:.1f} KB)")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify enriched_heatmap alignment and hybrid PDF output.")
    parser.add_argument("--n-regions", type=int, default=3000)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--out-pdf", type=Path, default=None)
    args = parser.parse_args()
    return run_check(n_regions=args.n_regions, n_bins=args.n_bins, out_pdf=args.out_pdf)


if __name__ == "__main__":
    raise SystemExit(main())
