"""Execute the tutorial code, check its printed output and (re)draw its figures.

Convention in docs/tutorials/*.md:

* ```` ```python ```` blocks are executed in order, one shared namespace per
  page, inside a temporary working directory, with the Agg backend.
* A ```` ```text ```` block directly after a python block is its expected
  stdout.
* Image links ``![...](../images/<name>.png)`` between a python block and the
  next python block are the figures that block draws, in order of creation.
  Figures are closed after every block.
* ```` ```py ````, ```` ```bash ```` and other blocks are shown but never
  executed (for things needing your own files, a cluster...).
* A page containing ``<!-- requires: pkg -->`` is skipped when ``pkg`` cannot be
  imported (the pymisha page runs against pymisha's bundled example database).

    python docs/run_docs.py            # check all tutorials, exit 1 on problems
    python docs/run_docs.py --update   # rewrite text blocks and images from fresh output
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS_DIR = Path(__file__).parent
TUTORIALS = DOCS_DIR / "tutorials"
IMAGES = DOCS_DIR / "images"
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*\n", re.S | re.M)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+\.png)\)")
REQUIRES = re.compile(r"<!-- requires: ([\w.]+) -->")
_DIM_TOL = 0.15


def missing_requirement(path: Path) -> str | None:
    """Name of a package the page needs that is not importable, else None."""
    import importlib.util

    for pkg in REQUIRES.findall(path.read_text(encoding="utf-8")):
        if importlib.util.find_spec(pkg) is None:
            return pkg
    return None


def _blocks(text: str):
    """Yield (lang, body, start, end) for every fenced block."""
    for m in FENCE.finditer(text):
        yield m.group(1), m.group(2), m.start(), m.end()


def _png_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def run_page(path: Path, save_images: bool = False) -> list[dict[str, Any]]:
    """Run a page.

    Returns one dict per python block: code, expected stdout (or None), actual
    stdout, the span of the expected block, and the image paths it should draw
    with the number of figures it did draw and their rendered PNG dimensions.
    """
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 100)
    pd.set_option("display.max_columns", 20)
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 110})

    text = path.read_text(encoding="utf-8")
    blocks = list(_blocks(text))
    py_idx = [i for i, b in enumerate(blocks) if b[0] == "python"]
    results: list[dict[str, Any]] = []
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for k, i in enumerate(py_idx):
                _, body, _, end = blocks[i]
                expected, span = None, None
                if i + 1 < len(blocks):
                    nlang, nbody, nstart, nend = blocks[i + 1]
                    if nlang == "text" and text[end:nstart].strip() == "":
                        expected, span = nbody, (nstart, nend)
                stop = blocks[py_idx[k + 1]][2] if k + 1 < len(py_idx) else len(text)
                images = [(path.parent / m).resolve() for m in IMAGE.findall(text[end:stop])]

                before = set(plt.get_fignums())
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(body, f"{path.name}:block{k}", "exec"), namespace)
                new = [n for n in plt.get_fignums() if n not in before]
                rendered_dims: list[tuple[int, int] | None] = []
                if save_images:
                    for img, num in zip(images, new, strict=False):
                        img.parent.mkdir(parents=True, exist_ok=True)
                        plt.figure(num).savefig(img, bbox_inches="tight")
                else:
                    for _img, num in zip(images, new, strict=False):
                        png_buf = io.BytesIO()
                        plt.figure(num).savefig(png_buf, format="png", bbox_inches="tight")
                        rendered_dims.append(_png_dimensions(png_buf.getvalue()))
                plt.close("all")
                results.append(
                    {
                        "code": body,
                        "expected": expected,
                        "actual": buf.getvalue(),
                        "span": span,
                        "images": images,
                        "n_figures": len(new),
                        "rendered_dims": rendered_dims,
                    }
                )
        finally:
            os.chdir(cwd)
    return results


def _norm(s: str | None) -> str:
    return "\n".join(line.rstrip() for line in (s or "").strip("\n").splitlines())


def check_page(path: Path) -> list[str]:
    errors: list[str] = []
    if missing_requirement(path):
        return errors
    for r in run_page(path):
        first = r["code"].strip().splitlines()[0]
        where = f"{path.name}: block starting {first!r}"
        if r["expected"] is not None and _norm(r["expected"]) != _norm(r["actual"]):
            errors.append(
                f"{where}: output changed:\n--- expected\n{_norm(r['expected'])}\n"
                f"--- actual\n{_norm(r['actual'])}"
            )
        elif r["expected"] is None and r["actual"].strip():
            errors.append(f"{where}: prints output but has no text block")
        if len(r["images"]) != r["n_figures"]:
            errors.append(
                f"{where}: draws {r['n_figures']} figure(s) but links {len(r['images'])} image(s)"
            )
        for idx, img in enumerate(r["images"]):
            if not img.is_file():
                errors.append(f"{where}: image {img.name} missing; run with --update")
                continue
            if idx < len(r["rendered_dims"]) and r["rendered_dims"][idx] is not None:
                fresh_w, fresh_h = r["rendered_dims"][idx]
                disk_dims = _png_dimensions(img.read_bytes())
                if disk_dims is None:
                    errors.append(f"{where}: image {img.name} is not a valid PNG; run with --update")
                else:
                    disk_w, disk_h = disk_dims
                    if (
                        abs(disk_w - fresh_w) / max(fresh_w, 1) > _DIM_TOL
                        or abs(disk_h - fresh_h) / max(fresh_h, 1) > _DIM_TOL
                    ):
                        errors.append(
                            f"{where}: image {img.name} dimensions ({disk_w}x{disk_h}) "
                            f"are stale vs fresh render ({fresh_w}x{fresh_h}); run with --update"
                        )
    return errors


def _overview_referenced() -> bool:
    for md_path in (DOCS_DIR / "index.md", DOCS_DIR.parent / "README.md"):
        if md_path.is_file() and "overview.png" in md_path.read_text(encoding="utf-8"):
            return True
    return False


def _check_overview_sync() -> list[str]:
    qs_plot = IMAGES / "quickstart-plot.png"
    overview = IMAGES / "overview.png"
    if not qs_plot.is_file() and not overview.is_file() and not _overview_referenced():
        return []
    if not overview.is_file():
        return ["overview.png missing in docs/images; run with --update"]
    if qs_plot.is_file() and overview.read_bytes() != qs_plot.read_bytes():
        return ["overview.png differs from quickstart-plot.png; run with --update"]
    return []


def update_page(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    out = []
    pos = 0
    for r in run_page(path, save_images=True):
        if r["span"] is None:
            continue
        out.append(text[pos : r["span"][0]])
        out.append("```text\n" + _norm(r["actual"]) + "\n```\n")
        pos = r["span"][1]
    out.append(text[pos:])
    path.write_text("".join(out), encoding="utf-8")


def main(argv: list[str]) -> int:
    pages = sorted(TUTORIALS.glob("*.md"))
    if "--update" in argv:
        for p in pages:
            if pkg := missing_requirement(p):
                print(f"skipped {p.name}: needs {pkg}")
                continue
            update_page(p)
            print(f"updated {p.name}")
        qs_plot = IMAGES / "quickstart-plot.png"
        if qs_plot.is_file():
            IMAGES.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(qs_plot, IMAGES / "overview.png")
            print("synced overview.png <- quickstart-plot.png")
        return 0
    errors = [e for p in pages for e in check_page(p)]
    errors.extend(_check_overview_sync())
    for e in errors:
        print(e, end="\n\n")
    print(f"{len(pages)} pages, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
