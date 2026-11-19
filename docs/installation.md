# Installation

Python 3.10 or later. Core dependencies: numpy, pandas, matplotlib.

```bash
pip install "git+https://github.com/vmindel/pyenrichedheatmap"
```

Optional extras:

```bash
pip install "pyenrichedheatmap[stats] @ git+https://github.com/vmindel/pyenrichedheatmap"    # scipy, scikit-learn
pip install "pyenrichedheatmap[bigwig] @ git+https://github.com/vmindel/pyenrichedheatmap"   # pyBigWig
pip install "pyenrichedheatmap[pymisha] @ git+https://github.com/vmindel/pyenrichedheatmap"  # pymisha
```

- `stats`: `eh.hclust` needs scipy. `eh.kmeans` uses scikit-learn when installed and a
  NumPy implementation otherwise; `mat.smooth` likewise falls back to NumPy.
- `bigwig`: needed only for `eh.read_bigwig`.
- `pymisha`: the package never imports pymisha itself; you call `gextract`. The extra
  is a convenience for installing both.

## Development

```bash
git clone https://github.com/vmindel/pyenrichedheatmap
cd pyenrichedheatmap
pip install -e ".[dev]"
pytest                              # includes running every tutorial
python docs/run_docs.py --update    # refresh tutorial output and figures
mkdocs serve
```

The pymisha tutorial runs against pymisha's bundled example database and is skipped
when pymisha is not installed.

## Claude Code skill

```bash
pyenrichedheatmap-install-skills
```

See [Claude Code skill](claude-skill.md).
