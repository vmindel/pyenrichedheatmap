# Installation

`pyenrichedheatmap` requires **Python 3.10+** and keeps its core footprint light
(`numpy`, `pandas`, `matplotlib`).

## Basic install

```bash
pip install pyenrichedheatmap
```

Or directly from GitHub:

```bash
pip install "git+https://github.com/vmindel/pyenrichedheatmap.git"
```

## Optional extras

Install only the optional integrations you need:

```bash
# Clustering (k-means, hierarchical) and SciPy Gaussian smoothing
pip install "pyenrichedheatmap[stats]"

# BigWig loading via pyBigWig
pip install "pyenrichedheatmap[bigwig]"

# pymisha database and virtual track loading
pip install "pyenrichedheatmap[pymisha]"
```

Combine extras as needed:

```bash
pip install "pyenrichedheatmap[stats,bigwig]"
```

## Development & documentation

```bash
git clone https://github.com/vmindel/pyenrichedheatmap.git
cd pyenrichedheatmap
pip install -e ".[dev]"
pytest
mkdocs serve
```

## Verify your install

```python
import enriched_heatmap as eh

print(eh.__version__)
```

## Installing the Claude Code skill

If you use Claude Code, install the bundled Agent Skill once after installing the package:

```bash
pyenrichedheatmap-install-skills
```

See [Claude Code skill](claude-skill.md) for all installer options.
