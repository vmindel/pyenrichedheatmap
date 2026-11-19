"""Enriched heatmaps from pandas DataFrames, drawn on matplotlib axes.

The data path is plain tables: :func:`bins` makes one row per bin, pymisha's
``gextract`` (or :func:`read_bigwig`) fills a value per bin, :func:`matrices` pivots
the result. Drawing functions have the shape ``fn(matrix, ax, rows, **style)``, where
``rows`` (from :func:`arrange`) is the one object that keeps every panel aligned.
"""

from enriched_heatmap.bigwig import read_bigwig
from enriched_heatmap.bins import bins, matrices, matrix
from enriched_heatmap.datasets import example_genome, example_peaks, example_values
from enriched_heatmap.draw import (
    annotation,
    colorbar,
    group_labels,
    heatmap,
    legend,
    limits,
    palette,
    profile,
)
from enriched_heatmap.layout import Grid, figure, plot
from enriched_heatmap.matrix import NormalizedMatrix
from enriched_heatmap.ordering import hclust, kmeans, score
from enriched_heatmap.rows import Rows, arrange

__version__ = "0.2.0"

__all__ = [
    "Grid",
    "NormalizedMatrix",
    "Rows",
    "annotation",
    "arrange",
    "bins",
    "colorbar",
    "example_genome",
    "example_peaks",
    "example_values",
    "figure",
    "group_labels",
    "hclust",
    "heatmap",
    "kmeans",
    "legend",
    "limits",
    "matrices",
    "matrix",
    "palette",
    "plot",
    "profile",
    "read_bigwig",
    "score",
]
