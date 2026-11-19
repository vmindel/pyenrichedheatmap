"""pyenrichedheatmap (enriched_heatmap): Publication-grade Python EnrichedHeatmap + ComplexHeatmap composition."""

from enriched_heatmap.layout import HeatmapList
from enriched_heatmap.loaders import (
    from_array,
    from_bigwig,
    from_dataframe,
    from_interchange,
    from_pymisha,
)
from enriched_heatmap.matrix import NormalizedMatrix
from enriched_heatmap.ordering import (
    cluster_hierarchical,
    cluster_kmeans,
    compute_order_scores,
)
from enriched_heatmap.profiles import ProfileResult, compute_profile
from enriched_heatmap.raster import (
    compute_target_raster_rows,
    downsample_1d_rows,
    downsample_categorical_rows,
    downsample_rows,
)
from enriched_heatmap.tracks import (
    BarTrack,
    BaseTrack,
    CategoricalTrack,
    EnrichedTrack,
    MatrixTrack,
    PointTrack,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NormalizedMatrix",
    "from_array",
    "from_dataframe",
    "from_interchange",
    "from_bigwig",
    "from_pymisha",
    "BaseTrack",
    "EnrichedTrack",
    "MatrixTrack",
    "CategoricalTrack",
    "BarTrack",
    "PointTrack",
    "HeatmapList",
    "compute_order_scores",
    "cluster_kmeans",
    "cluster_hierarchical",
    "compute_profile",
    "ProfileResult",
    "downsample_rows",
    "downsample_1d_rows",
    "downsample_categorical_rows",
    "compute_target_raster_rows",
]
