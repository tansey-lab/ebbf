"""
Input format detection and loading utilities for Xenium spatial transcriptomics data.

Supports two input formats:
1. Xenium Ranger output directory (direct from 10x Genomics instrument)
2. SpatialData zarr directory (converted via spatialdata-io)
"""

import os
from enum import Enum

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


class InputFormat(Enum):
    """Supported input formats for Xenium data."""

    XENIUM_RANGER = "xenium"
    SPATIALDATA = "spatialdata"
    UNKNOWN = "unknown"


def detect_input_format(input_dir: str) -> InputFormat:
    """
    Detect the input format of a directory.

    Args:
        input_dir: Path to input directory

    Returns:
        InputFormat enum indicating the detected format
    """
    # Check for Xenium Ranger format
    xenium_markers = [
        "transcripts.parquet",
        "nucleus_boundaries.parquet",
    ]
    has_xenium = all(os.path.exists(os.path.join(input_dir, f)) for f in xenium_markers)

    if has_xenium:
        return InputFormat.XENIUM_RANGER

    # Check for SpatialData format
    spatialdata_markers = [
        "points/transcripts",
        "shapes/nucleus_boundaries",
    ]
    has_spatialdata = all(os.path.exists(os.path.join(input_dir, p)) for p in spatialdata_markers)

    if has_spatialdata:
        return InputFormat.SPATIALDATA

    return InputFormat.UNKNOWN


def has_cell_boundaries(input_dir: str, input_format: InputFormat | None = None) -> bool:
    """
    Check if the input directory contains cell boundary data.

    Args:
        input_dir: Path to input directory
        input_format: Optional pre-detected format (auto-detected if None)

    Returns:
        True if cell boundaries are available
    """
    if input_format is None:
        input_format = detect_input_format(input_dir)

    if input_format == InputFormat.XENIUM_RANGER:
        return os.path.exists(os.path.join(input_dir, "cell_boundaries.parquet"))
    elif input_format == InputFormat.SPATIALDATA:
        return os.path.exists(os.path.join(input_dir, "shapes/cell_boundaries/shapes.parquet"))
    return False


def load_cell_boundaries(
    input_dir: str, input_format: InputFormat | None = None
) -> gpd.GeoDataFrame:
    """
    Load cell boundaries from either Xenium Ranger or SpatialData format.

    Args:
        input_dir: Path to input directory
        input_format: Optional pre-detected format (auto-detected if None)

    Returns:
        GeoDataFrame with cell polygons indexed by cell_id

    Raises:
        FileNotFoundError: If no cell boundaries found
        ValueError: If format is unknown
    """
    if input_format is None:
        input_format = detect_input_format(input_dir)

    if input_format == InputFormat.XENIUM_RANGER:
        return _load_xenium_cell_boundaries(input_dir)
    elif input_format == InputFormat.SPATIALDATA:
        return _load_spatialdata_cell_boundaries(input_dir)
    else:
        raise ValueError(
            f"Unknown input format for '{input_dir}'. "
            "Expected either Xenium Ranger output or SpatialData zarr directory."
        )


def load_nucleus_boundaries(
    input_dir: str, input_format: InputFormat | None = None
) -> gpd.GeoDataFrame:
    """
    Load nucleus boundaries from either Xenium Ranger or SpatialData format.

    Args:
        input_dir: Path to input directory
        input_format: Optional pre-detected format (auto-detected if None)

    Returns:
        GeoDataFrame with nucleus polygons indexed by cell_id

    Raises:
        FileNotFoundError: If no nucleus boundaries found
        ValueError: If format is unknown
    """
    if input_format is None:
        input_format = detect_input_format(input_dir)

    if input_format == InputFormat.XENIUM_RANGER:
        return _load_xenium_nucleus_boundaries(input_dir)
    elif input_format == InputFormat.SPATIALDATA:
        return _load_spatialdata_nucleus_boundaries(input_dir)
    else:
        raise ValueError(
            f"Unknown input format for '{input_dir}'. "
            "Expected either Xenium Ranger output or SpatialData zarr directory."
        )


def get_transcripts_path(input_dir: str, input_format: InputFormat | None = None) -> str:
    """
    Get the path to the transcripts parquet file/directory.

    Args:
        input_dir: Path to input directory
        input_format: Optional pre-detected format (auto-detected if None)

    Returns:
        Path to transcripts parquet

    Raises:
        ValueError: If format is unknown
    """
    if input_format is None:
        input_format = detect_input_format(input_dir)

    if input_format == InputFormat.XENIUM_RANGER:
        return os.path.join(input_dir, "transcripts.parquet")
    elif input_format == InputFormat.SPATIALDATA:
        points_path = os.path.join(input_dir, "points/transcripts/points.parquet")
        if os.path.isdir(points_path):
            return points_path
        return os.path.join(input_dir, "points/transcripts")
    else:
        raise ValueError(
            f"Unknown input format for '{input_dir}'. "
            "Expected either Xenium Ranger output or SpatialData zarr directory."
        )


def get_coordinate_columns(input_format: InputFormat) -> dict[str, str]:
    """
    Get the column name mapping for coordinates.

    Args:
        input_format: The input format

    Returns:
        Dict mapping standard names to format-specific names
    """
    if input_format == InputFormat.XENIUM_RANGER:
        return {"x": "x_location", "y": "y_location"}
    else:
        return {"x": "x", "y": "y"}


def _vertices_to_polygons(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert Xenium vertex-based boundary format to GeoDataFrame with Shapely polygons.

    Args:
        df: DataFrame with columns ['cell_id', 'vertex_x', 'vertex_y']

    Returns:
        GeoDataFrame with cell_id as index and geometry column containing polygons
    """
    polygons = []
    cell_ids = []

    for cell_id, group in df.groupby("cell_id"):
        coords = list(zip(group["vertex_x"], group["vertex_y"]))
        if len(coords) >= 3:
            # Close the polygon if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygons.append(Polygon(coords))
            cell_ids.append(cell_id)

    gdf = gpd.GeoDataFrame({"geometry": polygons}, index=cell_ids)
    gdf.index.name = "cell_id"
    return gdf


def _load_xenium_cell_boundaries(xenium_dir: str) -> gpd.GeoDataFrame:
    """Load cell boundaries from Xenium Ranger output directory."""
    path = os.path.join(xenium_dir, "cell_boundaries.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cell boundaries not found at {path}")

    df = pd.read_parquet(path)
    return _vertices_to_polygons(df)


def _load_xenium_nucleus_boundaries(xenium_dir: str) -> gpd.GeoDataFrame:
    """Load nucleus boundaries from Xenium Ranger output directory."""
    path = os.path.join(xenium_dir, "nucleus_boundaries.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nucleus boundaries not found at {path}")

    df = pd.read_parquet(path)
    return _vertices_to_polygons(df)


def _load_spatialdata_cell_boundaries(zarr_dir: str) -> gpd.GeoDataFrame:
    """Load cell boundaries from SpatialData zarr directory."""
    path = os.path.join(zarr_dir, "shapes/cell_boundaries/shapes.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cell boundaries not found at {path}")

    gdf = gpd.read_parquet(path)
    return gdf


def _load_spatialdata_nucleus_boundaries(zarr_dir: str) -> gpd.GeoDataFrame:
    """Load nucleus boundaries from SpatialData zarr directory."""
    path = os.path.join(zarr_dir, "shapes/nucleus_boundaries/shapes.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nucleus boundaries not found at {path}")

    gdf = gpd.read_parquet(path)
    return gdf
