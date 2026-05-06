"""
Visualization module for comparing cell boundaries from different segmentation methods.

Creates smooth density boundaries from EBBF transcript assignments and compares
them with default Xenium cell boundaries.
"""

import logging
import os

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPoint, Polygon
from skimage.measure import find_contours

from ebbf.data.formats import (
    InputFormat,
    detect_input_format,
    has_cell_boundaries,
    load_cell_boundaries,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def smooth_density_region(
    points: np.ndarray,
    grid_size: int = 256,
    bandwidth: float | None = None,
    threshold_rel: float = 0.2,
    spline_smooth: float = 5.0,
    spline_points: int = 300,
    padding: float = 0.08,
) -> Polygon | None:
    """
    Build a smooth enclosing boundary from 2D points via KDE + contour + spline.

    Args:
        points: 2D array of shape (n, 2) containing x, y coordinates
        grid_size: Size of the KDE evaluation grid
        bandwidth: KDE bandwidth (auto-computed if None)
        threshold_rel: Relative threshold for contour detection
        spline_smooth: Spline smoothing parameter
        spline_points: Number of points in the smoothed boundary
        padding: Relative padding around the point bounding box

    Returns:
        A shapely Polygon or None on failure
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    if len(pts) < 3:
        return None

    x, y = pts[:, 0], pts[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)

    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy

    xx = np.linspace(xmin, xmax, grid_size)
    yy = np.linspace(ymin, ymax, grid_size)

    if bandwidth is None:
        scale = 0.5 * (np.std(x) + np.std(y))
        bandwidth = max(scale * 0.2, 1e-6)

    # Bin points into a 2D histogram, then smooth with a Gaussian filter.
    # This is O(n + grid^2) vs O(n * grid^2) for direct KDE evaluation.
    Z, _, _ = np.histogram2d(y, x, bins=[yy, xx])
    # Convert bandwidth from data-space units to grid-cell units
    sigma_x = bandwidth / ((xmax - xmin) / grid_size)
    sigma_y = bandwidth / ((ymax - ymin) / grid_size)
    Z = gaussian_filter(Z.astype(float), sigma=[sigma_y, sigma_x])

    level = threshold_rel * Z.max()
    contours = find_contours(Z, level=level)

    if not contours:
        return None

    contour = max(contours, key=len)
    rows, cols = contour[:, 0], contour[:, 1]
    # Histogram has grid_size-1 bins; map contour indices to bin centers
    bin_centers_x = 0.5 * (xx[:-1] + xx[1:])
    bin_centers_y = 0.5 * (yy[:-1] + yy[1:])
    cx = np.interp(cols, np.arange(grid_size - 1), bin_centers_x)
    cy = np.interp(rows, np.arange(grid_size - 1), bin_centers_y)
    boundary = np.column_stack([cx, cy])

    if np.linalg.norm(boundary[0] - boundary[-1]) > 1e-9:
        boundary = np.vstack([boundary, boundary[0]])

    try:
        tck, _ = splprep([boundary[:, 0], boundary[:, 1]], s=spline_smooth, per=True)
        u_new = np.linspace(0, 1, spline_points, endpoint=False)
        xs, ys = splev(u_new, tck)
        smooth_boundary = np.column_stack([xs, ys])
    except Exception:
        smooth_boundary = boundary

    try:
        poly = Polygon(smooth_boundary)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def build_ebbf_boundaries(
    ef_tx_fname: str,
    original_tx_fname: str,
) -> gpd.GeoDataFrame:
    """
    Build smooth density boundaries per ef_cluster_id from transcript positions.

    Args:
        ef_tx_fname: Path to the EBBF transcript assignment parquet file
        original_tx_fname: Path to original transcripts parquet file with x, y coordinates

    Returns:
        GeoDataFrame with EBBF cell polygons
    """
    import polars as pl

    logger.info("Loading EBBF transcripts for boundary visualization...")
    ef_lf = (
        pl.scan_parquet(ef_tx_fname)
        .select(["transcript_id", "ef_cluster_id"])
        .filter(pl.col("ef_cluster_id").is_not_null())
    )

    logger.info("Joining to original transcript positions...")
    orig_cols = pl.scan_parquet(original_tx_fname).collect_schema().names()
    if "x" in orig_cols and "y" in orig_cols:
        orig_lf = pl.scan_parquet(original_tx_fname).select(
            [
                pl.col("transcript_id"),
                pl.col("x").alias("x_location"),
                pl.col("y").alias("y_location"),
            ]
        )
    elif "x_location" in orig_cols and "y_location" in orig_cols:
        orig_lf = pl.scan_parquet(original_tx_fname).select(
            ["transcript_id", "x_location", "y_location"]
        )
    else:
        raise ValueError(
            f"Could not find coordinate columns in {original_tx_fname}. "
            "Expected 'x', 'y' or 'x_location', 'y_location'."
        )

    tx = ef_lf.join(orig_lf, on="transcript_id", how="left").collect(engine="streaming").to_pandas()

    logger.info(
        "Computing smooth density boundaries for %d clusters...",
        tx["ef_cluster_id"].nunique(),
    )

    grouped = tx.groupby("ef_cluster_id")
    results = []
    n_fallback = 0
    n_failed = 0
    n_smooth = 0

    for cluster_id, group in grouped:
        points = group[["x_location", "y_location"]].values
        n_pts = len(points)

        if n_pts < 3:
            n_failed += 1
            continue

        # For very small clusters, use convex hull as fallback
        if n_pts < 10:
            poly = MultiPoint([(p[0], p[1]) for p in points]).convex_hull
            if poly.geom_type == "Polygon":
                results.append(
                    {
                        "ef_cluster_id": cluster_id,
                        "geometry": poly,
                        "boundary_method": "convex_hull",
                    }
                )
                n_fallback += 1
            else:
                n_failed += 1
            continue

        # Try smooth density boundary
        # Use smaller grid for small clusters
        gs = 64 if n_pts < 50 else (128 if n_pts < 200 else 256)
        poly = smooth_density_region(
            points, grid_size=gs, threshold_rel=0.15, spline_smooth=3.0, spline_points=200
        )

        if poly is not None and poly.geom_type == "Polygon" and poly.area > 0:
            results.append(
                {
                    "ef_cluster_id": cluster_id,
                    "geometry": poly,
                    "boundary_method": "smooth_density",
                }
            )
            n_smooth += 1
        else:
            # Fallback to convex hull
            poly = MultiPoint([(p[0], p[1]) for p in points]).convex_hull
            if poly.geom_type == "Polygon":
                results.append(
                    {
                        "ef_cluster_id": cluster_id,
                        "geometry": poly,
                        "boundary_method": "convex_hull",
                    }
                )
                n_fallback += 1
            else:
                n_failed += 1

        if len(results) % 5000 == 0 and len(results) > 0:
            logger.info("  Processed %d clusters...", len(results))

    logger.info("  Smooth density: %d", n_smooth)
    logger.info("  Convex hull fallback: %d", n_fallback)
    logger.info("  Failed/degenerate: %d", n_failed)

    gdf_ebbf = gpd.GeoDataFrame(results, geometry="geometry")
    gdf_ebbf["segmentation"] = "EBBF"
    gdf_ebbf["ef_cluster_id"] = gdf_ebbf["ef_cluster_id"].astype(str)
    logger.info("  EBBF: %d cell polygons", len(gdf_ebbf))

    return gdf_ebbf


def load_default_boundaries(xenium_dir: str) -> gpd.GeoDataFrame:
    """
    Load default Xenium cell boundaries.

    Supports both Xenium Ranger output directories and SpatialData zarr directories.

    Args:
        xenium_dir: Path to Xenium output directory or SpatialData zarr directory

    Returns:
        GeoDataFrame with default cell polygons
    """
    input_format = detect_input_format(xenium_dir)

    if input_format == InputFormat.UNKNOWN:
        logger.warning(
            "Could not detect input format for %s. "
            "Expected either Xenium Ranger output or SpatialData zarr directory.",
            xenium_dir,
        )
        return gpd.GeoDataFrame()

    if not has_cell_boundaries(xenium_dir, input_format):
        logger.warning("No cell boundaries found in %s", xenium_dir)
        return gpd.GeoDataFrame()

    logger.info(
        "Loading default Xenium boundaries (%s format)...",
        input_format.value,
    )
    gdf_default = load_cell_boundaries(xenium_dir, input_format)
    gdf_default = gdf_default.reset_index(names="cell_id")
    gdf_default["segmentation"] = "Default"
    gdf_default["boundary_method"] = "xenium_default"
    logger.info("  Default: %d cell polygons", len(gdf_default))

    return gdf_default


def _render_patch(
    gdf_default: gpd.GeoDataFrame,
    gdf_ebbf: gpd.GeoDataFrame,
    patch_bounds: tuple,
    output_prefix: str,
    patch_idx: int,
    dpi: int = 150,
):
    """
    Render a single patch comparing default and EBBF boundaries.

    Args:
        gdf_default: GeoDataFrame with default cell polygons
        gdf_ebbf: GeoDataFrame with EBBF cell polygons
        patch_bounds: Tuple of (xmin, ymin, xmax, ymax)
        output_prefix: Output file prefix
        patch_idx: Patch index for filename
        dpi: Output DPI
    """
    xmin, ymin, xmax, ymax = patch_bounds

    # Filter to geometries intersecting this patch
    patch_box = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    gdf_default_patch = gdf_default[gdf_default.intersects(patch_box)]
    gdf_ebbf_patch = gdf_ebbf[gdf_ebbf.intersects(patch_box)]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.invert_yaxis()
        ax.set_aspect("equal")

    # Default boundaries
    if len(gdf_default_patch) > 0:
        gdf_default_patch.plot(
            ax=axes[0], edgecolor="#1f77b4", facecolor="none", linewidth=0.3, alpha=0.8
        )
    axes[0].set_title(f"Default Segmentation (n={len(gdf_default_patch):,})")

    # EBBF boundaries
    if len(gdf_ebbf_patch) > 0:
        gdf_ebbf_patch.plot(
            ax=axes[1], edgecolor="#ff7f0e", facecolor="none", linewidth=0.3, alpha=0.8
        )
    axes[1].set_title(f"EBBF Boundaries (n={len(gdf_ebbf_patch):,})")

    # Overlay
    if len(gdf_default_patch) > 0:
        gdf_default_patch.plot(
            ax=axes[2], edgecolor="#1f77b4", facecolor="none", linewidth=0.3, alpha=0.6
        )
    if len(gdf_ebbf_patch) > 0:
        gdf_ebbf_patch.plot(
            ax=axes[2], edgecolor="#ff7f0e", facecolor="none", linewidth=0.3, alpha=0.6
        )
    axes[2].set_title("Overlay")
    axes[2].legend(["Default", "EBBF"], loc="upper right", fontsize=8)

    plt.tight_layout()
    output_path = f"{output_prefix}_patch_{patch_idx:03d}.pdf"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", output_path)


MAX_PATCHES = 10


def build_combined_boundaries(
    ef_tx_fname: str,
    original_tx_fname: str,
    xenium_dir: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Build a combined GeoDataFrame of EBBF + default Xenium cell boundaries.

    The `segmentation` column distinguishes rows ("EBBF" vs "Default").
    """
    gdf_ebbf = build_ebbf_boundaries(ef_tx_fname, original_tx_fname)
    gdf_default = (
        load_default_boundaries(xenium_dir) if xenium_dir is not None else gpd.GeoDataFrame()
    )

    if len(gdf_default) > 0 and len(gdf_ebbf) > 0:
        gdf_combined = pd.concat([gdf_default, gdf_ebbf], ignore_index=True)
    elif len(gdf_ebbf) > 0:
        gdf_combined = gdf_ebbf
    elif len(gdf_default) > 0:
        gdf_combined = gdf_default
    else:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(gdf_combined, geometry="geometry")


def visualize_boundaries(
    output_dir: str,
    ef_tx_fname: str | None = None,
    original_tx_fname: str | None = None,
    xenium_dir: str | None = None,
    combined_boundaries: gpd.GeoDataFrame | str | None = None,
    patch_size: float = 500.0,
    dpi: int = 150,
):
    """
    Create boundary comparison visualizations.

    Generates separate PDF files for:
    - Full tissue overview: up to MAX_PATCHES randomly-sampled populated tiles
    - Zoomed regions at the center

    Either pass a pre-built `combined_boundaries` (GeoDataFrame or path to a
    cell_boundaries_combined.parquet), or pass the raw inputs to build one.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, "boundaries")

    if combined_boundaries is None:
        if ef_tx_fname is None or original_tx_fname is None:
            raise ValueError(
                "visualize_boundaries needs either combined_boundaries or "
                "(ef_tx_fname, original_tx_fname)"
            )
        gdf_combined = build_combined_boundaries(ef_tx_fname, original_tx_fname, xenium_dir)
    elif isinstance(combined_boundaries, str):
        gdf_combined = gpd.read_parquet(combined_boundaries)
    else:
        gdf_combined = combined_boundaries

    if len(gdf_combined) == 0:
        logger.warning("No boundaries to visualize")
        return

    if "segmentation" in gdf_combined.columns:
        gdf_default = gdf_combined[gdf_combined["segmentation"] == "Default"]
        gdf_ebbf = gdf_combined[gdf_combined["segmentation"] == "EBBF"]
    else:
        gdf_default = gpd.GeoDataFrame()
        gdf_ebbf = gdf_combined

    xmin, ymin, xmax, ymax = gdf_combined.total_bounds
    logger.info("Rendering boundary comparison visualizations...")
    logger.info("  Total bounds: (%.1f, %.1f) to (%.1f, %.1f)", xmin, ymin, xmax, ymax)

    # Render up to MAX_PATCHES randomly-sampled populated patches across the tissue
    logger.info("Rendering up to %d sampled tissue patches...", MAX_PATCHES)
    n_patches_x = int(np.ceil((xmax - xmin) / patch_size))
    n_patches_y = int(np.ceil((ymax - ymin) / patch_size))

    grid = [(i, j) for i in range(n_patches_x) for j in range(n_patches_y)]
    rng = np.random.default_rng(0)
    rng.shuffle(grid)

    sindex_ebbf = gdf_ebbf.sindex if len(gdf_ebbf) > 0 else None
    sindex_default = gdf_default.sindex if len(gdf_default) > 0 else None

    rendered = 0
    for i, j in grid:
        if rendered >= MAX_PATCHES:
            break
        px_min = xmin + i * patch_size
        py_min = ymin + j * patch_size
        px_max = min(px_min + patch_size, xmax)
        py_max = min(py_min + patch_size, ymax)
        bbox = (px_min, py_min, px_max, py_max)

        has_content = (
            sindex_ebbf is not None and len(list(sindex_ebbf.intersection(bbox))) > 0
        ) or (sindex_default is not None and len(list(sindex_default.intersection(bbox))) > 0)
        if not has_content:
            continue

        _render_patch(gdf_default, gdf_ebbf, bbox, output_prefix, rendered, dpi=dpi)
        rendered += 1

    # Render zoomed center regions
    logger.info("Rendering zoomed center regions...")
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    zoom_size = 200.0  # 200 micron zoom windows

    zoom_regions = [
        ("center", cx - zoom_size / 2, cy - zoom_size / 2, cx + zoom_size / 2, cy + zoom_size / 2),
        (
            "center_left",
            cx - zoom_size * 1.5,
            cy - zoom_size / 2,
            cx - zoom_size / 2,
            cy + zoom_size / 2,
        ),
        (
            "center_right",
            cx + zoom_size / 2,
            cy - zoom_size / 2,
            cx + zoom_size * 1.5,
            cy + zoom_size / 2,
        ),
        (
            "center_top",
            cx - zoom_size / 2,
            cy - zoom_size * 1.5,
            cx + zoom_size / 2,
            cy - zoom_size / 2,
        ),
        (
            "center_bottom",
            cx - zoom_size / 2,
            cy + zoom_size / 2,
            cx + zoom_size / 2,
            cy + zoom_size * 1.5,
        ),
    ]

    for name, rx0, ry0, rx1, ry1 in zoom_regions:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax in axes:
            ax.set_xlim(rx0, rx1)
            ax.set_ylim(ry0, ry1)
            ax.invert_yaxis()
            ax.set_aspect("equal")

        # Filter to geometries in this region
        region_box = Polygon([(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)])
        gdf_default_region = gdf_default[gdf_default.intersects(region_box)]
        gdf_ebbf_region = gdf_ebbf[gdf_ebbf.intersects(region_box)]

        if len(gdf_default_region) > 0:
            gdf_default_region.plot(
                ax=axes[0], edgecolor="#1f77b4", facecolor="none", linewidth=0.5, alpha=0.8
            )
        axes[0].set_title(f"Default - {name}")

        if len(gdf_ebbf_region) > 0:
            gdf_ebbf_region.plot(
                ax=axes[1], edgecolor="#ff7f0e", facecolor="none", linewidth=0.5, alpha=0.8
            )
        axes[1].set_title(f"EBBF - {name}")

        if len(gdf_default_region) > 0:
            gdf_default_region.plot(
                ax=axes[2], edgecolor="#1f77b4", facecolor="none", linewidth=0.5, alpha=0.6
            )
        if len(gdf_ebbf_region) > 0:
            gdf_ebbf_region.plot(
                ax=axes[2], edgecolor="#ff7f0e", facecolor="none", linewidth=0.5, alpha=0.6
            )
        axes[2].set_title(f"Overlay - {name}")
        axes[2].legend(["Default", "EBBF"], loc="upper right", fontsize=8)

        plt.tight_layout()
        output_path = f"{output_prefix}_zoom_{name}.pdf"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", output_path)

    logger.info("Visualization complete!")
