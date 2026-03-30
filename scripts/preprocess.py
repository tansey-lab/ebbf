"""
Legacy script for converting Xenium data to SpatialData zarr format and processing.

For direct zarr processing, use the CLI:
    ebbf-preprocess --zarr_dir /path/to/data.zarr --output /path/to/stdata.npz
"""

import os

import pandas as pd
import polars as pl

from ebbf.data.st_dataset import process_zarr
from ebbf.utils import compute_polygons

if __name__ == "__main__":
    cancer = "coad"

    output_folder = f"/data1/tanseyw/projects/tosh/nuc_diffusion/data/spatch/{cancer}"
    zarr_dir = os.path.join(output_folder, "spatialdata.zarr")
    os.makedirs(output_folder, exist_ok=True)

    xenium_folder = f"/data1/tanseyw/projects/spatial_data/spatch/{cancer}/xenium/"

    # Create spatialdata.zarr structure
    tx_folder = os.path.join(zarr_dir, "points/transcripts/")
    nucs_folder = os.path.join(zarr_dir, "shapes/nucleus_boundaries")
    cell_folder = os.path.join(zarr_dir, "shapes/cell_boundaries")

    os.makedirs(tx_folder, exist_ok=True)
    os.makedirs(nucs_folder, exist_ok=True)
    os.makedirs(cell_folder, exist_ok=True)

    points_fname = os.path.join(tx_folder, "points.parquet")
    nucs_fname = os.path.join(nucs_folder, "shapes.parquet")
    cell_fname = os.path.join(cell_folder, "shapes.parquet")

    # Convert nucleus boundaries
    ndf = pd.read_csv(os.path.join(xenium_folder, "segmentation_mask/nucleus_boundaries.csv"))
    nuc_df = compute_polygons(ndf)
    nuc_df.to_parquet(nucs_fname)

    # Convert cell boundaries
    cdf = pd.read_csv(os.path.join(xenium_folder, "segmentation_mask/cell_boundaries.csv"))
    cell_df = compute_polygons(cdf)
    cell_df.to_parquet(cell_fname)

    # Convert transcripts
    tx_df = pl.scan_parquet(os.path.join(xenium_folder, "transcripts/transcripts.parquet"))
    tx_df = tx_df.rename({"x_location": "x", "y_location": "y", "z_location": "z"})
    tx_df.sink_parquet(points_fname)

    # Process the zarr directory
    stdata = process_zarr(zarr_dir=zarr_dir)
    stdata.save(os.path.join(output_folder, "stdata.npz"))
