import argparse
import os

import pandas as pd
import polars as pl

from ebbf.utils import compute_polygons
from ebbf.data.tma import process_zarr


def main():
    parser = argparse.ArgumentParser(description="Preprocess spatial transcriptomics data for EBB&Flow")
    parser.add_argument('--zarr_dir', type=str, required=True,
                        help="Path to SpatialData zarr directory")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path for tma.npz file")
    parser.add_argument('--box_width', type=float, default=400,
                        help="Box width for spatial sampling (default: 400)")
    parser.add_argument('--max_dist', type=float, default=40,
                        help="Maximum distance for histogram binning (default: 40)")
    parser.add_argument('--n_bins', type=int, default=100,
                        help="Number of histogram bins (default: 100)")
    args = parser.parse_args()

    tma = process_zarr(
        zarr_dir=args.zarr_dir,
        max_dist=args.max_dist,
        n_bins=args.n_bins,
        box_width=args.box_width)
    tma.save(args.output)
    print(f"Saved TMA to {args.output}")
