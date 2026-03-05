import argparse
import logging

from ebbf.data.st_dataset import process_zarr

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess spatial transcriptomics data for EBB&Flow"
    )
    parser.add_argument(
        "--zarr_dir",
        type=str,
        required=True,
        help="Path to input directory (SpatialData zarr or Xenium output)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for stdata.npz file"
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["spatialdata", "xenium"],
        default=None,
        help="Input format: 'spatialdata' or 'xenium'. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--box_width",
        type=float,
        default=400,
        help="Box width for spatial sampling (default: 400)",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=40,
        help="Maximum distance for histogram binning (default: 40)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=100,
        help="Number of histogram bins (default: 100)",
    )
    args = parser.parse_args()

    tma = process_zarr(
        zarr_dir=args.zarr_dir,
        max_dist=args.max_dist,
        n_bins=args.n_bins,
        box_width=args.box_width,
        input_format=args.input_format,
    )
    tma.save(args.output)
    logger.info("Saved STDataset to %s", args.output)
