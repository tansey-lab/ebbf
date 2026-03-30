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
    parser.add_argument("--output", type=str, required=True, help="Output path for stdata.npz file")
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
    args = parser.parse_args()

    stdata = process_zarr(
        zarr_dir=args.zarr_dir,
        box_width=args.box_width,
        input_format=args.input_format,
    )
    stdata.save(args.output)
    logger.info("Saved STDataset to %s", args.output)
