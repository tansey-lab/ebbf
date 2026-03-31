import argparse
import logging

import polars as pl

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
        "--transcripts_output",
        type=str,
        default="transcripts.parquet",
        help="Output path for filtered transcripts parquet (default: transcripts.parquet)",
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
    args = parser.parse_args()

    stdata, points_folder, column_mapping = process_zarr(
        zarr_dir=args.zarr_dir,
        box_width=args.box_width,
        input_format=args.input_format,
    )
    stdata.save(args.output)
    logger.info("Saved STDataset to %s", args.output)

    # Write out original transcripts with normalized column names for postprocessing
    x_col = column_mapping["x"]
    y_col = column_mapping["y"]
    df = pl.scan_parquet(points_folder)
    df = df.filter(pl.col("is_gene"))
    df = df.select(
        [
            pl.col("transcript_id"),
            pl.col(x_col).alias("x"),
            pl.col(y_col).alias("y"),
            pl.col("overlaps_nucleus"),
        ]
    )
    df.sink_parquet(args.transcripts_output)
    logger.info("Saved transcripts to %s", args.transcripts_output)
