import argparse
import logging
from pathlib import Path

from ebbf.inference.visualization import build_combined_boundaries


def main():
    parser = argparse.ArgumentParser(
        description="Build the combined EBBF + default cell_boundaries_combined.parquet"
    )
    parser.add_argument("--ef_tx", type=str, required=True)
    parser.add_argument("--original_transcripts", type=str, required=True)
    parser.add_argument("--xenium_dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="cell_boundaries_combined.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    )

    gdf = build_combined_boundaries(
        ef_tx_fname=args.ef_tx,
        original_tx_fname=args.original_transcripts,
        xenium_dir=args.xenium_dir,
    )
    if len(gdf) == 0:
        raise SystemExit("No boundaries built; nothing to write")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out)
    logging.getLogger(__name__).info("Wrote %s (%d polygons)", out, len(gdf))
