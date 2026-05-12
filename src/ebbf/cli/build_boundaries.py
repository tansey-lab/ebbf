import argparse
import logging
from pathlib import Path

from ebbf.inference.visualization import build_ebbf_boundaries_to_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Build EBBF cell boundaries (cell_boundaries_ebbf.parquet)"
    )
    parser.add_argument("--ef_tx", type=str, required=True)
    parser.add_argument("--original_transcripts", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="cell_boundaries_ebbf.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of ef clusters to process per chunk",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    )

    out = Path(args.output)
    n = build_ebbf_boundaries_to_parquet(
        ef_tx_fname=args.ef_tx,
        original_tx_fname=args.original_transcripts,
        output_path=out,
        chunk_size=args.chunk_size,
    )
    if n == 0:
        raise SystemExit("No boundaries built; nothing to write")
