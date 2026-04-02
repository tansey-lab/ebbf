import argparse

from ebbf.inference.reporting import reporting


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML reports comparing EBBF vs default segmentation"
    )
    parser.add_argument(
        "anndata",
        type=str,
        help="Path to EBBF AnnData h5ad file (from post-processing)",
    )
    parser.add_argument(
        "--xenium_dir",
        type=str,
        required=True,
        help="Path to Xenium output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for output report files (default: current directory)",
    )
    parser.add_argument(
        "--ef_tx",
        type=str,
        default=None,
        help="Path to EBBF transcript parquet file (for boundary visualization)",
    )
    parser.add_argument(
        "--original_transcripts",
        type=str,
        default=None,
        help="Path to original transcripts parquet file (for boundary visualization)",
    )
    parser.add_argument(
        "--viz_patch_size",
        type=float,
        default=500.0,
        help="Size of each boundary visualization patch in microns (default: 500)",
    )
    args = parser.parse_args()

    reporting(
        anndata_fname=args.anndata,
        xenium_dir=args.xenium_dir,
        output_dir=args.output_dir,
        ef_tx_fname=args.ef_tx,
        original_tx_fname=args.original_transcripts,
        viz_patch_size=args.viz_patch_size,
    )
