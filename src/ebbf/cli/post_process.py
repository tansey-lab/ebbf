import argparse

from ebbf.data.st_dataset import STDataset
from ebbf.inference.post_process import post_process


def main():
    parser = argparse.ArgumentParser(description="Post-process EBB&Flow inference results")
    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input parquet files from inference",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to stdata.npz file")
    parser.add_argument(
        "--cluster_distance",
        type=float,
        default=1.5,
        help="Linkage distance for clustering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ef.parquet",
        help="Output file path (default: ef.parquet)",
    )
    parser.add_argument(
        "--anndata_output",
        type=str,
        default="ef.h5ad",
        help="Output AnnData file path (default: ef.h5ad).",
    )
    parser.add_argument(
        "--cd_cutoff",
        type=float,
        default=2.0,
        help="Centroid distance cutoff for AnnData filtering (default: 2.0)",
    )
    parser.add_argument(
        "--original_transcripts",
        type=str,
        default=None,
        help="Path to original transcripts parquet file for distance computation",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Directory for temporary files (default: current directory)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate boundary comparison visualizations",
    )
    parser.add_argument(
        "--xenium_dir",
        type=str,
        default=None,
        help="Path to Xenium output directory for loading default cell boundaries",
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default=None,
        help="Directory for visualization output files (default: work_dir)",
    )
    parser.add_argument(
        "--viz_patch_size",
        type=float,
        default=500.0,
        help="Size of each visualization patch in microns (default: 500)",
    )
    args = parser.parse_args()

    stdata = STDataset.load(args.dataset)
    anndata_fname = args.anndata_output if args.anndata_output else None
    post_process(
        input_files=args.input_files,
        work_dir=args.work_dir,
        dataset=stdata,
        cluster_distance=args.cluster_distance,
        output_fname=args.output,
        anndata_fname=anndata_fname,
        cd_cutoff=args.cd_cutoff,
        original_tx_fname=args.original_transcripts,
        visualize=args.visualize,
        xenium_dir=args.xenium_dir,
        viz_output_dir=args.viz_output_dir,
        viz_patch_size=args.viz_patch_size,
    )
