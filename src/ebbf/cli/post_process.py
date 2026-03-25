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
    parser.add_argument("--tma", type=str, required=True, help="Path to stdata.npz file")
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
        "--work_dir",
        type=str,
        default=".",
        help="Directory for temporary files (default: current directory)",
    )
    args = parser.parse_args()

    tma = STDataset.load(args.tma)
    anndata_fname = args.anndata_output if args.anndata_output else None
    post_process(
        input_files=args.input_files,
        work_dir=args.work_dir,
        tma=tma,
        cluster_distance=args.cluster_distance,
        output_fname=args.output,
        anndata_fname=anndata_fname,
        cd_cutoff=args.cd_cutoff,
    )
