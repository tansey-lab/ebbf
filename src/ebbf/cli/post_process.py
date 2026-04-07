import argparse

from ebbf.data.st_dataset import STDataset
from ebbf.inference.post_process import (
    map_clusters_to_cells,
    post_process,
    post_process_merge,
    post_process_prepare,
)


def main():
    parser = argparse.ArgumentParser(description="Post-process EBB&Flow inference results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "prepare", "map", "merge"],
        default="full",
        help="Execution mode: full pipeline, prepare-only, map worker, or merge",
    )
    parser.add_argument(
        "input_files",
        type=str,
        nargs="*",
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
        "--n_workers",
        type=int,
        default=None,
        help="Number of CPU workers for post-processing (default: auto)",
    )
    parser.add_argument("--input_step4", type=str, default=None, help="Step 4 parquet input path")
    parser.add_argument(
        "--worker_id",
        type=int,
        default=None,
        help="Worker ID for map mode (0-indexed)",
    )
    args = parser.parse_args()

    stdata = STDataset.load(args.dataset)
    anndata_fname = args.anndata_output if args.anndata_output else None
    if args.mode == "full":
        post_process(
            input_files=args.input_files,
            work_dir=args.work_dir,
            dataset=stdata,
            cluster_distance=args.cluster_distance,
            output_fname=args.output,
            anndata_fname=anndata_fname,
            cd_cutoff=args.cd_cutoff,
            original_tx_fname=args.original_transcripts,
            n_workers=args.n_workers,
        )
    elif args.mode == "prepare":
        post_process_prepare(
            input_files=args.input_files,
            work_dir=args.work_dir,
            dataset=stdata,
            cluster_distance=args.cluster_distance,
            output_fname=args.output,
            n_workers=args.n_workers,
        )
    elif args.mode == "map":
        if args.input_step4 is None:
            raise ValueError("--input_step4 is required for --mode map")
        if args.worker_id is None or args.n_workers is None:
            raise ValueError("--worker_id and --n_workers are required for --mode map")
        map_clusters_to_cells(
            input_path=args.input_step4,
            output_path=args.output,
            nuc_centroid=stdata.nuc_xy,
            nuc_ellipse_mat=stdata.nuc_ellipse_mat,
            nuc_cell_id=stdata.nuc_cell_id,
            n_workers=1,
            worker_id=args.worker_id,
            worker_n=args.n_workers,
        )
    elif args.mode == "merge":
        post_process_merge(
            mapped_files=args.input_files,
            dataset=stdata,
            output_fname=args.output,
            anndata_fname=anndata_fname,
            cd_cutoff=args.cd_cutoff,
            original_tx_fname=args.original_transcripts,
            work_dir=args.work_dir,
        )
