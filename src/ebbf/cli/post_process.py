import argparse

from ebbf.data.st_dataset import STDataset
from ebbf.inference.post_process import (
    map_clusters_to_cells,
    post_process,
    post_process_merge,
    post_process_prepare,
)


def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", type=str, required=True, help="Path to stdata.npz file")
    parser.add_argument(
        "--output",
        type=str,
        default="ef.parquet",
        help="Output file path (default: ef.parquet)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Directory for temporary files (default: current directory)",
    )


def _add_denovo_filter_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--denovo_min_tx_pctile",
        type=float,
        default=25.0,
        help=(
            "Drop de novo cells with transcript counts below this percentile of"
            " input cell transcript counts (default: 25.0; set 0 to disable)"
        ),
    )
    parser.add_argument(
        "--denovo_min_area_pctile",
        type=float,
        default=5.0,
        help=(
            "Drop de novo cells with convex-hull area below this percentile of"
            " input nucleus ellipse areas (default: 5.0; set 0 to disable)"
        ),
    )
    parser.add_argument(
        "--denovo_filter_report",
        type=str,
        default=None,
        help=(
            "Path to write a per-de-novo-cluster filter report parquet"
            " (columns: ef_cluster_id, n_tx, area, thresholds, pass flags, kept)."
            " Defaults to '<anndata_output stem>.denovo_filter_report.parquet'."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Post-process EBB&Flow inference results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # full
    parser_full = subparsers.add_parser("full", help="Run full post-process pipeline")
    parser_full.add_argument(
        "input_files", type=str, nargs="+", help="Input parquet files from inference"
    )
    _add_common_args(parser_full)
    parser_full.add_argument(
        "--cluster_distance",
        type=float,
        default=1.5,
        help="Linkage distance for clustering",
    )
    parser_full.add_argument(
        "--anndata_output",
        type=str,
        default="ef.h5ad",
        help="Output AnnData file path (default: ef.h5ad).",
    )
    parser_full.add_argument(
        "--cd_cutoff",
        type=float,
        default=2.0,
        help="Centroid distance cutoff for AnnData filtering (default: 2.0)",
    )
    parser_full.add_argument(
        "--original_transcripts",
        type=str,
        default=None,
        help="Path to original transcripts parquet file for distance computation",
    )
    parser_full.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of CPU workers for post-processing (default: auto)",
    )
    _add_denovo_filter_args(parser_full)

    # prepare
    parser_prepare = subparsers.add_parser("prepare", help="Run post-process prepare stage")
    parser_prepare.add_argument(
        "input_files", type=str, nargs="+", help="Input parquet files from inference"
    )
    _add_common_args(parser_prepare)
    parser_prepare.add_argument(
        "--cluster_distance",
        type=float,
        default=1.5,
        help="Linkage distance for clustering",
    )
    parser_prepare.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of CPU workers for prepare stage (default: auto)",
    )
    parser_prepare.add_argument(
        "--noise_meta_output",
        type=str,
        default=None,
        help=(
            "Path to write noise metadata parquet"
            " (enables noise transcript separation before DBSCAN)"
        ),
    )

    # map
    parser_map = subparsers.add_parser(
        "map", help="Run post-process map stage for one worker shard"
    )
    _add_common_args(parser_map)
    parser_map.add_argument(
        "--input_step4", type=str, required=True, help="Step 4 parquet input path"
    )
    parser_map.add_argument("--worker_id", type=int, required=True, help="Worker ID for map stage")
    parser_map.add_argument(
        "--n_workers",
        type=int,
        required=True,
        help="Total number of map workers",
    )

    # merge
    parser_merge = subparsers.add_parser("merge", help="Run post-process merge/finalize stage")
    parser_merge.add_argument(
        "input_files", type=str, nargs="+", help="Mapped parquet files from map workers"
    )
    _add_common_args(parser_merge)
    parser_merge.add_argument(
        "--anndata_output",
        type=str,
        default="ef.h5ad",
        help="Output AnnData file path (default: ef.h5ad).",
    )
    parser_merge.add_argument(
        "--cd_cutoff",
        type=float,
        default=2.0,
        help="Centroid distance cutoff for AnnData filtering (default: 2.0)",
    )
    parser_merge.add_argument(
        "--original_transcripts",
        type=str,
        default=None,
        help="Path to original transcripts parquet file for distance computation",
    )
    parser_merge.add_argument(
        "--noise_meta",
        type=str,
        default=None,
        help=(
            "Path to noise metadata parquet from prepare stage"
            " (re-attaches noise columns and appends noise rows)"
        ),
    )
    _add_denovo_filter_args(parser_merge)

    args = parser.parse_args()

    stdata = STDataset.load(args.dataset)
    if args.command == "full":
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
            n_workers=args.n_workers,
            denovo_min_tx_pctile=args.denovo_min_tx_pctile,
            denovo_min_area_pctile=args.denovo_min_area_pctile,
            denovo_filter_report_fname=args.denovo_filter_report,
        )
    elif args.command == "prepare":
        post_process_prepare(
            input_files=args.input_files,
            work_dir=args.work_dir,
            dataset=stdata,
            cluster_distance=args.cluster_distance,
            output_fname=args.output,
            noise_meta_fname=args.noise_meta_output,
            n_workers=args.n_workers,
        )
    elif args.command == "map":
        map_clusters_to_cells(
            input_path=args.input_step4,
            output_path=args.output,
            nuc_centroid=stdata.nuc_xy,
            nuc_ellipse_mat=stdata.nuc_ellipse_mat,
            nuc_cell_id=stdata.nuc_cell_id,
            worker_id=args.worker_id,
            worker_n=args.n_workers,
        )
    elif args.command == "merge":
        anndata_fname = args.anndata_output if args.anndata_output else None
        post_process_merge(
            mapped_files=args.input_files,
            dataset=stdata,
            output_fname=args.output,
            anndata_fname=anndata_fname,
            cd_cutoff=args.cd_cutoff,
            original_tx_fname=args.original_transcripts,
            noise_meta_fname=args.noise_meta,
            work_dir=args.work_dir,
            denovo_min_tx_pctile=args.denovo_min_tx_pctile,
            denovo_min_area_pctile=args.denovo_min_area_pctile,
            denovo_filter_report_fname=args.denovo_filter_report,
        )
