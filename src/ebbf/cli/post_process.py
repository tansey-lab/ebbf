import argparse

from ebbf.data.tma import TMA
from ebbf.inference.post_process import post_process


def main():
    parser = argparse.ArgumentParser(
        description="Post-process EBB&Flow inference results"
    )
    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input parquet files from inference",
    )
    parser.add_argument("--tma", type=str, required=True, help="Path to tma.npz file")
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
        "--work_dir",
        type=str,
        default=".",
        help="Directory for temporary files (default: current directory)",
    )
    args = parser.parse_args()

    tma = TMA.load(args.tma)
    post_process(
        input_files=args.input_files,
        work_dir=args.work_dir,
        tma=tma,
        cluster_distance=args.cluster_distance,
        output_fname=args.output,
    )
