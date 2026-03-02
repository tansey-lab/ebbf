import argparse
import os

from ebbf.data.tma import TMA
from ebbf.inference.post_process import post_process


def main():
    parser = argparse.ArgumentParser(description="Post-process EBB&Flow inference results")
    parser.add_argument('--model_directory', type=str, required=True, help="Model directory")
    parser.add_argument('--tma_directory', type=str, required=True, help="TMA directory")
    parser.add_argument('--cluster_distance', type=float, default=1.5, help='Linkage distance for clustering')
    args = parser.parse_args()

    tma = TMA.load(os.path.join(args.tma_directory, "tma.npz"))
    post_process(directory=args.model_directory, tma=tma, cluster_distance=args.cluster_distance)
