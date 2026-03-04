import argparse
import logging
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pyarrow.dataset as ds
import torch
from omegaconf import OmegaConf

from ebbf.data.datasets import TxPatchDataset
from ebbf.data.tma import TMA
from ebbf.inference.infer import FlowInference, infer
from ebbf.model.model import instantiate_model

logger = logging.getLogger(__name__)


def main():
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description="Run EBB&Flow inference")
    parser.add_argument(
        "--model_directory", type=str, required=True, help="Model directory"
    )
    parser.add_argument(
        "--zarr_dir", type=str, required=True, help="Path to SpatialData zarr directory"
    )
    parser.add_argument(
        "--n_workers", type=int, required=True, help="Number of workers"
    )
    parser.add_argument(
        "--worker_id", type=int, required=True, help="Current worker id"
    )
    parser.add_argument(
        "--tma_fname", type=str, default=None, help="TMA filename (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for parquet files (default: current directory)",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing output",
    )
    args = parser.parse_args()

    model_directory: str = args.model_directory
    zarr_dir: str = args.zarr_dir
    n_workers: int = args.n_workers
    worker_id: int = args.worker_id
    tma_fname: str | None = args.tma_fname
    output_dir: str = args.output_dir
    overwrite: bool = bool(args.overwrite)

    # Set up directory and filename
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"part-{str(worker_id).zfill(4)}.parquet")

    # Check to see if filename already exists. Exit if we are not overwriting
    if os.path.isfile(filename) and (not overwrite):
        logger.warning("File exists and we are not overwriting. Exiting now...")
        sys.exit(0)

    config = OmegaConf.load(os.path.join(model_directory, "config.yaml"))

    # Load in the data - resolve parquet path from zarr directory
    points_path = os.path.join(zarr_dir, "points/transcripts/points.parquet")
    if os.path.isdir(points_path):
        tx_path = points_path
    else:
        tx_path = os.path.join(zarr_dir, "points/transcripts")
    tx_dataset = ds.dataset(tx_path, format="parquet")
    tma = TMA.load(tma_fname if tma_fname else config.data.tma_fname)
    curr_bboxes = np.array_split(tma.boxes, n_workers)[worker_id]

    dataset = TxPatchDataset(
        boxes=curr_bboxes,
        tx_dataset=tx_dataset,
        nuc_xy=tma.nuc_xy,
        genes=tma.genes,
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_size,
        overlap=20,  # x_overlap = y_overlap
        max_tx=config.data.max_tx,
        shuffle=True,
        nuc_ellipse_mat=tma.nuc_ellipse_mat,
        nuc_ctype=np.zeros_like(tma.nuc_ctype_idx),
    )

    # Load in the model
    state_dict = torch.load(
        os.path.join(model_directory, "chkpoint.pt"), map_location="cpu"
    )
    model = instantiate_model(config)
    model.train(False)
    model.load_state_dict(state_dict["model"])
    device_str = config.infer.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)
    flow_inference = FlowInference(
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_padding,
        img_resolution=config.model.img_resolution,
        model=model,
        device=device,
        atol=1e-5,
        rtol=1e-5,
    )

    # Run inference
    df = infer(
        dataset=dataset,
        flow_inference=flow_inference,
        batch_size=config.train.batch_size,
    )

    # Save out the data
    df.to_parquet(filename)
