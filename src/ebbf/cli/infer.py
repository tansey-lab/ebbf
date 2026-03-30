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
from ebbf.data.st_dataset import STDataset, detect_input_format
from ebbf.inference.infer import FlowInference, infer
from ebbf.model.model import instantiate_model

logger = logging.getLogger(__name__)


def main():
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description="Run EBB&Flow inference")
    parser.add_argument("--model_directory", type=str, required=True, help="Model directory")
    parser.add_argument(
        "--zarr_dir",
        type=str,
        required=True,
        help="Path to input directory (SpatialData zarr or Xenium output)",
    )
    parser.add_argument("--n_workers", type=int, required=True, help="Number of workers")
    parser.add_argument("--worker_id", type=int, required=True, help="Current worker id")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to STDataset file (.npz)",
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
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["spatialdata", "xenium"],
        default=None,
        help="Input format: 'spatialdata' or 'xenium'. Auto-detected if not specified.",
    )
    args = parser.parse_args()

    model_directory: str = args.model_directory
    zarr_dir: str = args.zarr_dir
    n_workers: int = args.n_workers
    worker_id: int = args.worker_id
    dataset: str = args.dataset
    output_dir: str = args.output_dir
    overwrite: bool = bool(args.overwrite)
    input_format: str | None = args.input_format

    # Set up directory and filename
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"part-{str(worker_id).zfill(4)}.parquet")

    # Check to see if filename already exists. Exit if we are not overwriting
    if os.path.isfile(filename) and (not overwrite):
        logger.warning("File exists and we are not overwriting. Exiting now...")
        sys.exit(0)

    config = OmegaConf.load(os.path.join(model_directory, "config.yaml"))

    # Detect or use provided input format
    if input_format is None:
        input_format = detect_input_format(zarr_dir)
        logger.info("Detected input format: %s", input_format)

    # Load in the data - resolve parquet path based on input format
    if input_format == "xenium":
        tx_path = os.path.join(zarr_dir, "transcripts.parquet")
    else:
        points_path = os.path.join(zarr_dir, "points/transcripts/points.parquet")
        if os.path.isdir(points_path):
            tx_path = points_path
        else:
            tx_path = os.path.join(zarr_dir, "points/transcripts")
    tx_dataset = ds.dataset(tx_path, format="parquet")
    stdata = STDataset.load(dataset)
    curr_bboxes = np.array_split(stdata.boxes, n_workers)[worker_id]

    # Set column names based on input format
    if input_format == "xenium":
        x_col, y_col = "x_location", "y_location"
    else:
        x_col, y_col = "x", "y"

    dataset = TxPatchDataset(
        boxes=curr_bboxes,
        tx_dataset=tx_dataset,
        nuc_xy=stdata.nuc_xy,
        genes=stdata.genes,
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_size,
        overlap=20,  # x_overlap = y_overlap
        max_tx=config.data.max_tx,
        shuffle=True,
        nuc_ellipse_mat=stdata.nuc_ellipse_mat,
        nuc_ctype=np.zeros_like(stdata.nuc_ctype_idx),
        x_col=x_col,
        y_col=y_col,
    )

    # Load in the model
    state_dict = torch.load(os.path.join(model_directory, "chkpoint.pt"), map_location="cpu")
    model = instantiate_model(config)
    model.train(False)
    model.load_state_dict(state_dict["model"])
    model.float()
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
