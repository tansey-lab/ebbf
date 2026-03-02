import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Run EBB&Flow inference")
    parser.add_argument(
        "--model_directory", type=str, required=True, help="Model directory"
    )
    parser.add_argument(
        "--transcripts_fname", type=str, required=True, help="Transcripts filename"
    )
    parser.add_argument(
        "--n_workers", type=int, required=True, help="Number of workers"
    )
    parser.add_argument(
        "--worker_id", type=int, required=True, help="Current worker id"
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing output",
    )
    args = parser.parse_args()

    model_directory: str = args.model_directory
    transcripts_fname: str = args.transcripts_fname
    n_workers: int = args.n_workers
    worker_id: int = args.worker_id
    overwrite: bool = bool(args.overwrite)

    # Set up directory and filename
    outdir = os.path.join(model_directory, "tx")
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"part-{str(worker_id).zfill(4)}.parquet")

    # Check to see if filename already exists. Exit if we are not overwriting
    if os.path.isfile(filename) and (not overwrite):
        print("File exists and we are not overwriting. Exiting now...", file=sys.stderr)
        sys.exit(0)

    config = OmegaConf.load(os.path.join(model_directory, "config.yaml"))

    # Load in the data
    tx_dataset = ds.dataset(transcripts_fname, format="parquet")
    tma = TMA.load(config.data.tma_fname)
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
    device = torch.device("cuda")
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
    n_processes = len(os.sched_getaffinity(0))
    df = infer(
        dataset=dataset,
        flow_inference=flow_inference,
        num_workers=n_processes,
        batch_size=config.train.batch_size,
    )

    # Save out the data
    df.to_parquet(filename)
