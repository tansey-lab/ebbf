import argparse
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from ebbf.train_parallel import main_distributed, main_single

logger = logging.getLogger(__name__)


def main():
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description="Train EBB&Flow model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config.output_folder)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Override output_folder if provided via CLI
    if args.output_dir is not None:
        config.output_folder = args.output_dir

    if "prev_model_folder" in config:
        prev_model_folder = config.prev_model_folder
        uconfig = OmegaConf.load(os.path.join(prev_model_folder, "config.yaml"))
        uconfig.model_initialization = os.path.join(prev_model_folder, "chkpoint.pt")
        uconfig.data.tma_fname = os.path.join(prev_model_folder, "stdata.npz")
        config = OmegaConf.merge(uconfig, config)

    world_size = torch.cuda.device_count()
    if world_size > 0:
        logger.info(
            "Found %d CUDA device(s), using distributed training...", world_size
        )
        mp.spawn(main_distributed, args=(world_size, config), nprocs=world_size)
    else:
        logger.info("No CUDA devices found, running on CPU...")
        main_single(config, torch.device("cpu"))
