import argparse
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from ebbf.config import EBBFConfig
from ebbf.train_parallel import main_distributed, main_single

logger = logging.getLogger(__name__)


def main():
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description="Train EBB&Flow model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to STDataset file (.npz)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    args = parser.parse_args()

    # Load YAML and merge with structured config schema for validation
    schema = OmegaConf.structured(EBBFConfig)
    yaml_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(schema, yaml_config)

    dataset_path: str = args.dataset
    output_dir: str = args.output_dir
    resume_from: str | None = args.resume_from

    if config.get("prev_model_folder"):
        prev_model_folder = config.prev_model_folder
        uconfig = OmegaConf.load(os.path.join(prev_model_folder, "config.yaml"))
        config = OmegaConf.merge(uconfig, config)
        if resume_from is None:
            resume_from = os.path.join(prev_model_folder, "chkpoint.pt")

    world_size = torch.cuda.device_count()
    if world_size > 1:
        logger.info("Found %d CUDA device(s), using distributed training...", world_size)
        mp.spawn(
            main_distributed,
            args=(world_size, config, dataset_path, output_dir, resume_from),
            nprocs=world_size,
        )
    elif world_size == 1:
        logger.info("Found 1 CUDA device, using single-GPU training...")
        main_single(config, torch.device("cuda:0"), dataset_path, output_dir, resume_from)
    else:
        logger.info("No CUDA devices found, running on CPU...")
        main_single(config, torch.device("cpu"), dataset_path, output_dir, resume_from)
