import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ebbf.data.datasets import nuclear_patch_collate_fn
from ebbf.data.sampling import TranscriptSampler
from ebbf.data.st_dataset import STDataset


def instantiate_train_dataloader(
    config: OmegaConf,
    distributed: bool,
    num_workers: int,
    world_size: int | None = None,
    rank: int | None = None,
):

    tma = STDataset.load(config.data.tma_fname)
    config.n_genes = tma.genes.shape[0]
    config.n_ctypes = 1

    dataset = tma.get_nuc_dataset(
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_padding,
        dataset_size=int(config.data.dataset_size),
        min_nucs=int(config.data.min_nucs),
    )
    collate_fn = nuclear_patch_collate_fn

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        in_order=False,
        num_workers=num_workers,
        multiprocessing_context=("fork" if distributed else None),
        sampler=sampler,
    )
    return dataloader


def instantiate_transcript_sampler(
    config: OmegaConf, device: torch.device | None = None
):
    return TranscriptSampler(
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_padding,
        max_tx=config.data.max_tx,
        min_tx_factor=config.data.min_tx_factor,
        max_tx_factor=config.data.max_tx_factor,
        oversample_tx_factor=config.data.oversample_tx_factor,
        img_resolution=config.model.img_resolution,
        draw_ellipse=(config.model.conditioned is not None),
        max_nuc_dropout_prob=config.data.max_nuc_dropout_prob,
        triangle_dirichlet_scale=(
            config.data.triangle_dirichlet_scale
            if "triangle_dirichlet_scale" in config.data
            else 25.0
        ),
        max_cell_expansion_factor=(
            config.data.max_cell_expansion_factor
            if "max_cell_expansion_factor" in config.data
            else 1.25
        ),
        max_noise_prob=(
            config.data.max_noise_prob if "max_noise_prob" in config.data else 0.1
        ),
        max_noise_scale=(
            config.data.max_noise_scale if "max_noise_scale" in config.data else 2.0
        ),
        device=device,
    )
