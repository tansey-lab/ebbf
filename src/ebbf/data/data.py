import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ebbf.config import EBBFConfig
from ebbf.data.datasets import nuclear_patch_collate_fn
from ebbf.data.sampling import TranscriptSampler
from ebbf.data.st_dataset import STDataset


def load_tissue_mask(path: str):
    """Load a tissue-boundary shapefile as a GeoDataFrame.

    Supports GeoParquet (.parquet/.geoparquet) and GeoJSON (.geojson/.json),
    plus anything else geopandas.read_file accepts (e.g. ESRI shapefiles).
    """
    import geopandas as gpd

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".geoparquet"):
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def instantiate_train_dataloader(
    config: EBBFConfig,
    dataset_path: str,
    distributed: bool,
    num_workers: int,
    world_size: int | None = None,
    rank: int | None = None,
    tissue_mask_path: str | None = None,
):

    stdata = STDataset.load(dataset_path)
    config.n_genes = stdata.genes.shape[0]
    config.n_ctypes = 1

    tissue_mask = load_tissue_mask(tissue_mask_path) if tissue_mask_path else None

    dataset = stdata.get_nuc_dataset(
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_padding,
        dataset_size=int(config.data.dataset_size),
        min_nucs=int(config.data.min_nucs),
        tissue_mask=tissue_mask,
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
        num_workers=num_workers,
        multiprocessing_context=("fork" if (distributed and num_workers > 0) else None),
        sampler=sampler,
    )
    return dataloader


def instantiate_transcript_sampler(config: EBBFConfig, device: torch.device | None = None):
    return TranscriptSampler(
        patch_size=config.data.patch_size,
        patch_padding=config.data.patch_padding,
        max_tx=config.data.max_tx,
        min_tx_factor=config.data.min_tx_factor,
        max_tx_factor=config.data.max_tx_factor,
        oversample_tx_factor=config.data.oversample_tx_factor,
        img_resolution=config.model.img_resolution,
        n_genes=config.n_genes,
        draw_ellipse=(config.model.conditioned is not None),
        max_nuc_dropout_prob=config.data.max_nuc_dropout_prob,
        triangle_dirichlet_scale=config.data.triangle_dirichlet_scale,
        max_cell_expansion_factor=config.data.max_cell_expansion_factor,
        max_noise_prob=config.data.max_noise_prob,
        max_noise_scale=config.data.max_noise_scale,
        max_nucleus_swap_prob=config.data.max_nucleus_swap_prob,
        nucleus_swap_radius=config.data.nucleus_swap_radius,
        max_garbage_prob=config.data.max_garbage_prob,
        min_garbage_intercellular_ratio=config.data.min_garbage_intercellular_ratio,
        max_garbage_intercellular_ratio=config.data.max_garbage_intercellular_ratio,
        raster_resolution=config.data.raster_resolution,
        min_uniform_prob=config.data.min_uniform_prob,
        max_uniform_prob=config.data.max_uniform_prob,
        device=device,
    )
