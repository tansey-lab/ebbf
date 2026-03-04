import geopandas as gpd
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ebbf.data.datasets import (
    NuclearPatchDataset,
    nuclear_patch_collate_fn,
)
from ebbf.data.sampling import NoSimTranscriptSampler, TranscriptSampler
from ebbf.data.tma import TMA
from ebbf.data.utils import RTree


def create_nuclear_patch_dataset(config: OmegaConf, train: bool = True):

    tma_fname: str = config.data.tma_fname
    nuc_stats_fname: str = config.data.nuc_stats_fname
    nuc2gene_probs_fname: str = config.data.nuc2gene_probs_fname
    patch_size: float = config.data.patch_size
    patch_padding: float = config.data.patch_padding
    patches_per_punch: int = (
        config.data.train_patches_per_punch
        if train
        else config.data.val_patches_per_punch
    )
    min_nucs: int = config.data.min_nucs

    tma_shapes = gpd.read_parquet(tma_fname)
    nuc_stats = np.load(nuc_stats_fname, allow_pickle=True)
    nuc2gene_probs = load_npz(nuc2gene_probs_fname)
    ctypes = np.unique(nuc_stats["nuc2cell_type"])
    ctypes2ctype_idx = {x: i for i, x in enumerate(ctypes)}
    nuc2ctype = np.array([ctypes2ctype_idx[x] for x in nuc_stats["nuc2cell_type"]])
    punches = (
        tma_shapes.set_index("tma_id").loc[nuc_stats["punch_ids"]]["geometry"].values
    )
    punch_bounds = punches.bounds
    n_punches = punch_bounds.shape[0]

    dataset = NuclearPatchDataset(
        nuc2xy=nuc_stats["nuc2pos"],
        nuc2total_gene_count=nuc_stats["nuc2total_counts"],
        nuc2gene_probs=nuc2gene_probs,
        nuc2ctype=nuc2ctype,
        punch_bounds=punch_bounds,
        patch_size=patch_size,
        patch_padding=patch_padding,
        patches_per_punch=(patches_per_punch * n_punches),
        min_nucs=min_nucs,
    )
    return dataset


def instantiate_train_dataloader(
    config: OmegaConf,
    distributed: bool,
    num_workers: int,
    world_size: int | None = None,
    rank: int | None = None,
):

    tma = TMA.load(config.data.tma_fname)
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
    if config.data.sampler == "empirical-bayes":
        tma = TMA.load(config.data.tma_fname)

        bin_width = float(tma.max_dist / tma.ellipse_hist_array.shape[-1])

        punch_nuc_tx = []
        rtree = RTree(xy=tma.nuc_xy, nuc_total_counts=tma.nuc_total_tx_counts)
        for box in tma.boxes:
            x = rtree.lookup(
                xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3], only_index=False
            )
            punch_nuc_tx.append(x["nuc_total_counts"].sum().item())
        punch_nuc_tx = np.array(punch_nuc_tx)

        ## Centroid hist goes over all the tx
        punch_total_tx = tma.centroid_hist_array[:, -1, :].sum(axis=-1)
        if config.data.corr_mode == "punch":
            correction_factor = punch_total_tx / np.clip(
                punch_nuc_tx, a_min=1.0, a_max=None
            )
        else:
            correction_factor = float(punch_total_tx.sum() / punch_nuc_tx.sum())

        if config.data.sample_mode == "ellipse":
            hist_array = tma.ellipse_hist_array
        else:
            hist_array = tma.centroid_hist_array

        tx_sampler = TranscriptSampler(
            patch_size=config.data.patch_size,
            patch_padding=config.data.patch_padding,
            max_tx=config.data.max_tx,
            min_tx_factor=config.data.min_tx_factor,
            max_tx_factor=config.data.max_tx_factor,
            oversample_tx_factor=config.data.oversample_tx_factor,
            hists=hist_array,
            img_resolution=config.model.img_resolution,
            sample_mode=config.data.sample_mode,
            hist_mode=config.data.hist_mode,
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
            correction_factor=correction_factor,
            bin_width=bin_width,
            device=device,
        )

    elif config.data.sampler == "no-simulation":
        tx_sampler = NoSimTranscriptSampler(
            patch_size=config.data.patch_size,
            patch_padding=config.data.patch_padding,
            max_tx=config.data.max_tx,
            img_resolution=config.model.img_resolution,
            draw_ellipse=(config.model.conditioned is not None),
            max_nuc_dropout_prob=config.data.max_nuc_dropout_prob,
            perturbation_scale=config.data.perturbation_scale,
            device=device,
        )
    else:
        from ebbf.data.sim_sampling import SimTranscriptSampler

        tx_sampler = SimTranscriptSampler(config=config, device=device)

    return tx_sampler
