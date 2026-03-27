import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ChainDataset, DataLoader
from torchdiffeq import odeint
from tqdm import tqdm

from ebbf.data.datasets import TxPatchDataset, tx_patch_collate_fn
from ebbf.data.utils import ellipse_grid_indices

logger = logging.getLogger(__name__)


class FlowInference:
    def __init__(
        self,
        patch_size: float,
        patch_padding: float,
        img_resolution: int,
        model: torch.nn.Module,
        device: torch.device,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):

        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.img_resolution = img_resolution
        self.model = model.to(device)
        self.model.train(False)

        self.device = device
        self.atol = atol
        self.rtol = rtol
        self.L = (self.patch_size + 2 * self.patch_padding) / self.patch_size

    def flow(self, batch: dict[str, torch.Tensor], n_intermediates: int = 0):
        with torch.no_grad():
            batch = {key: val.to(self.device) for key, val in batch.items()}
            patch_center = torch.repeat_interleave(
                batch["patch_center"], repeats=batch["tx_seqlens"], dim=0
            )

            if ("nuc_ellipse_mat" in batch) and ("nuc_xy" in batch):
                ellipse_coordinates, ellipse_seqlens = ellipse_grid_indices(
                    S=batch["nuc_ellipse_mat"],
                    c=batch["nuc_xy"],
                    L=self.L,
                    M=self.img_resolution,
                )
                batch["nuc_ellipse_coordinates"] = ellipse_coordinates
                batch["nuc_ellipse_seqlens"] = ellipse_seqlens

            def ode_func(t, tx_pos_t: torch.Tensor):
                pred = self.model(
                    tx_pos=tx_pos_t,
                    gene_idx=batch["tx_gene_idx"],
                    t=torch.atleast_1d(t),
                    tx_seqlens=batch["tx_seqlens"],
                    nuc_seqlens=(batch["nuc_seqlens"] if "nuc_seqlens" in batch else None),
                    nuc_ellipse_pos=(
                        batch["nuc_ellipse_coordinates"]
                        if "nuc_ellipse_coordinates" in batch
                        else None
                    ),
                    nuc_ellipse_seqlens=(
                        batch["nuc_ellipse_seqlens"] if "nuc_ellipse_seqlens" in batch else None
                    ),
                    nuc_ctypes=(batch["nuc_ctype"] if "nuc_ctype" in batch else None),
                )
                return pred

            time_grid = torch.linspace(
                start=0.0, end=1.0, steps=(n_intermediates + 2), device=self.device
            )

            x = odeint(
                ode_func,
                batch["tx_xy"],
                time_grid,
                atol=self.atol,
                rtol=self.rtol,
                options={"dtype": torch.float32},
            )

            ## Scale back to the original coordinates
            tx_pos: torch.Tensor = (0.5 * self.patch_size * x) + patch_center.unsqueeze(0)

            ## Record initial distance to patch center
            dist = torch.sum(torch.square(tx_pos[0] - patch_center), dim=-1)

            if n_intermediates > 0:
                return (batch["tx_idx"], tx_pos, dist)
            else:
                return (batch["tx_idx"], tx_pos[-1], dist)


def infer(
    dataset: TxPatchDataset | ChainDataset,
    flow_inference: FlowInference,
    batch_size: int,
):

    dataloader = DataLoader(
        dataset,
        num_workers=0,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=tx_patch_collate_fn,
    )
    accum = {}
    for batch in tqdm(dataloader):
        tx_idx, tx_pos, dist = flow_inference.flow(batch)
        patch_center = torch.repeat_interleave(
            batch["patch_center"], repeats=batch["tx_seqlens"], dim=0
        )
        tx_initial_pos = (0.5 * flow_inference.patch_size * batch["tx_xy"]) + patch_center
        for ii, xy_final, d, gene_idx, xy_initial in zip(
            tx_idx.cpu().tolist(),
            tx_pos.cpu().tolist(),
            dist.cpu().tolist(),
            batch["tx_gene_idx"].cpu().tolist(),
            tx_initial_pos.cpu().tolist(),
        ):
            if (ii not in accum) or (d < accum[ii]["dist"]):
                accum[ii] = {
                    "dist": d,
                    "xy": xy_final,
                    "gene_idx": gene_idx,
                    "xy_initial": xy_initial,
                }

    tx_idx = np.array(list(accum.keys()))
    tx_xy = np.array([val["xy"] for val in accum.values()])
    tx_gene_idx = np.array([val["gene_idx"] for val in accum.values()])
    tx_xy_initial = np.array([val["xy_initial"] for val in accum.values()])
    tx_dist = np.array([val["dist"] for val in accum.values()])

    df = pd.DataFrame(
        {
            "tx_idx": tx_idx,
            "tx_x": tx_xy[:, 0],
            "tx_y": tx_xy[:, 1],
            "tx_initial_x": tx_xy_initial[:, 0],
            "tx_initial_y": tx_xy_initial[:, 1],
            "tx_gene_idx": tx_gene_idx,
            "tx_dist": tx_dist,
        }
    )

    return df
