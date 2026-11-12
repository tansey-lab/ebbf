import logging
import os
import tempfile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm

from ebbf.data.datasets import TxPatchDataset, tx_patch_collate_fn
from ebbf.data.utils import ellipse_grid_indices
from ebbf.model.ema import EMA

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
        if isinstance(model, EMA):
            model = model.model
        self.model = model.to(device)
        self.model.train(False)

        self.device = device
        self.atol = atol
        self.rtol = rtol
        self.L = (self.patch_size + 2 * self.patch_padding) / self.patch_size

    def prepare_inputs(self, batch: dict[str, torch.Tensor]):
        batch = {key: val.to(self.device) for key, val in batch.items()}
        if ("nuc_ellipse_mat" in batch) and ("nuc_xy" in batch):
            ellipse_coordinates, ellipse_seqlens = ellipse_grid_indices(
                S=batch["nuc_ellipse_mat"],
                c=batch["nuc_xy"],
                L=self.L,
                M=self.img_resolution,
            )
            batch["nuc_ellipse_coordinates"] = ellipse_coordinates
            batch["nuc_ellipse_seqlens"] = ellipse_seqlens
        return batch

    def _model_kwargs(self, batch: dict) -> dict:
        return dict(
            gene_idx=batch["tx_gene_idx"],
            tx_seqlens=batch["tx_seqlens"],
            nuc_seqlens=batch.get("nuc_seqlens"),
            nuc_ellipse_pos=batch.get("nuc_ellipse_coordinates"),
            nuc_ellipse_seqlens=batch.get("nuc_ellipse_seqlens"),
        )

    def flow(self, batch: dict[str, torch.Tensor], n_intermediates: int = 0):
        with torch.no_grad():
            batch = self.prepare_inputs(batch)
            patch_center = torch.repeat_interleave(
                batch["patch_center"], repeats=batch["tx_seqlens"], dim=0
            )
            kwargs = self._model_kwargs(batch)

            def ode_func(t, tx_pos_t: torch.Tensor):
                pred = self.model(tx_pos=tx_pos_t, t=torch.atleast_1d(t), **kwargs)
                return pred[:, :2]  ## velocity only; ignore noise logit during integration

            time_grid = torch.linspace(
                start=0.0, end=1.0, steps=(n_intermediates + 2), device=self.device
            )

            ## torchdiffeq keeps solver time in float64 (as in ndiff); MPS has no float64
            options = {"dtype": torch.float32} if self.device.type == "mps" else None
            x = odeint(
                ode_func,
                batch["tx_xy"],
                time_grid,
                atol=self.atol,
                rtol=self.rtol,
                options=options,
            )

            ## Scale back to the original coordinates
            tx_pos: torch.Tensor = (0.5 * self.patch_size * x) + patch_center.unsqueeze(0)

            ## Record initial distance to patch center
            dist = torch.sum(torch.square(tx_pos[0] - patch_center), dim=-1)

            ## Evaluate noise head at t=0 and t=1 to get noise probabilities
            pred_0 = self.model(tx_pos=x[0], t=torch.tensor([0.0], device=self.device), **kwargs)
            pred_1 = self.model(tx_pos=x[-1], t=torch.tensor([1.0], device=self.device), **kwargs)
            noise_logits = torch.stack([pred_0[:, 2], pred_1[:, 2]], dim=0)  ## (2, S)

            return {
                "tx_idx": batch["tx_idx"],
                "tx_pos": tx_pos if n_intermediates > 0 else tx_pos[-1],
                "sq_dist": dist,
                "tx_noise_prob": torch.sigmoid(noise_logits),
            }

    def instantaneous_vector_field(
        self,
        batch: dict[str, torch.Tensor],
        t: float,
        gene_idx: torch.Tensor,
        rescale: bool = True,
        return_points: bool = True,
        restrict_vector: bool = True,
    ):
        with torch.no_grad():
            batch = self.prepare_inputs(batch)
            kwargs = self._model_kwargs(batch)
            kwargs.pop("gene_idx")
            X = self.model.encode_and_backbone(
                tx_pos=batch["tx_xy"],
                gene_idx=batch["tx_gene_idx"],
                t=torch.tensor([t], device=self.device),
                **kwargs,
            )
            vector_field = self.model.tx_grid_decoder.vector_field(
                X=X, items=gene_idx.to(self.device), restrict_vector=restrict_vector
            )

            if (not restrict_vector) and (vector_field.shape[-1] == 3):
                vector_field[..., 2] = torch.sigmoid(vector_field[..., 2])

            if rescale:
                vector_field[..., :2] = 0.5 * self.patch_size * vector_field[..., :2]

        if not return_points:
            return vector_field

        K = self.model.img_resolution
        L = self.model.padding_bound

        # Pixel i maps to: -L + (i + 0.5) * (2L / K); x varies along dim 0, y along dim 1
        coords_1d = -L + (torch.arange(K, device=self.device) + 0.5) * (2 * L / K)
        coords_x, coords_y = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        pixel_centers = torch.stack([coords_x, coords_y], dim=-1)  # K x K x 2
        points = batch["patch_center"][:, None, None, :] + (
            0.5 * self.patch_size * pixel_centers[None, ...]
        )  ## B x K x K x 2
        return vector_field, points

    def bottleneck_encoding(
        self,
        batch: dict,
        t: float = 0.0,
        pool: str = "meanmax",
        include_mid: bool = True,
        return_bounds: bool = True,
    ):
        """Encoder-trunk (bottleneck) representation for each patch in ``batch``.

        pool: 'none' -> B x C_b x m x m; 'mean' -> B x C_b; 'meanmax' -> B x 2*C_b.
        With pool='none' the two spatial axes are (x, y), not (row=y, col=x).
        Optionally also returns B x 4 inner-patch bounds (xmin, ymin, xmax, ymax) in um.
        """
        with torch.no_grad():
            batch = self.prepare_inputs(batch)
            B = batch["tx_seqlens"].shape[0]
            t_vec = torch.full((B,), float(t), device=self.device, dtype=torch.float32)

            h = self.model.bottleneck(
                gene_idx=batch["tx_gene_idx"],
                tx_pos=batch["tx_xy"],
                tx_seqlens=batch["tx_seqlens"],
                t=t_vec,
                nuc_ellipse_pos=batch.get("nuc_ellipse_coordinates"),
                nuc_ellipse_seqlens=batch.get("nuc_ellipse_seqlens"),
                nuc_seqlens=batch.get("nuc_seqlens"),
                include_mid=include_mid,
                pool=pool,
            )

            if not return_bounds:
                return h

            half = 0.5 * self.patch_size
            pc = batch["patch_center"]
            bounds = torch.stack(
                [pc[:, 0] - half, pc[:, 1] - half, pc[:, 0] + half, pc[:, 1] + half], dim=-1
            )
        return h, bounds


def _batch_to_table(flow_inference: FlowInference, batch: dict[str, torch.Tensor]) -> pa.Table:
    """Run inference on a single batch and return results as an Arrow table."""
    result = flow_inference.flow(batch)
    tx_seqlens = batch["tx_seqlens"]
    patch_center = torch.repeat_interleave(batch["patch_center"], repeats=tx_seqlens, dim=0)
    tx_initial_pos = ((0.5 * flow_inference.patch_size * batch["tx_xy"]) + patch_center).numpy()
    patch_index = torch.repeat_interleave(batch["patch_indices"], repeats=tx_seqlens, dim=0)
    subset_index = torch.repeat_interleave(batch["subset_index"], repeats=tx_seqlens, dim=0)

    tx_pos = result["tx_pos"].cpu().numpy()
    noise_prob = result["tx_noise_prob"].cpu().numpy()  ## (2, S)
    patch_index = patch_index.numpy()

    return pa.table(
        {
            "tx_idx": result["tx_idx"].cpu().numpy(),
            "tx_x": tx_pos[:, 0],
            "tx_y": tx_pos[:, 1],
            "tx_initial_x": tx_initial_pos[:, 0],
            "tx_initial_y": tx_initial_pos[:, 1],
            "tx_gene_idx": batch["tx_gene_idx"].numpy(),
            "tx_dist": result["sq_dist"].cpu().numpy(),
            "patch_idx_0": patch_index[:, 0],
            "patch_idx_1": patch_index[:, 1],
            "patch_idx_2": patch_index[:, 2],
            "subset_idx": subset_index.numpy(),
            "tx_noise_prob_0": noise_prob[0],
            "tx_noise_prob_1": noise_prob[1],
        }
    )


## Each transcript keeps the copy from the patch whose center it lies closest to;
## ties are broken by patch / subset index.
DEDUP_SORT_COLS = ["tx_dist", "patch_idx_0", "patch_idx_1", "patch_idx_2", "subset_idx"]


def infer(
    dataset: TxPatchDataset,
    flow_inference: FlowInference,
    num_workers: int,
    batch_size: int,
    filename: str,
):
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=tx_patch_collate_fn,
    )

    ## Stream batch results to a temporary parquet, then dedup in polars
    output_dir = os.path.dirname(filename)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".parquet", dir=output_dir or None, delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    writer = None
    try:
        for batch in tqdm(dataloader):
            table = _batch_to_table(flow_inference, batch)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        logger.warning("No patches with transcripts and nuclei; writing nothing to %s", filename)
        os.remove(tmp_path)
        return

    (
        pl.scan_parquet(tmp_path)
        .sort(DEDUP_SORT_COLS)
        .unique(subset=["tx_idx"], keep="first")
        .sink_parquet(filename)
    )
    os.remove(tmp_path)
