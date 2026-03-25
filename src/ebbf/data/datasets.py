import logging
import math
from collections.abc import Iterator
from copy import deepcopy

import numpy as np
import pyarrow.dataset as ds
import torch
from scipy.sparse import csr_array
from torch.utils.data import Dataset, IterableDataset

from ebbf.data.patch import NuclearBatch
from ebbf.data.utils import RTree, TensorSequence, TorchCSR

logger = logging.getLogger(__name__)


class NuclearPatchDataset(Dataset):
    def __init__(
        self,
        nuc2xy: np.ndarray,  ## n_nucs x 2
        nuc2total_gene_count: np.ndarray,  ## n_nucs
        nuc2gene_probs: csr_array,  ## n_nucs x n_genes
        nuc2ctype: np.ndarray,  ## n_nucs
        nuc2cell_count: np.ndarray,
        patch_size: float,
        patch_padding: float,
        dataset_size: int,
        min_nucs: int = 5,
        nuc2ellipse_mat: np.ndarray | None = None,  ## n_nucs x 2 x 2
        cell_triangle_num: np.ndarray | None = None,
        cell_triangle_xy: np.ndarray | None = None,
        cell_triangle_probs: np.ndarray | None = None,
    ):

        # Compute bounds directly from nuclei positions
        self.data_min = torch.from_numpy(nuc2xy.min(axis=0)).float()
        self.data_max = torch.from_numpy(nuc2xy.max(axis=0)).float()

        self.nuc2gene_probs = TorchCSR(nuc2gene_probs)
        self.n_genes: int = nuc2gene_probs.shape[1]
        self.nuc2total_gene_count = torch.from_numpy(nuc2total_gene_count).float()
        self.nuc2ctype = torch.from_numpy(nuc2ctype).long()
        self.nuc2xy = torch.from_numpy(nuc2xy).float()
        self.nuc2ellipse_mat = (
            torch.from_numpy(nuc2ellipse_mat).float() if nuc2ellipse_mat is not None else None
        )
        # Store raw numpy array for lazy RTree initialization
        # (RTree uses C extension that doesn't survive fork())
        self._nuc2xy_numpy = nuc2xy
        self._rtree = None
        self._rtree_pid = None  # Track which process created the RTree
        self.nuc2cell_count = torch.from_numpy(nuc2cell_count).float()

        if cell_triangle_num is None:
            self.cell_polygons = None
        else:
            self.cell_polygons = TensorSequence(
                seqlens=torch.from_numpy(cell_triangle_num).long(),
                cell_triangle_xy=torch.from_numpy(cell_triangle_xy).float(),
                cell_triangle_probs=torch.from_numpy(cell_triangle_probs).float(),
            )

        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.min_nucs = min_nucs
        self.dataset_size = dataset_size

    @property
    def rtree(self):
        """Lazy initialization of RTree to handle fork() correctly.

        The rtree library uses C extensions that don't survive fork().
        By checking the process ID, we ensure each forked process creates
        its own RTree instance.
        """
        import os

        current_pid = os.getpid()
        if self._rtree is None or self._rtree_pid != current_pid:
            self._rtree = RTree(self._nuc2xy_numpy)
            self._rtree_pid = current_pid
        return self._rtree

    def __len__(self):
        n = self.dataset_size
        return n

    def _sample_patch(self):
        ## Sample a patch from the data region
        n_nuc = 0
        n_tries = 0
        while n_nuc < self.min_nucs:
            n_tries += 1
            if n_tries % 10000 == 0:
                logger.warning(
                    f"_sample_patch: {n_tries} tries to find patch with "
                    f">={self.min_nucs} nuclei (found {n_nuc})"
                )

            ## Sample the center of the patch within data bounds
            patch_center = torch.rand(1, 2)
            patch_center = (self.data_max - self.data_min) * patch_center + self.data_min

            idx = self.rtree.lookup_center_radius(
                x_center=patch_center[0, 0].item(),
                y_center=patch_center[0, 1].item(),
                radius=(0.5 * self.patch_size + self.patch_padding),
            )

            nuc2xy = self.nuc2xy[idx]

            ## Count the number of nuclei that fall within the actual patch
            inner_min = patch_center - 0.5 * self.patch_size
            inner_max = patch_center + 0.5 * self.patch_size
            inner_min_mask = (inner_min <= nuc2xy).all(axis=-1)  ## n_nuc
            inner_max_mask = (inner_max >= nuc2xy).all(axis=-1)  ## n_nuc
            inner_mask = inner_min_mask & inner_max_mask
            n_nuc = torch.count_nonzero(inner_mask).item()

        nuc2genes_idx_seq, nuc2genes_probs_seq, nuc2genes_seqlens = (
            self.nuc2gene_probs.slice_rows_sequence(idx)
        )
        nuc2total_gene_count = self.nuc2total_gene_count[idx]
        nuc2cell_count = self.nuc2cell_count[idx]
        nuc2ctype = self.nuc2ctype[idx]

        result = {
            "patch_center": patch_center,
            "nuc_idx": idx,
            "nuc2xy": nuc2xy,
            "nuc2ctype": nuc2ctype,
            "nuc2total_gene_count": nuc2total_gene_count,
            "nuc2cell_count": nuc2cell_count,
            "nuc2genes_idx_seq": nuc2genes_idx_seq,
            "nuc2genes_probs_seq": nuc2genes_probs_seq,
            "nuc2genes_seqlens": nuc2genes_seqlens,
        }

        if self.nuc2ellipse_mat is not None:
            result["nuc2ellipse_mat"] = self.nuc2ellipse_mat[idx]
        else:
            result["nuc2ellipse_mat"] = torch.zeros(1)

        if self.cell_polygons is not None:
            cell_result = self.cell_polygons.slice(idx)
            result["nuc2cell_triangle_xy"] = cell_result["cell_triangle_xy"]
            result["nuc2cell_triangle_probs"] = cell_result["cell_triangle_probs"]
            result["nuc2cell_triangle_seqlens"] = cell_result["seqlens"]

        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._sample_patch()


def nuclear_patch_collate_fn(batch: tuple[dict[str, torch.Tensor]]):
    batch = NuclearBatch(
        patch_center=torch.concat([x["patch_center"] for x in batch], dim=0),
        nuc_idx=torch.concat([x["nuc_idx"] for x in batch], dim=0),
        nuc2xy=torch.concat([x["nuc2xy"] for x in batch], dim=0),
        nuc2ctype=torch.concat([x["nuc2ctype"] for x in batch], dim=0),
        nuc2total_gene_count=torch.concat([x["nuc2total_gene_count"] for x in batch], dim=0),
        nuc2cell_count=torch.concat([x["nuc2cell_count"] for x in batch], dim=0),
        nuc2genes_idx_seq=torch.concat([x["nuc2genes_idx_seq"] for x in batch], dim=0),
        nuc2genes_probs_seq=torch.concat([x["nuc2genes_probs_seq"] for x in batch], dim=0),
        nuc2ellipse_mat=torch.concat([x["nuc2ellipse_mat"] for x in batch], dim=0),
        nuc2genes_seqlens=torch.concat([x["nuc2genes_seqlens"] for x in batch], dim=0),
        nuc_seqlens=torch.tensor([x["nuc2xy"].shape[0] for x in batch]),
        nuc2cell_triangle_xy=torch.concat([x["nuc2cell_triangle_xy"] for x in batch], dim=0),
        nuc2cell_triangle_probs=torch.concat([x["nuc2cell_triangle_probs"] for x in batch], dim=0),
        nuc2cell_triangle_seqlens=torch.concat(
            [x["nuc2cell_triangle_seqlens"] for x in batch], dim=0
        ),
    )
    return batch


class TxPatchDataset(IterableDataset):
    def __init__(
        self,
        boxes: np.ndarray,  ## xmin, ymin, xmax, ymax
        tx_dataset: ds.Dataset,
        nuc_xy: np.ndarray,
        genes: np.ndarray,
        patch_size: float,  # width = height
        patch_padding: float,
        overlap: float,  # x_overlap = y_overlap
        max_tx: int,
        shuffle: bool = False,
        nuc_ellipse_mat: np.ndarray | None = None,
        nuc_ctype: np.ndarray | None = None,
        x_col: str = "x",
        y_col: str = "y",
    ):

        self.boxes = boxes
        self.patch_padding = patch_padding
        self.patch_size = patch_size
        self.overlap = overlap
        self.tx_dataset = tx_dataset
        self.nuc_xy = nuc_xy
        self.nmax = max_tx
        self.shuffle = shuffle
        self.nuc_ellipse_mat = nuc_ellipse_mat
        self.nuc_ctype = nuc_ctype
        self.x_col = x_col
        self.y_col = y_col
        gene2idx = {x: i for i, x in enumerate(genes)}
        self.gene_lookup = np.vectorize(gene2idx.get)

        # Calculate patch grid
        self.step_size = patch_size - overlap
        self.patch_coords = []
        for box_idx, (xmin, ymin, xmax, ymax) in enumerate(self.boxes):
            nx_patches = math.ceil((xmax - xmin - overlap) / self.step_size)
            ny_patches = math.ceil((ymax - ymin - overlap) / self.step_size)
            self.patch_coords.extend(
                [(box_idx, i, j) for i in range(nx_patches) for j in range(ny_patches)]
            )
        self.seeds = {x: (i + 100) for i, x in enumerate(self.patch_coords)}

    def _get_patch_bounds(
        self, box_idx: int, patch_i: int, patch_j: int
    ) -> tuple[float, float, float, float]:
        """Get the bounding box for patch (i, j)"""
        box_xmin, box_ymin, _, _ = self.boxes[box_idx]
        xmin = box_xmin + patch_i * self.step_size
        ymin = box_ymin + patch_j * self.step_size
        xmax = xmin + self.patch_size
        ymax = ymin + self.patch_size
        return xmin, xmax, ymin, ymax

    def _generate_subsets_for_patch(
        self, box_idx: int, patch_i: int, patch_j: int
    ) -> Iterator[dict]:
        """Generate all dataset elements for a single patch"""
        res = {}

        # Get patch bounds
        xmin, xmax, ymin, ymax = self._get_patch_bounds(box_idx, patch_i, patch_j)

        ## Patch-level info
        res["patch_center"] = torch.tensor(
            [[xmin + 0.5 * self.patch_size, ymin + 0.5 * self.patch_size]],
        )
        res["patch_indices"] = torch.tensor([[box_idx, patch_i, patch_j]], dtype=torch.int64)

        # Get the transcript info
        gene_cond = ds.field("is_gene")
        min_cond = (ds.field(self.x_col) >= xmin) & (ds.field(self.y_col) >= ymin)
        max_cond = (ds.field(self.x_col) <= xmax) & (ds.field(self.y_col) <= ymax)
        patch_table = self.tx_dataset.filter(gene_cond & min_cond & max_cond).to_table(
            ["transcript_id", self.x_col, self.y_col, "feature_name"]
        )

        res["tx_xy"] = torch.from_numpy(
            np.stack(
                (
                    patch_table[self.x_col].to_numpy(),
                    patch_table[self.y_col].to_numpy(),
                ),
                axis=1,
            )
        ).float()
        n_tx = res["tx_xy"].shape[0]

        ## No transcripts here, we create an empty element
        if n_tx == 0:
            return

        res["tx_gene_idx"] = torch.from_numpy(
            self.gene_lookup(patch_table["feature_name"].to_numpy())
        ).long()
        res["tx_nuc_idx"] = torch.from_numpy(np.full((n_tx,), fill_value=-1, dtype=np.int64)).long()
        res["tx_idx"] = torch.from_numpy(patch_table["transcript_id"].to_numpy().copy()).long()

        ## Get nuclei info
        pad = self.patch_padding
        nuc_mask = (
            (self.nuc_xy[:, 0] > (xmin - pad))
            & (self.nuc_xy[:, 1] > (ymin - pad))
            & (self.nuc_xy[:, 0] < (xmax + pad))
            & (self.nuc_xy[:, 1] < (ymax + pad))
        )
        res["nuc_xy"] = torch.from_numpy(self.nuc_xy[nuc_mask]).float()

        if self.nuc_ctype is not None:
            res["nuc_ctype"] = torch.from_numpy(self.nuc_ctype[nuc_mask]).long()
        if self.nuc_ellipse_mat is not None:
            res["nuc_ellipse_mat"] = torch.from_numpy(
                2 * self.nuc_ellipse_mat[nuc_mask] / self.patch_size
            ).float()

        res["n_nuc"] = res["nuc_xy"].shape[0]

        ## Standardize coordinates
        res["tx_xy"] = 2.0 * (res["tx_xy"] - res["patch_center"]) / self.patch_size
        res["nuc_xy"] = 2.0 * (res["nuc_xy"] - res["patch_center"]) / self.patch_size

        if n_tx <= self.nmax:
            res["n_tx"] = n_tx
            res["subset_index"] = 0
            yield res
        else:
            # Multiple subsets needed
            n_subsets = math.ceil(n_tx / self.nmax)

            # Shuffle points if requested
            if self.shuffle:
                ## There is a single seed per patch
                seed = self.seeds[box_idx, patch_i, patch_j]
                indices = torch.randperm(n_tx, generator=torch.Generator().manual_seed(seed))
            else:
                indices = torch.arange(n_tx)

            for subset_idx in range(n_subsets):
                start_idx = subset_idx * self.nmax
                end_idx = min(start_idx + self.nmax, n_tx)

                # If this is the last subset and it's smaller than nmax,
                # include some overlap to ensure we have exactly nmax points
                if ((end_idx - start_idx) < self.nmax) and subset_idx > 0:
                    start_idx = max(0, n_tx - self.nmax)
                    end_idx = n_tx

                sub_indices = indices[start_idx:end_idx]

                sub_res = {}
                for key in res.keys():
                    if key.startswith("tx_"):
                        sub_res[key] = res[key][sub_indices]
                    else:
                        sub_res[key] = deepcopy(res[key])
                sub_res["n_tx"] = sub_indices.shape[0]
                sub_res["subset_index"] = subset_idx
                yield sub_res

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all dataset elements"""
        # Create random number generator
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Split patches across workers
            patches_per_worker = len(self.patch_coords) // num_workers
            start_idx = worker_id * patches_per_worker
            if worker_id == num_workers - 1:
                end_idx = len(self.patch_coords)
            else:
                end_idx = start_idx + patches_per_worker

            worker_patches = self.patch_coords[start_idx:end_idx]
        else:
            # Single process
            worker_patches = self.patch_coords

        # Iterate through patches
        for box_idx, patch_i, patch_j in worker_patches:
            # Generate all subsets for this patch
            for element in self._generate_subsets_for_patch(box_idx, patch_i, patch_j):
                if (element["n_tx"] > 0) and (element["n_nuc"] > 0):
                    yield element


def tx_patch_collate_fn(batch: list[dict]):
    result = {}
    elem1 = batch[0]
    keys = list(elem1.keys())
    for key in keys:
        if torch.is_tensor(elem1[key]):
            result[key] = torch.concat([elem[key] for elem in batch], dim=0)
        else:
            result[key] = torch.tensor([elem[key] for elem in batch])
    result["tx_seqlens"] = result.pop("n_tx")
    result["nuc_seqlens"] = result.pop("n_nuc")
    return result
