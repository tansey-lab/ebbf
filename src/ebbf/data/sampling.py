import numpy as np
import torch
import torch.nn.functional as F
from torch.linalg import vector_norm

from ebbf.data.patch import NuclearBatch
from ebbf.data.utils import ellipse_grid_indices, segmented_cumsum, segmented_sum


def dirchlet_sample_sequence(alpha: torch.Tensor, seqlens: torch.Tensor):
    device = alpha.device
    seg_ids = torch.arange(len(seqlens), device=device).repeat_interleave(seqlens)

    gamma_samples = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
    segment_sums = torch.zeros(seqlens.shape[0], device=device).index_add_(
        0, seg_ids, gamma_samples
    )
    dirichlet_samples = gamma_samples / segment_sums[seg_ids]

    return dirichlet_samples


def bootstrap_sample_sequence(
    counts: torch.Tensor,
    seqlens: torch.Tensor,
    with_offsets: bool,
    min_factor: float | torch.Tensor = 1.0,
    max_factor: float | torch.Tensor = 1.0,
    oversample_factor: float | torch.Tensor = 1.0,
    max_size: int | float | None = None,
):
    device = counts.device
    B = seqlens.shape[0]

    ## For each element in data, get the batch it belongs to
    indices = torch.repeat_interleave(torch.arange(B, device=device), seqlens)

    ## Compute total sums
    batch_sums = torch.zeros(B, dtype=counts.dtype, device=device)
    batch_sums.scatter_add_(0, indices, counts)

    ## perturb, threshold, oversample to get sample_sizes
    num_factor = (max_factor - min_factor) * torch.rand(
        batch_sums.shape[0], device=device
    ) + min_factor
    sample_sizes = num_factor * batch_sums
    if max_size is not None:
        sample_sizes = torch.clip(sample_sizes, max=max_size)
    sample_sizes = (oversample_factor * sample_sizes).long()

    ## Normalize counts to probability vectors
    probs = counts.float() / torch.repeat_interleave(batch_sums.float(), seqlens)

    ## Compute cumulative probabilities
    ## cu_probs = torch.cumsum(probs, dim=0)
    cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0)
    seg_cu_probs = segmented_cumsum(probs, cu_seqlens=cu_seqlens)
    seq_offsets = torch.repeat_interleave(torch.arange(B, device=device), seqlens)
    cu_probs = seg_cu_probs + seq_offsets

    ## Sample bootstrap indices
    total_num = torch.sum(sample_sizes, dtype=torch.long)
    boot_offsets = torch.repeat_interleave(torch.arange(B, device=device), sample_sizes)
    z = torch.rand(total_num, device=device) + boot_offsets.float()
    boot_indices = torch.searchsorted(cu_probs, z, side="left")

    ## boot_indices should fall within appropriate bounds
    lb_idx = cu_seqlens[boot_offsets]
    ub_idx = cu_seqlens[boot_offsets + 1] - 1
    boot_indices = torch.clamp(boot_indices, min=lb_idx, max=ub_idx)

    if not with_offsets:
        cu_sample_sizes = F.pad((seqlens - 1).cumsum(-1), pad=(1, 0), value=0)[:-1]
        boot_indices = boot_indices - torch.repeat_interleave(
            cu_sample_sizes, sample_sizes
        )
        boot_indices = boot_indices - boot_offsets
    return (boot_indices, sample_sizes)


def categorical_sample_index_sequence(
    probs: torch.Tensor, idx: torch.Tensor, seqlens: torch.Tensor, with_offsets: bool
):
    """
    probs, seqlens defines a sequence of probability vectors.

    if cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1,0), value=0)

    then probs[cu_seqlens[i]:cu_seqlens[i+1]] defines the ith probability vector.

    This function returns a vector of indices out_idx such that

    out_idx[i] is a random draw from the idx[i]-th probability vector. i.e.

    from probs[cu_seqlens[idx[i]]:cu_seqlens[idx[i]+1]]
    """

    n_samples = idx.shape[0]
    device = probs.device

    ## Compute cumulative probabilities
    ## cu_probs = torch.cumsum(probs, dim=0)
    cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0)
    seg_cu_probs = segmented_cumsum(probs, cu_seqlens=cu_seqlens)
    seq_offsets = torch.repeat_interleave(
        torch.arange(seqlens.shape[0], device=device), seqlens
    )
    cu_probs = seg_cu_probs + seq_offsets

    ## Sample positions
    z = torch.rand(n_samples, device=device) + idx.float()
    indices = torch.searchsorted(cu_probs, z, side="left")

    ## indices should fall within appropriate bounds
    lb_idx = cu_seqlens[idx]
    ub_idx = cu_seqlens[idx + 1] - 1
    indices = torch.clamp(indices, min=lb_idx, max=ub_idx)

    ## Subtract out the offsets
    if not with_offsets:
        cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0)
        indices = indices - cu_seqlens[idx]

    return indices


def clip_sequence(seqlens: torch.Tensor, max_len: int):

    # 1) cumulative raw‐starts in `data`
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0), value=0)  # (B+1,)

    # 2) clipped lengths: how many to keep from each seq
    lengths = torch.minimum(
        seqlens, torch.tensor(max_len, device=seqlens.device)
    )  # (B,)

    # 3) flattened “new” sequence‐starts in sdata
    cum_eff = F.pad(lengths.cumsum(0), (1, 0), value=0)  # (B+1,)
    total_keep = int(lengths.sum().item())  # total number of elements

    # 4) for each kept element, which sequence it came from?
    #    e.g. [0 repeated lengths[0] times, 1 repeated lengths[1] times, …]
    block_idx = torch.arange(seqlens.size(0), device=seqlens.device).repeat_interleave(
        lengths
    )  # (total_keep,)

    # 5) for each kept element, its offset within that sequence:
    #    global position in sdata minus that sequence's cumulative-start
    local_pos = (
        torch.arange(total_keep, device=seqlens.device) - cum_eff[block_idx]
    )  # (total_keep,)

    # 6) actual indices into original `data`:
    idxs = cu_seqlens[:-1][block_idx] + local_pos  # (total_keep,)

    return (idxs, lengths)


class TranscriptSampler:
    def __init__(
        self,
        patch_size: float,
        patch_padding: float,
        max_tx: int,
        min_tx_factor: int,
        max_tx_factor: int,
        oversample_tx_factor: int,
        hists: np.ndarray,  ## n_punches x (n_ctypes+1) x n_bins
        bin_width: float,
        correction_factor: np.ndarray | float,
        sample_mode: str,
        img_resolution: int,
        draw_ellipse: bool = False,
        max_nuc_dropout_prob: float = 0.0,
        triangle_dirichlet_scale: float = 25.0,
        max_cell_expansion_factor: float = 1.25,
        max_noise_prob: float = 0.1,
        max_noise_scale: float = 2.0,
        hist_mode: str = "punch+ctype",
        device: torch.device | None = None,
        eps: float = 1e-8,
    ):

        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.max_tx = max_tx
        self.min_tx_factor = min_tx_factor
        self.max_tx_factor = max_tx_factor
        self.oversample_tx_factor = oversample_tx_factor
        self.bin_width = bin_width
        self.hist_mode = hist_mode
        self.n_punches = hists.shape[0]
        self.sample_mode = sample_mode
        self.draw_ellipse = draw_ellipse
        self.max_nuc_dropout_prob = max_nuc_dropout_prob
        self.img_resolution = img_resolution
        self.triangle_dirichlet_scale = triangle_dirichlet_scale
        self.max_cell_expansion_factor = max_cell_expansion_factor
        self.max_noise_prob = max_noise_prob
        self.max_noise_scale = max_noise_scale

        hists_ = hists.astype(float) + eps
        if hist_mode == "punch+ctype":
            hist_probs = hists_ / hists_.sum(axis=-1, keepdims=True)
        elif hist_mode == "punch":
            hist_probs = hists_[:, -1, :] / hists_[:, -1, :].sum(axis=-1, keepdims=True)
        elif hist_mode == "ctype":
            hists_ = hists_.sum(axis=0)
            hist_probs = hists_ / hists_.sum(axis=-1, keepdims=True)
        else:
            hists_ = hists_[:, -1, :].sum(axis=0)
            hist_probs = hists_ / hists_.sum()

        self.hist_probs = torch.from_numpy(hist_probs).float()

        ## Correction factor = # of total transcripts / # of nuclear transcripts
        self.correction_factor = torch.tensor(correction_factor).float()

        if len(self.correction_factor.shape) == 0:  ## Global correction factor
            self.correction_factor = torch.repeat_interleave(
                self.correction_factor, self.n_punches
            )

        if device is not None:
            self.hist_probs = self.hist_probs.to(device)
            self.correction_factor = self.correction_factor.to(device)

    def _sample_hist_distance(self, ctype: torch.Tensor, punch_idx: torch.Tensor):
        n_tx = ctype.shape[0]
        if self.hist_mode == "punch+ctype":  ## hist_probs - (p x c x k)
            probs = self.hist_probs[punch_idx, ctype]
        elif self.hist_mode == "punch":  ## hist_probs - (p x k)
            probs = self.hist_probs[punch_idx]
        elif self.hist_mode == "ctype":  ## hist_probs - (c x k)
            probs = self.hist_probs[ctype]
        else:  ## hist_probs - (k,)
            probs = torch.repeat_interleave(
                self.hist_probs.unsqueeze(0), repeats=n_tx, dim=0
            )

        bin_idx = torch.multinomial(probs, num_samples=1).squeeze().float()
        offset = torch.rand(ctype.shape[0], device=ctype.device)
        tx_dist = self.bin_width * (offset + bin_idx)

        return tx_dist

    def _sample_isotropic_offsets(self, ctype: torch.Tensor, punch_idx: torch.Tensor):
        n_tx = ctype.shape[0]
        device = ctype.device

        tx_dist = self._sample_hist_distance(ctype, punch_idx)
        tx_direction = torch.randn(n_tx, 2, device=device)
        tx_direction: torch.Tensor = tx_direction / vector_norm(
            tx_direction, dim=-1, keepdim=True
        )

        tx_offset = tx_dist.unsqueeze(-1) * tx_direction

        return tx_offset

    def _sample_ellipse_offsets(
        self,
        ctype: torch.Tensor,
        punch_idx: torch.Tensor,
        nuc2ellipse_mat: torch.Tensor,
    ):

        n_tx = ctype.shape[0]
        device = ctype.device
        tx_offset = torch.empty(n_tx, 2, dtype=torch.float32, device=device)

        ## With some probability, we generate tx within the nucleus
        within_nuc_prob = 1.0 / self.correction_factor[punch_idx]
        within_nuc_mask = torch.rand(n_tx, device=device) <= within_nuc_prob
        n_within_nuc = torch.count_nonzero(within_nuc_mask)

        ## Generate points on the unit circle
        unit_circle_pts = torch.randn(n_tx, 2, device=device)
        unit_circle_pts: torch.Tensor = unit_circle_pts / vector_norm(
            unit_circle_pts, dim=-1, keepdim=True
        )

        ## For points inside the nucleus, transform uniform over disk to ellipse
        radius = torch.sqrt(torch.rand(n_within_nuc, 1, device=device))
        rand_within_circle_pts = radius * unit_circle_pts[within_nuc_mask]  ## S x 2
        inside_nuc_pts = torch.einsum(
            "bij,bj->bi", nuc2ellipse_mat[within_nuc_mask], rand_within_circle_pts
        )

        ## For points outside the nucleus, generate random point on ellipse and
        ## the amount sampled from
        outside_nuc_mask = ~within_nuc_mask
        dist_from_ellipse = self._sample_hist_distance(
            ctype[outside_nuc_mask], punch_idx[outside_nuc_mask]
        )
        on_ellipse_pts = torch.einsum(
            "bij, bj -> bi",
            nuc2ellipse_mat[outside_nuc_mask],
            unit_circle_pts[outside_nuc_mask],
        )
        norm_factor: torch.Tensor = 1.0 + (
            dist_from_ellipse / vector_norm(on_ellipse_pts, dim=-1)
        )
        outside_nuc_pts = on_ellipse_pts * norm_factor.unsqueeze(-1)

        tx_offset[within_nuc_mask] = inside_nuc_pts
        tx_offset[outside_nuc_mask] = outside_nuc_pts

        return tx_offset

    def _sample_polygon_offset(
        self,
        tx2nuc_idx: torch.Tensor,
        nuc2xy: torch.Tensor,
        nuc2cell_triangle_probs: torch.Tensor,
        nuc2cell_triangle_xy: torch.Tensor,
        nuc2cell_triangle_seqlens: torch.Tensor,
    ):

        device = tx2nuc_idx.device
        n_tx = tx2nuc_idx.shape[0]
        n_nucs = nuc2xy.shape[0]

        ## perturb the triangle probabilities
        perturbed_nuc2cell_triangle_probs = dirchlet_sample_sequence(
            self.triangle_dirichlet_scale * nuc2cell_triangle_probs,
            seqlens=nuc2cell_triangle_seqlens,
        )

        indices = categorical_sample_index_sequence(
            probs=perturbed_nuc2cell_triangle_probs,
            idx=tx2nuc_idx,
            seqlens=nuc2cell_triangle_seqlens,
            with_offsets=True,
        )

        verts = nuc2cell_triangle_xy[indices]
        r1, r2 = torch.rand(2, n_tx, 1, device=device)  ## n x 1
        rootr1 = torch.sqrt(r1)
        a_idx = torch.arange(n_tx, device=device)
        tx_pos = (
            (1 - rootr1) * verts[a_idx, 0, :]
            + rootr1 * (1 - r2) * verts[a_idx, 1, :]
            + rootr1 * r2 * verts[a_idx, 2, :]
        )

        ## Place in offset position
        tx_offset = tx_pos - nuc2xy[tx2nuc_idx]

        ## Perturb the sizes of the polygons
        sizes = 1.0 + (self.max_cell_expansion_factor - 1.0) * torch.rand(
            n_nucs, 1, device=device
        )
        tx_offset = tx_offset * sizes[tx2nuc_idx]

        ## Perturb with a v. small amount of white noise
        tx_offset = tx_offset + 0.1 * torch.randn_like(tx_offset)

        return tx_offset

    def sample_transcripts(self, nuc_batch: NuclearBatch):
        patch_center = nuc_batch.patch_center
        punch_idx = nuc_batch.punch_idx
        nuc_seqlens = nuc_batch.nuc_seqlens
        nuc2ellipse_mat = nuc_batch.nuc2ellipse_mat
        nuc2ctype = nuc_batch.nuc2ctype
        nuc2xy = nuc_batch.nuc2xy

        B = patch_center.shape[0]
        device = patch_center.device

        ## Subsample nuclei of origin
        # corr_factor = self.correction_factor[punch_idx]
        # unfiltered_tx2nuc_idx, sample_sizes = bootstrap_sample_sequence(
        #     counts=nuc_batch.nuc2total_gene_count,
        #     seqlens=nuc_seqlens,
        #     with_offsets=True,
        #     min_factor=(self.min_tx_factor*corr_factor),
        #     max_factor=(self.max_tx_factor*corr_factor),
        #     oversample_factor=self.oversample_tx_factor,
        #     max_size=self.max_tx)

        unfiltered_tx2nuc_idx, sample_sizes = bootstrap_sample_sequence(
            counts=nuc_batch.nuc2cell_count,
            seqlens=nuc_seqlens,
            with_offsets=True,
            min_factor=self.min_tx_factor,
            max_factor=self.max_tx_factor,
            oversample_factor=self.oversample_tx_factor,
            max_size=self.max_tx,
        )

        unfiltered_tx2nuc_idx.shape[0]

        ## Sample the genes of each transcript
        indices = categorical_sample_index_sequence(
            probs=nuc_batch.nuc2genes_probs_seq,
            idx=unfiltered_tx2nuc_idx,
            seqlens=nuc_batch.nuc2genes_seqlens,
            with_offsets=True,
        )

        unfiltered_tx_genes = nuc_batch.nuc2genes_idx_seq[indices]

        unfiltered_tx_2_ctype = nuc2ctype[unfiltered_tx2nuc_idx]
        unfiltered_tx_batch_indices = torch.repeat_interleave(
            torch.arange(B).to(device), sample_sizes
        )
        unfiltered_tx_2_punch_idx = punch_idx[unfiltered_tx_batch_indices]

        ## Sample distances from nuclei for each transcript
        if self.sample_mode == "ellipse":
            unfiltered_tx_offset = self._sample_ellipse_offsets(
                ctype=unfiltered_tx_2_ctype,
                punch_idx=unfiltered_tx_2_punch_idx,
                nuc2ellipse_mat=nuc2ellipse_mat[unfiltered_tx2nuc_idx],
            )
        elif self.sample_mode == "polygon":
            unfiltered_tx_offset = self._sample_polygon_offset(
                tx2nuc_idx=unfiltered_tx2nuc_idx,
                nuc2xy=nuc2xy,
                nuc2cell_triangle_probs=nuc_batch.nuc2cell_triangle_probs,
                nuc2cell_triangle_xy=nuc_batch.nuc2cell_triangle_xy,
                nuc2cell_triangle_seqlens=nuc_batch.nuc2cell_triangle_seqlens,
            )
        else:
            unfiltered_tx_offset = self._sample_isotropic_offsets(
                ctype=unfiltered_tx_2_ctype, punch_idx=unfiltered_tx_2_punch_idx
            )

        ## Corrupt some number of tx per patch
        per_patch_corrupt_frac = self.max_noise_prob * torch.rand(B, device=device)
        per_tx_corrupt_prob = per_patch_corrupt_frac[unfiltered_tx_batch_indices]
        per_tx_uniform = torch.rand_like(per_tx_corrupt_prob)
        corrupt_mask = per_tx_uniform < per_tx_corrupt_prob
        n_corrupt = torch.count_nonzero(corrupt_mask)
        corrupt_scales = 1.0 + (self.max_noise_scale - 1.0) * torch.rand(
            n_corrupt, 1, device=device
        )
        unfiltered_tx_offset[corrupt_mask] = (
            corrupt_scales * unfiltered_tx_offset[corrupt_mask]
        )

        ## Sample xy positions of transcripts
        unfiltered_tx_nuc_xy = nuc2xy[unfiltered_tx2nuc_idx]
        unfiltered_tx_xy = unfiltered_tx_nuc_xy + unfiltered_tx_offset

        ## Filter down to transcripts that fall within the patch
        patch_center_rep = patch_center[unfiltered_tx_batch_indices]
        inner_min = patch_center_rep - 0.5 * self.patch_size
        inner_max = patch_center_rep + 0.5 * self.patch_size
        inner_min_mask = (inner_min <= unfiltered_tx_xy).all(axis=-1)
        inner_max_mask = (inner_max >= unfiltered_tx_xy).all(axis=-1)
        mask = inner_min_mask & inner_max_mask

        filtered_tx_xy = unfiltered_tx_xy[mask]
        filtered_tx_nuc_xy = unfiltered_tx_nuc_xy[mask]
        filtered_tx_genes = unfiltered_tx_genes[mask]
        filtered_tx2nuc_idx = unfiltered_tx2nuc_idx[mask]

        ## Subset to get down to max_tx
        filtered_tx_seqlens = torch.zeros(B, dtype=torch.long, device=device)
        filtered_tx_seqlens = filtered_tx_seqlens.scatter_add_(
            0, unfiltered_tx_batch_indices, mask.long()
        )

        idxs, tx_seqlens = clip_sequence(filtered_tx_seqlens, max_len=self.max_tx)

        tx_xy = filtered_tx_xy[idxs]
        tx_nuc_xy = filtered_tx_nuc_xy[idxs]
        tx_genes = filtered_tx_genes[idxs]
        filtered_tx2nuc_idx[idxs]

        ## Scale xy positions to [-1,1]
        patch_center_rep = torch.repeat_interleave(patch_center, tx_seqlens, dim=0)
        tx_xy = 2.0 * (tx_xy - patch_center_rep) / self.patch_size
        tx_nuc_xy = 2.0 * (tx_nuc_xy - patch_center_rep) / self.patch_size

        nuc_patch_center_rep = torch.repeat_interleave(patch_center, nuc_seqlens, dim=0)
        nuc2xy = 2.0 * (nuc2xy - nuc_patch_center_rep) / self.patch_size

        if self.max_nuc_dropout_prob > 0:
            ## Generate per-batch dropout probs
            prob = self.max_nuc_dropout_prob * torch.rand(B, device=device)
            prob = torch.repeat_interleave(prob, nuc_seqlens)

            ## Generate probability per-nucleus
            drop_p = torch.rand(prob.shape[0], device=device)
            keep_mask = drop_p > prob

            ## Drop it like it's hot
            nuc2xy = nuc2xy[keep_mask]
            nuc2ctype = nuc2ctype[keep_mask]
            nuc_seqlens = segmented_sum(keep_mask.to(torch.long), nuc_seqlens)

            if nuc2ellipse_mat is not None:
                nuc2ellipse_mat = nuc2ellipse_mat[keep_mask]

        result = {
            "tx_xy": tx_xy,
            "tx_nuc_xy": tx_nuc_xy,
            "tx_gene_idx": tx_genes,
            "tx_seqlens": tx_seqlens,
            "nuc2xy": nuc2xy,
            "nuc2ctype": nuc2ctype,
            "nuc_seqlens": nuc_seqlens,
        }

        if self.draw_ellipse:
            L = (self.patch_size + 2 * self.patch_padding) / self.patch_size
            S = 2.0 * nuc2ellipse_mat / self.patch_size
            ellipse_coordinates, ellipse_seqlens = ellipse_grid_indices(
                S=S, c=nuc2xy, L=L, M=self.img_resolution
            )
            result["nuc2ellipse_coordinates"] = ellipse_coordinates
            result["nuc2ellipse_seqlens"] = ellipse_seqlens

        return result


class NoSimTranscriptSampler:
    def __init__(
        self,
        patch_size: float,
        patch_padding: float,
        max_tx: int,
        img_resolution: int,
        draw_ellipse: bool = False,
        max_nuc_dropout_prob: float = 0.0,
        perturbation_scale: float = 0.0,
        device: torch.device | None = None,
        eps: float = 1e-8,
    ):

        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.max_tx = max_tx
        self.img_resolution = img_resolution
        self.draw_ellipse = draw_ellipse
        self.max_nuc_dropout_prob = max_nuc_dropout_prob
        self.perturbation_scale = perturbation_scale
        self.device = device
        self.eps = eps

    def sample_transcripts(
        self,
        patch_center: torch.Tensor,  ## (B, 2)
        tx_xy: torch.Tensor,  ## (T,2)
        tx_gene_idx: torch.Tensor,  ## (T,)
        tx_nuc_xy: torch.Tensor,  ## (T,2)
        tx_nuc_dist: torch.Tensor,  ## (T,)
        tx_seqlens: torch.Tensor,  ## (B, 2)
        nuc2xy: torch.Tensor,  ## (S, 2)
        nuc2ctype: torch.Tensor,  ## (S,)
        nuc_seqlens: torch.Tensor,  ## (B,)
        nuc2ellipse_mat: torch.Tensor | None = None,  ## (S, 2, 2)
        **kwargs,
    ):
        B = tx_seqlens.shape[0]
        n_nuc = nuc2xy.shape[0]
        n_tx = tx_xy.shape[0]
        device = tx_xy.device

        ## Standardize coordinates
        patch_center_rep = torch.repeat_interleave(patch_center, tx_seqlens, dim=0)
        tx_xy = 2.0 * (tx_xy - patch_center_rep) / self.patch_size
        tx_nuc_xy = 2.0 * (tx_nuc_xy - patch_center_rep) / self.patch_size

        nuc_patch_center_rep = torch.repeat_interleave(patch_center, nuc_seqlens, dim=0)
        nuc2xy = 2.0 * (nuc2xy - nuc_patch_center_rep) / self.patch_size

        ## Perturb coordinates
        if self.perturbation_scale > 0:
            ## Perturbation scale is on the original data scale
            scale = self.perturbation_scale * 2.0 / self.patch_size
            perturbation = scale * torch.randn(n_tx, 2, device=device)
            tx_xy = tx_xy + perturbation

        if self.max_nuc_dropout_prob > 0:
            ## Generate per-batch dropout probs
            prob = self.max_nuc_dropout_prob * torch.rand(B, device=device)
            prob = torch.repeat_interleave(prob, nuc_seqlens)

            ## Generate probability per-nucleus
            drop_p = torch.rand(n_nuc, device=device)
            keep_mask = drop_p > prob

            ## Drop it like it's hot
            nuc2xy = nuc2xy[keep_mask]
            nuc2ctype = nuc2ctype[keep_mask]
            nuc_seqlens = segmented_sum(keep_mask.to(torch.long), nuc_seqlens)

            if nuc2ellipse_mat is not None:
                nuc2ellipse_mat = nuc2ellipse_mat[keep_mask]

        result = {
            "tx_xy": tx_xy,
            "tx_nuc_xy": tx_nuc_xy,
            "tx_gene_idx": tx_gene_idx,
            "tx_seqlens": tx_seqlens,
            "nuc2xy": nuc2xy,
            "nuc2ctype": nuc2ctype,
            "nuc_seqlens": nuc_seqlens,
        }

        if self.draw_ellipse:
            L = (self.patch_size + 2 * self.patch_padding) / self.patch_size
            S = 2.0 * nuc2ellipse_mat / self.patch_size
            ellipse_coordinates, ellipse_seqlens = ellipse_grid_indices(
                S=S, c=nuc2xy, L=L, M=self.img_resolution
            )
            result["nuc2ellipse_coordinates"] = ellipse_coordinates
            result["nuc2ellipse_seqlens"] = ellipse_seqlens

        return result
