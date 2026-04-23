import torch
import torch.nn.functional as F

from ebbf.data.noise_genes import sample_noise_genes
from ebbf.data.patch import NuclearBatch
from ebbf.data.radius_graph import radius_graph
from ebbf.data.rasterize import rasterize_triangles_fast
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
        boot_indices = boot_indices - torch.repeat_interleave(cu_sample_sizes, sample_sizes)
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
    seq_offsets = torch.repeat_interleave(torch.arange(seqlens.shape[0], device=device), seqlens)
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
    lengths = torch.minimum(seqlens, torch.tensor(max_len, device=seqlens.device))  # (B,)

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


def sample_swap_neighbors(
    edges: torch.Tensor,  ## (2, n_edges)
    idx: torch.Tensor,  ## (m,) transcript-to-nucleus mapping
    N: int,  ## total number of nuclei
) -> torch.Tensor:
    """For each transcript, sample a uniform random swap neighbor of its nucleus.

    Nuclei with no swap edges get a self-loop (swap to themselves).
    """
    device = edges.device

    # Symmetrize edges (both directions)
    src = torch.cat([edges[0], edges[1]])
    dst = torch.cat([edges[1], edges[0]])

    # Add self-loops for isolated nuclei
    ones = torch.ones(src.shape[0], device=device, dtype=torch.long)
    degree = torch.zeros(N, device=device, dtype=torch.long)
    degree.scatter_add_(0, src, ones)

    isolated = torch.where(degree == 0)[0]
    if isolated.numel() > 0:
        src = torch.cat([src, isolated])
        dst = torch.cat([dst, isolated])

    # Re-sort by source nucleus
    sort_idx = torch.argsort(src)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]

    # Recompute degree and offsets with self-loops included
    ones = torch.ones(src_sorted.shape[0], device=device, dtype=torch.long)
    degree = torch.zeros(N, device=device, dtype=torch.long)
    degree.scatter_add_(0, src_sorted, ones)

    offsets = torch.zeros(N + 1, device=device, dtype=torch.long)
    torch.cumsum(degree, dim=0, out=offsets[1:])

    # For each transcript, sample a random neighbor of its nucleus
    deg = degree[idx]
    random_offsets = (torch.rand(idx.shape[0], device=device) * deg.float()).long()
    global_idx = offsets[idx] + random_offsets
    swap_idx = dst_sorted[global_idx]

    return swap_idx


def sample_intercellular_locations(
    sample_seqlens: torch.Tensor,
    nuc2cell_triangle_xy: torch.Tensor,
    nuc2cell_triangle_seqlens: torch.Tensor,
    nuc_seqlens: torch.Tensor,
    patch_center: torch.Tensor,
    patch_size: float,
    resolution: int = 64,
) -> torch.Tensor:
    """Sample random positions in the intercellular space for each batch element."""
    B = nuc_seqlens.shape[0]
    device = nuc_seqlens.device

    indices = torch.repeat_interleave(torch.arange(B, device=device), nuc_seqlens)

    triangle_seqlens = torch.zeros(B, dtype=nuc2cell_triangle_seqlens.dtype, device=device)
    triangle_seqlens.scatter_add_(0, indices, nuc2cell_triangle_seqlens)

    bounds = torch.concat(
        [patch_center - (patch_size / 2), patch_center + (patch_size / 2)], dim=-1
    )

    pixel_bounds, pixel_seqlens = rasterize_triangles_fast(
        nuc2cell_triangle_xy=nuc2cell_triangle_xy,
        triangle_seqlens=triangle_seqlens,
        bounds=bounds,
        resolution=resolution,
        mode="negative",
    )

    pixel_cu_seqlens = F.pad(pixel_seqlens.cumsum(-1), pad=(1, 0), value=0)

    batch_indices = torch.arange(B, device=device)
    sample_batch_ids = batch_indices.repeat_interleave(sample_seqlens)

    lo = pixel_cu_seqlens[sample_batch_ids]
    hi = pixel_cu_seqlens[sample_batch_ids + 1]

    rand_frac = torch.rand(sample_seqlens.sum(), device=device)
    sample_pixel_idx = lo + (rand_frac * (hi - lo).float()).long()

    sample_pixel_bounds = pixel_bounds[sample_pixel_idx]
    lo_xy = sample_pixel_bounds[:, :2]
    hi_xy = sample_pixel_bounds[:, 2:]
    sample_xy = lo_xy + torch.rand_like(lo_xy) * (hi_xy - lo_xy)
    return sample_xy


class TranscriptSampler:
    def __init__(
        self,
        patch_size: float,
        patch_padding: float,
        max_tx: int,
        min_tx_factor: int,
        max_tx_factor: int,
        oversample_tx_factor: int,
        img_resolution: int,
        n_genes: int,
        draw_ellipse: bool = False,
        max_nuc_dropout_prob: float = 0.0,
        triangle_dirichlet_scale: float = 25.0,
        max_cell_expansion_factor: float = 1.25,
        max_noise_prob: float = 0.1,
        max_noise_scale: float = 2.0,
        max_nucleus_swap_prob: float = 0.0,
        nucleus_swap_radius: float = 15.0,
        max_garbage_prob: float = 0.0,
        min_garbage_intercellular_ratio: float = 3.0,
        max_garbage_intercellular_ratio: float = 10.0,
        raster_resolution: int = 64,
        min_uniform_prob: float = 0.1,
        max_uniform_prob: float = 0.8,
        device: torch.device | None = None,
    ):
        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.max_tx = max_tx
        self.min_tx_factor = min_tx_factor
        self.max_tx_factor = max_tx_factor
        self.oversample_tx_factor = oversample_tx_factor
        self.draw_ellipse = draw_ellipse
        self.max_nuc_dropout_prob = max_nuc_dropout_prob
        self.img_resolution = img_resolution
        self.n_genes = n_genes
        self.triangle_dirichlet_scale = triangle_dirichlet_scale
        self.max_cell_expansion_factor = max_cell_expansion_factor
        self.max_noise_prob = max_noise_prob
        self.max_noise_scale = max_noise_scale
        self.max_nucleus_swap_prob = max_nucleus_swap_prob
        self.nucleus_swap_radius = nucleus_swap_radius
        self.max_garbage_prob = max_garbage_prob
        self.min_garbage_intercellular_ratio = min_garbage_intercellular_ratio
        self.max_garbage_intercellular_ratio = max_garbage_intercellular_ratio
        self.raster_resolution = raster_resolution
        self.min_uniform_prob = min_uniform_prob
        self.max_uniform_prob = max_uniform_prob
        self.device = device

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
        sizes = 1.0 + (self.max_cell_expansion_factor - 1.0) * torch.rand(n_nucs, 1, device=device)
        tx_offset = tx_offset * sizes[tx2nuc_idx]

        ## Perturb with a v. small amount of white noise
        tx_offset = tx_offset + 0.1 * torch.randn_like(tx_offset)

        return tx_offset

    def sample_transcripts(self, nuc_batch: NuclearBatch):
        patch_center = nuc_batch.patch_center
        nuc_seqlens = nuc_batch.nuc_seqlens
        nuc2ellipse_mat = nuc_batch.nuc2ellipse_mat
        nuc2ctype = nuc_batch.nuc2ctype
        nuc2xy = nuc_batch.nuc2xy

        B = patch_center.shape[0]
        device = patch_center.device

        ## Subsample nuclei of origin
        unfiltered_tx2nuc_idx, sample_sizes = bootstrap_sample_sequence(
            counts=nuc_batch.nuc2cell_count,
            seqlens=nuc_seqlens,
            with_offsets=True,
            min_factor=self.min_tx_factor,
            max_factor=self.max_tx_factor,
            oversample_factor=self.oversample_tx_factor,
            max_size=self.max_tx,
        )

        ## Sample the genes of each transcript
        indices = categorical_sample_index_sequence(
            probs=nuc_batch.nuc2genes_probs_seq,
            idx=unfiltered_tx2nuc_idx,
            seqlens=nuc_batch.nuc2genes_seqlens,
            with_offsets=True,
        )

        unfiltered_tx_genes = nuc_batch.nuc2genes_idx_seq[indices]

        unfiltered_tx_batch_indices = torch.repeat_interleave(
            torch.arange(B).to(device), sample_sizes
        )

        ## Optionally swap each transcript's nucleus of origin with a nearby neighbour
        if self.max_nucleus_swap_prob > 0:
            total_num_nuclei = nuc_seqlens.sum()
            nuc_edges = radius_graph(
                nuc2xy, nuc_seqlens, radius=self.nucleus_swap_radius, max_edges_per_point=32
            )
            tx2swap_nuc_idx = sample_swap_neighbors(
                nuc_edges, unfiltered_tx2nuc_idx, total_num_nuclei
            )
            per_nucleus_swap_frac = self.max_nucleus_swap_prob * torch.rand(
                total_num_nuclei, device=device
            )
            per_tx_swap_prob = per_nucleus_swap_frac[unfiltered_tx2nuc_idx]
            swap_mask = torch.rand_like(per_tx_swap_prob) < per_tx_swap_prob
            unfiltered_tx2generating_nuc_idx = torch.where(
                swap_mask, tx2swap_nuc_idx, unfiltered_tx2nuc_idx
            )
        else:
            unfiltered_tx2generating_nuc_idx = unfiltered_tx2nuc_idx

        ## Sample distances from nuclei for each transcript
        unfiltered_tx_offset = self._sample_polygon_offset(
            tx2nuc_idx=unfiltered_tx2generating_nuc_idx,
            nuc2xy=nuc2xy,
            nuc2cell_triangle_probs=nuc_batch.nuc2cell_triangle_probs,
            nuc2cell_triangle_xy=nuc_batch.nuc2cell_triangle_xy,
            nuc2cell_triangle_seqlens=nuc_batch.nuc2cell_triangle_seqlens,
        )

        ## Corrupt some number of tx per patch (scale offset to push them further from nucleus)
        per_patch_corrupt_frac = self.max_noise_prob * torch.rand(B, device=device)
        per_tx_corrupt_prob = per_patch_corrupt_frac[unfiltered_tx_batch_indices]
        corrupt_mask = torch.rand_like(per_tx_corrupt_prob) < per_tx_corrupt_prob
        n_corrupt = torch.count_nonzero(corrupt_mask)
        corrupt_scales = 1.0 + (self.max_noise_scale - 1.0) * torch.rand(
            n_corrupt, 1, device=device
        )
        unfiltered_tx_offset[corrupt_mask] = corrupt_scales * unfiltered_tx_offset[corrupt_mask]

        ## Sample xy positions of transcripts
        unfiltered_tx_nuc_xy = nuc2xy[unfiltered_tx2nuc_idx]
        unfiltered_tx_xy = nuc2xy[unfiltered_tx2generating_nuc_idx] + unfiltered_tx_offset

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

        ## Subset to get down to max_tx
        filtered_tx_seqlens = torch.zeros(B, dtype=torch.long, device=device)
        filtered_tx_seqlens = filtered_tx_seqlens.scatter_add_(
            0, unfiltered_tx_batch_indices, mask.long()
        )

        idxs, tx_seqlens = clip_sequence(filtered_tx_seqlens, max_len=self.max_tx)

        tx_xy = filtered_tx_xy[idxs]
        tx_nuc_xy = filtered_tx_nuc_xy[idxs]
        tx_genes = filtered_tx_genes[idxs]
        tx_batch_indices = torch.repeat_interleave(torch.arange(B).to(device), tx_seqlens)
        tx_is_noise = torch.zeros(tx_xy.shape[0], dtype=torch.float, device=device)

        #### Generate garbage (noise) transcripts ####
        if self.max_garbage_prob > 0.0:
            ## Per-patch garbage rate
            per_patch_garbage_frac = self.max_garbage_prob * torch.rand(B, device=device)
            per_tx_garbage_prob = per_patch_garbage_frac[tx_batch_indices]
            garbage_tx_mask = torch.rand_like(per_tx_garbage_prob) < per_tx_garbage_prob
            garbage_tx_seqlens = torch.zeros(B, dtype=torch.long, device=device)
            garbage_tx_seqlens.scatter_add_(0, tx_batch_indices, garbage_tx_mask.long())
            garbage_tx_batch_indices = torch.repeat_interleave(
                torch.arange(B, device=device), garbage_tx_seqlens
            )
            n_garbage_tx = garbage_tx_seqlens.sum()

            ## Garbage positions: mixture of intercellular-space and uniform-on-patch
            per_patch_intercellular_ratio = self.min_garbage_intercellular_ratio + (
                self.max_garbage_intercellular_ratio - self.min_garbage_intercellular_ratio
            ) * torch.rand(B, device=device)
            per_patch_intercellular_frac = per_patch_intercellular_ratio / (
                1.0 + per_patch_intercellular_ratio
            )
            per_garbage_tx_intercellular_prob = per_patch_intercellular_frac[
                garbage_tx_batch_indices
            ]
            garbage_tx_intercellular_mask = (
                torch.rand_like(per_garbage_tx_intercellular_prob)
                < per_garbage_tx_intercellular_prob
            )
            garbage_tx_intercellular_seqlens = torch.zeros(B, dtype=torch.long, device=device)
            garbage_tx_intercellular_seqlens.scatter_add_(
                0, garbage_tx_batch_indices, garbage_tx_intercellular_mask.long()
            )

            ## Generate intercellular garbage tx
            garbage_tx_intercellular_xy = sample_intercellular_locations(
                garbage_tx_intercellular_seqlens,
                nuc_batch.nuc2cell_triangle_xy,
                nuc_batch.nuc2cell_triangle_seqlens,
                nuc_batch.nuc_seqlens,
                nuc_batch.patch_center,
                patch_size=self.patch_size,
                resolution=self.raster_resolution,
            )

            ## Generate uniform garbage tx
            garbage_tx_uniform_seqlens = garbage_tx_seqlens - garbage_tx_intercellular_seqlens
            garbage_tx_uniform_batch_indices = torch.repeat_interleave(
                torch.arange(B, device=device), garbage_tx_uniform_seqlens
            )
            patch_bounds = torch.concat(
                [
                    patch_center - (self.patch_size / 2),
                    patch_center + (self.patch_size / 2),
                ],
                dim=-1,
            )
            per_garbage_tx_uniform_bounds = patch_bounds[garbage_tx_uniform_batch_indices]
            lo = per_garbage_tx_uniform_bounds[:, :2]
            hi = per_garbage_tx_uniform_bounds[:, 2:]
            garbage_tx_uniform_xy = lo + torch.rand_like(lo) * (hi - lo)

            ## Assemble all garbage tx positions
            garbage_tx_xy = torch.empty(n_garbage_tx, 2, dtype=torch.float, device=device)
            garbage_tx_xy[garbage_tx_intercellular_mask] = garbage_tx_intercellular_xy
            garbage_tx_xy[~garbage_tx_intercellular_mask] = garbage_tx_uniform_xy

            ## Overwrite real tx positions with garbage positions; target = stay put
            tx_xy[garbage_tx_mask] = garbage_tx_xy
            tx_nuc_xy[garbage_tx_mask] = garbage_tx_xy

            ## Sample garbage gene identities
            tx_genes[garbage_tx_mask] = sample_noise_genes(
                sample_seqlens=garbage_tx_seqlens,
                nuc_seqlens=nuc_batch.nuc_seqlens,
                nuc2genes_idx_seq=nuc_batch.nuc2genes_idx_seq,
                nuc2genes_probs_seq=nuc_batch.nuc2genes_probs_seq,
                nuc2genes_seqlens=nuc_batch.nuc2genes_seqlens,
                nuc2total_gene_count=nuc_batch.nuc2total_gene_count,
                num_genes=self.n_genes,
                min_uniform_prob=self.min_uniform_prob,
                max_uniform_prob=self.max_uniform_prob,
            )

            ## Mark as noise
            tx_is_noise[garbage_tx_mask] = 1.0

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
            "tx_is_noise": tx_is_noise,
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
