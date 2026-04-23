import torch
import torch.nn.functional as F

from ebbf.data.utils import segmented_cumsum


def aggregate_batch_distribution(
    nuc_seqlens: torch.Tensor,
    nuc2genes_idx_seq: torch.Tensor,
    nuc2genes_probs_seq: torch.Tensor,
    nuc2genes_seqlens: torch.Tensor,
    nuc2total_gene_count: torch.Tensor,
    num_genes: int,
) -> torch.Tensor:
    device = nuc2genes_idx_seq.device
    B = nuc_seqlens.shape[0]
    M = nuc2genes_seqlens.shape[0]
    G = num_genes

    # Map each sparse entry to its nucleus index and batch index.
    nuc_idx_per_entry = torch.repeat_interleave(torch.arange(M, device=device), nuc2genes_seqlens)
    batch_idx_per_nuc = torch.repeat_interleave(torch.arange(B, device=device), nuc_seqlens)
    batch_idx_per_entry = batch_idx_per_nuc[nuc_idx_per_entry]

    # Compute weighted probabilities.
    weighted_probs = nuc2genes_probs_seq * nuc2total_gene_count[nuc_idx_per_entry]

    # Aggregate into dense (B, G) tensor.
    flat_index = batch_idx_per_entry * G + nuc2genes_idx_seq
    batch_gene_dist = torch.zeros(B * G, device=device, dtype=weighted_probs.dtype)
    batch_gene_dist.scatter_add_(0, flat_index.long(), weighted_probs)
    batch_gene_dist = batch_gene_dist.reshape(B, G)

    # Normalize to get per-batch gene distributions.
    row_sums = batch_gene_dist.sum(dim=1, keepdim=True)
    zero_mask = row_sums == 0
    row_sums = row_sums.masked_fill(zero_mask, 1.0)
    batch_gene_dist = batch_gene_dist / row_sums
    batch_gene_dist = batch_gene_dist.masked_fill(zero_mask.expand_as(batch_gene_dist), 1.0 / G)

    return batch_gene_dist


def sample_noise_genes(
    sample_seqlens: torch.Tensor,
    nuc_seqlens: torch.Tensor,
    nuc2genes_idx_seq: torch.Tensor,
    nuc2genes_probs_seq: torch.Tensor,
    nuc2genes_seqlens: torch.Tensor,
    nuc2total_gene_count: torch.Tensor,
    num_genes: int,
    min_uniform_prob: float,
    max_uniform_prob: float,
) -> torch.Tensor:
    device = nuc2genes_idx_seq.device
    B = nuc_seqlens.shape[0]
    G = num_genes

    batch_gene_dist = aggregate_batch_distribution(
        nuc_seqlens=nuc_seqlens,
        nuc2genes_idx_seq=nuc2genes_idx_seq,
        nuc2genes_probs_seq=nuc2genes_probs_seq,
        nuc2genes_seqlens=nuc2genes_seqlens,
        nuc2total_gene_count=nuc2total_gene_count,
        num_genes=G,
    )

    # Mix with uniform.
    uniform_prob = min_uniform_prob + (max_uniform_prob - min_uniform_prob) * torch.rand(
        B, device=device
    )
    noise_dist = (
        uniform_prob.unsqueeze(1) * (1.0 / G) + (1 - uniform_prob.unsqueeze(1)) * batch_gene_dist
    )

    # Sample noise genes via searchsorted.
    probs = noise_dist.flatten()  # (B*G,)
    seqlens = torch.full((B,), G, device=device, dtype=torch.long)
    cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0)
    seg_cu_probs = segmented_cumsum(probs, cu_seqlens=cu_seqlens)
    seq_offsets = torch.repeat_interleave(torch.arange(B, device=device), seqlens)
    cu_probs = seg_cu_probs + seq_offsets

    total_num = sample_seqlens.sum(dtype=torch.long)
    boot_offsets = torch.repeat_interleave(torch.arange(B, device=device), sample_seqlens)
    z = torch.rand(total_num, device=device) + boot_offsets.float()
    noise_gene_idx = torch.searchsorted(cu_probs, z, side="left")

    lb_idx = cu_seqlens[boot_offsets]
    ub_idx = cu_seqlens[boot_offsets + 1] - 1
    noise_gene_idx = torch.clamp(noise_gene_idx, min=lb_idx, max=ub_idx)

    # Remove offsets to get gene indices in [0, G)
    noise_gene_idx = noise_gene_idx - lb_idx

    return noise_gene_idx
