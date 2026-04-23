import torch
import torch.nn.functional as F


def radius_graph_cpu(
    xy: torch.Tensor,  ## (N, 2) float32
    seqlens: torch.Tensor,  ## (B,) int64
    radius: float,
) -> torch.Tensor:
    device = xy.device
    B = seqlens.shape[0]

    cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0).long()
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seqlens).long()

    all_edges = []
    for i in range(B):
        sub_xy = xy[batch_idx == i]
        dist_mat = torch.cdist(sub_xy, sub_xy, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        row, col = torch.where(dist_mat <= radius)
        mask = row < col
        edges = torch.stack([row[mask], col[mask]], dim=0)
        all_edges.append(edges + cu_seqlens[i])
    edges = torch.cat(all_edges, dim=1)
    return edges


def _radius_graph_cuda(
    xy: torch.Tensor,
    seqlens: torch.Tensor,
    radius: float,
    max_edges_per_point: int = 64,
) -> torch.Tensor:
    """Fallback CUDA radius graph without Triton — O(N^2) per batch but avoids the dependency."""
    device = xy.device
    B = seqlens.shape[0]

    cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0).long()
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seqlens).long()

    all_edges = []
    for i in range(B):
        sub_xy = xy[batch_idx == i]
        dist_mat = torch.cdist(sub_xy, sub_xy, p=2)
        row, col = torch.where(dist_mat <= radius)
        mask = row < col
        edges = torch.stack([row[mask], col[mask]], dim=0)
        all_edges.append(edges + cu_seqlens[i])

    if all_edges:
        edges = torch.cat(all_edges, dim=1)
    else:
        edges = torch.zeros((2, 0), dtype=torch.long, device=device)
    return edges


def radius_graph(
    xy: torch.Tensor,  ## (N, 2) float32
    seqlens: torch.Tensor,  ## (B,) int64
    radius: float,
    max_edges_per_point: int = 32,
) -> torch.Tensor:
    """Compute radius graph within each batch element.

    Returns edge_index of shape (2, E) where both endpoints belong to the
    same batch element and are within ``radius`` of each other.
    """
    if xy.is_cpu:
        return radius_graph_cpu(xy, seqlens, radius)

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _radius_graph_kernel(
            xy_ptr,
            batch_idx_ptr,
            cu_seqlens_ptr,
            edges_src_ptr,
            edges_dst_ptr,
            edge_counts_ptr,
            N,
            radius_sq,
            max_edges: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            if pid >= N:
                return

            src_x = tl.load(xy_ptr + pid * 2 + 0)
            src_y = tl.load(xy_ptr + pid * 2 + 1)

            batch = tl.load(batch_idx_ptr + pid)
            batch_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
            batch_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)

            count: tl.int32 = 0
            out_base = pid * max_edges

            for block_start in range(batch_start, batch_end, BLOCK_SIZE):
                still_need = count < max_edges
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = (offsets < batch_end) & still_need

                cand_x = tl.load(xy_ptr + offsets * 2 + 0, mask=mask, other=0.0)
                cand_y = tl.load(xy_ptr + offsets * 2 + 1, mask=mask, other=0.0)

                dx = cand_x - src_x
                dy = cand_y - src_y
                dist_sq = dx * dx + dy * dy

                valid = mask & (dist_sq <= radius_sq) & (offsets != pid)

                valid_int = valid.to(tl.int32)
                prefix = tl.cumsum(valid_int, axis=0) - valid_int
                block_valid_count = tl.sum(valid_int)

                write_pos = count + prefix
                write_mask = valid & (write_pos < max_edges)
                tl.store(edges_src_ptr + out_base + write_pos, pid, mask=write_mask)
                tl.store(edges_dst_ptr + out_base + write_pos, offsets, mask=write_mask)

                count += block_valid_count
                count = tl.minimum(count, max_edges)

            tl.store(edge_counts_ptr + pid, count)

        N = xy.shape[0]
        xy_c = xy.contiguous().float()
        B = seqlens.shape[0]
        cu_seqlens = F.pad(seqlens.cumsum(-1), pad=(1, 0), value=0).contiguous().long()
        batch_idx = (
            torch.repeat_interleave(torch.arange(B, device=xy.device), seqlens).contiguous().long()
        )

        edges_src = torch.empty(N * max_edges_per_point, dtype=torch.int64, device=xy.device)
        edges_dst = torch.empty(N * max_edges_per_point, dtype=torch.int64, device=xy.device)
        edge_counts = torch.zeros(N, dtype=torch.int32, device=xy.device)

        _radius_graph_kernel[(N,)](
            xy_c,
            batch_idx,
            cu_seqlens,
            edges_src,
            edges_dst,
            edge_counts,
            N=N,
            radius_sq=radius * radius,
            max_edges=max_edges_per_point,
            BLOCK_SIZE=256,
        )

        counts = edge_counts.long()
        total_edges = counts.sum().item()
        if total_edges == 0:
            return torch.zeros((2, 0), dtype=torch.int64, device=xy.device)

        offsets_per_point = torch.arange(N, device=xy.device) * max_edges_per_point
        slot_idx = torch.arange(max_edges_per_point, device=xy.device)
        valid_mask = slot_idx.unsqueeze(0) < counts.unsqueeze(1)
        flat_idx = offsets_per_point.unsqueeze(1) + slot_idx.unsqueeze(0)
        gather_idx = flat_idx[valid_mask]

        src = edges_src[gather_idx]
        dst = edges_dst[gather_idx]
        edges = torch.sort(torch.stack([src, dst], dim=0), dim=0).values
        return torch.unique(edges, dim=1)

    except (ImportError, Exception):
        return _radius_graph_cuda(xy, seqlens, radius, max_edges_per_point)
