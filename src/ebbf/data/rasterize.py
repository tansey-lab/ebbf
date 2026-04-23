import torch


def rasterize_triangles_fast(
    nuc2cell_triangle_xy: torch.Tensor,  ## (N, 3, 2) triangle vertices
    triangle_seqlens: torch.Tensor,  ## (B,) number of triangles per batch element
    bounds: torch.Tensor,  ## (B, 4) bounding boxes as (x_min, y_min, x_max, y_max)
    resolution: int,  ## number of pixels per dimension
    mode: str = "positive",  ## "positive" or "negative"
) -> tuple[torch.Tensor, torch.Tensor]:  ## (pixel_bounds: (M, 4), pixel_seqlens: (B,))
    """Fast rasterization using per-triangle bounding box pre-filtering.

    For each batch element, tests which pixel centers fall inside at least one
    triangle, then returns the corners of those pixels. Uses per-triangle
    axis-aligned bounding boxes to skip candidate pixels, rather than testing
    all R^2 pixels per triangle.

    Args:
        nuc2cell_triangle_xy: (N, 3, 2) concatenated triangle vertices.
        triangle_seqlens: (B,) number of triangles per batch element.
        bounds: (B, 4) per-batch bounding box (x_min, y_min, x_max, y_max).
        resolution: number of pixels per dimension; grid is resolution x resolution.
        mode: "positive" returns covered pixels, "negative" returns uncovered pixels.

    Returns:
        pixel_bounds: (M, 4) bounding box of each selected pixel (x_min, y_min, x_max, y_max).
        pixel_seqlens: (B,) number of selected pixels per batch element.
    """
    device = nuc2cell_triangle_xy.device
    dtype = nuc2cell_triangle_xy.dtype
    B = triangle_seqlens.shape[0]
    N = nuc2cell_triangle_xy.shape[0]
    R = resolution
    R2 = R * R

    ## --- bounds decomposition ---
    x_min_b, y_min_b, x_max_b, y_max_b = bounds.unbind(-1)  ## each (B,)
    origin = torch.stack([x_min_b, y_min_b], dim=-1)  ## (B, 2)
    scale = torch.stack([x_max_b - x_min_b, y_max_b - y_min_b], dim=-1)  ## (B, 2)
    step = scale / R  ## (B, 2)

    ## --- map each triangle to its batch element ---
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), triangle_seqlens)  ## (N,)

    ## --- transform triangle vertices to grid coordinates ---
    tri_origin = origin[batch_idx]  ## (N, 2)
    tri_scale = scale[batch_idx]  ## (N, 2)
    grid_verts = (nuc2cell_triangle_xy - tri_origin[:, None, :]) / tri_scale[:, None, :] * R
    ## grid_verts: (N, 3, 2) — dim=-1 is (col_float, row_float)

    ## --- per-triangle bounding boxes in grid space ---
    col_coords = grid_verts[..., 0]  ## (N, 3)
    row_coords = grid_verts[..., 1]  ## (N, 3)

    col_min = col_coords.min(dim=1).values.floor().long().clamp(0, R - 1)  ## (N,)
    col_max = col_coords.max(dim=1).values.ceil().long().sub(1).clamp(0, R - 1)
    row_min = row_coords.min(dim=1).values.floor().long().clamp(0, R - 1)
    row_max = row_coords.max(dim=1).values.ceil().long().sub(1).clamp(0, R - 1)

    n_cols = (col_max - col_min + 1).clamp(min=0)  ## (N,)
    n_rows = (row_max - row_min + 1).clamp(min=0)
    n_cands = n_cols * n_rows  ## (N,)

    total_cands = n_cands.sum().item()

    ## --- build covered mask ---
    covered = torch.zeros(B * R2, dtype=torch.uint8, device=device)

    if total_cands > 0:
        ## --- enumerate candidate (triangle, pixel) pairs ---
        tri_idx = torch.repeat_interleave(
            torch.arange(N, device=device), n_cands
        )  ## (total_cands,)

        offsets = torch.repeat_interleave(n_cands.cumsum(0) - n_cands, n_cands)  ## (total_cands,)
        local_idx = torch.arange(total_cands, device=device) - offsets

        n_cols_expanded = n_cols[tri_idx]  ## (total_cands,)
        local_row = local_idx // n_cols_expanded
        local_col = local_idx % n_cols_expanded

        row = row_min[tri_idx] + local_row  ## (total_cands,)
        col = col_min[tri_idx] + local_col

        ## --- compute pixel centers for candidate pairs ---
        batch_of_cand = batch_idx[tri_idx]  ## (total_cands,)
        px = origin[batch_of_cand, 0] + (col.to(dtype) + 0.5) * step[batch_of_cand, 0]
        py = origin[batch_of_cand, 1] + (row.to(dtype) + 0.5) * step[batch_of_cand, 1]
        p = torch.stack([px, py], dim=-1)  ## (total_cands, 2)

        ## --- point-in-triangle test on candidates only ---
        v0, v1, v2 = nuc2cell_triangle_xy[tri_idx].unbind(1)  ## each (total_cands, 2)

        def cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        d0 = cross2d(v1 - v0, p - v0)  ## (total_cands,)
        d1 = cross2d(v2 - v1, p - v1)
        d2 = cross2d(v0 - v2, p - v2)

        eps = 1e-7
        hit = ((d0 >= -eps) & (d1 >= -eps) & (d2 >= -eps)) | (
            (d0 <= eps) & (d1 <= eps) & (d2 <= eps)
        )  ## (total_cands,)

        ## --- scatter to (B, R^2) mask ---
        flat_pixel = row * R + col  ## (total_cands,)
        flat_idx = batch_of_cand * R2 + flat_pixel  ## (total_cands,)

        covered.scatter_reduce_(
            0, flat_idx.long(), hit.to(torch.uint8), reduce="amax", include_self=True
        )

    covered = covered.reshape(B, R2).bool()  ## (B, R^2)

    ## --- select pixels by mode and build output ---
    mask = covered if mode == "positive" else ~covered

    lo = 0.5 / R
    hi = 1.0 - 0.5 / R
    coords = torch.linspace(lo, hi, R, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    grid_norm = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  ## (R^2, 2)
    pixel_centers = origin[:, None, :] + grid_norm[None, :, :] * scale[:, None, :]

    half_step = 0.5 * step
    half_step_expanded = half_step[:, None, :].expand(-1, R2, -1)

    sel_centers = pixel_centers[mask]
    sel_half_step = half_step_expanded[mask]

    pixel_bounds = torch.stack(
        [
            sel_centers[:, 0] - sel_half_step[:, 0],
            sel_centers[:, 1] - sel_half_step[:, 1],
            sel_centers[:, 0] + sel_half_step[:, 0],
            sel_centers[:, 1] + sel_half_step[:, 1],
        ],
        dim=-1,
    )  ## (M, 4)

    pixel_seqlens = mask.sum(dim=-1).long()  ## (B,)

    return pixel_bounds, pixel_seqlens
