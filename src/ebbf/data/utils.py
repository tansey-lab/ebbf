import geopandas as gpd
import numpy as np
import shapely
import torch
import torch.nn.functional as F
from rtree import index
from scipy.sparse import csr_array

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def get_dtype(nested_list):
    curr = nested_list
    while isinstance(curr, list):
        curr = curr[0]
    if isinstance(curr, float):
        return torch.get_default_dtype()
    else:
        return torch.int64


class RTree:
    def __init__(self, xy: np.ndarray, **kwargs):
        self._keys = list(kwargs.keys())
        args = [
            arg.tolist() if hasattr(arg, "tolist") else list(arg)
            for arg in kwargs.values()
        ]
        self._dtypes = [get_dtype(arg) for arg in args]

        if len(args) == 0:

            def generator_function():
                for i, (x, y) in enumerate(xy):
                    yield (i, (float(x), float(y), float(x), float(y)), None)
        else:

            def generator_function():
                for i, ((x, y), obj) in enumerate(zip(xy, zip(*args))):
                    yield (i, (float(x), float(y), float(x), float(y)), obj)

        self.index_table = index.Index(generator_function())

    def get_bbox(self):
        xmin, ymin, xmax, ymax = self.index_table.bounds
        bbox = (xmin, xmax, ymin, ymax)
        return bbox

    def lookup_center_radius(
        self, x_center: float, y_center: float, radius: float, only_index: bool = True
    ):
        return self.lookup(
            xmin=(x_center - radius),
            xmax=(x_center + radius),
            ymin=(y_center - radius),
            ymax=(y_center + radius),
            only_index=only_index,
        )

    def lookup(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        only_index: bool = True,
    ):

        bounds = (xmin, ymin, xmax, ymax)

        if only_index:
            indices = torch.tensor(
                list(self.index_table.intersection(bounds)), dtype=torch.int64
            )
            return indices
        else:
            try:
                output = list(self.index_table.intersection(bounds, objects=True))
                indices, xy, objs = zip(
                    *[(x.id, (x.bounds[0], x.bounds[2]), x.object) for x in output]
                )
                result = {
                    "indices": torch.tensor(indices, dtype=torch.int64),
                    "xy": torch.tensor(xy),
                }
                if objs[0] is not None:
                    objs_tuple = tuple(zip(*objs))
                    for key, dtype, obj in zip(self._keys, self._dtypes, objs_tuple):
                        result[key] = torch.tensor(obj, dtype=dtype)
            except ValueError:
                result = {
                    "indices": torch.empty(0, dtype=torch.int64),
                    "xy": torch.empty(0, 2),
                }
            return result


class TxRTree:
    def __init__(self, tx_xy: np.ndarray, tx_genes: np.ndarray, tx_nuc_idx: np.ndarray):

        self.n_tx = tx_xy.shape[0]

        def generator_function(tx_xy, tx_genes, tx_nuc_idx):
            for i, ((x, y), gene, nuc_idx) in enumerate(
                zip(tx_xy, tx_genes, tx_nuc_idx)
            ):
                yield (
                    i,
                    (float(x), float(y), float(x), float(y)),
                    (int(gene), int(nuc_idx)),
                )

        self.lookup_table = index.Index(generator_function(tx_xy, tx_genes, tx_nuc_idx))

    def lookup(self, xmin: float, xmax: float, ymin: float, ymax: float):
        bounds = (xmin, ymin, xmax, ymax)
        output = self.lookup_table.intersection(bounds, objects=True)
        try:
            tidx, xy, gene, nidx = zip(
                *[
                    (x.id, (x.bounds[0], x.bounds[2]), x.object[0], x.object[1])
                    for x in output
                ]
            )
        except ValueError:
            tidx = []
            xy = []
            gene = []
            nidx = []

        n_tx = len(xy)
        if n_tx == 0:
            tx_idx = torch.empty(0, dtype=torch.int64)
            tx_xy = torch.empty(0, 2)
            tx_genes = torch.empty(0, dtype=torch.int64)
            tx_nuc_idx = torch.empty(0, dtype=torch.int64)
        else:
            tx_idx = torch.tensor(tidx, dtype=torch.int64)
            tx_xy = torch.tensor(xy)
            tx_genes = torch.tensor(gene, dtype=torch.int64)
            tx_nuc_idx = torch.tensor(nidx, dtype=torch.int64)
        return (tx_idx, tx_xy, tx_genes, tx_nuc_idx)


class TorchCSR:
    def __init__(self, arr: csr_array, device: torch.device | None = None):

        self.indptr = torch.from_numpy(arr.indptr).long()
        self.indices = torch.from_numpy(arr.indices).long()
        self.data = torch.from_numpy(arr.data).float()

        if device is not None:
            self.indptr = self.indptr.to(device)
            self.indices = self.indices.to(device)
            self.data = self.data.to(device)

    def slice_rows_sequence(self, idx: torch.Tensor):
        # 1. Compute each span’s start, end, and length
        starts = self.indptr[idx]  # (M,)
        ends = self.indptr[idx + 1]  # (M,)
        lengths = ends - starts  # (M,)

        # 2. Total number of elements we’ll gather
        total = int(lengths.sum().item())  # scalar

        # 3. Build a flat “span ID” array of length T:
        #    for each span i, repeat i exactly lengths[i] times
        span_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=idx.device), lengths
        )

        # 4. Build a flat “pos_in_span” array of length T:
        #    for each span i, positions 0..lengths[i]-1
        #    again via repeat_interleave on the per-span starts in the flat layout
        span_start_flat = torch.cumsum(lengths, dim=0) - lengths
        pos_in_span = torch.arange(total, device=idx.device) - torch.repeat_interleave(
            span_start_flat, lengths
        )  # (T,)

        # 5. Now compute the real indices into our tensors:
        real_idx = starts[span_ids] + pos_in_span  # (T,)

        indices_out = self.indices[real_idx]  # (T,)
        data_out = self.data[real_idx]  # (T,)

        return (indices_out, data_out, lengths)


class TensorSequence:
    def __init__(self, seqlens: torch.Tensor, **kwargs):
        self.seqlens = seqlens
        self.cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0), value=0)
        self.data: dict[str, torch.Tensor] = {k: v for k, v in kwargs.items()}

    def get(self, i: int):
        result = {
            k: v[self.cu_seqlens[i] : self.cu_seqlens[i + 1]]
            for k, v in self.data.items()
        }
        return result

    def slice(self, idx: torch.Tensor):
        starts = self.cu_seqlens[idx]
        lengths = self.seqlens[idx]
        rel_idx = torch.arange(
            lengths.sum(), device=idx.device
        ) - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
        flat_idx = torch.repeat_interleave(starts, lengths) + rel_idx
        result = {}
        for k, v in self.data.items():
            result[k] = v[flat_idx]
        result["seqlens"] = lengths
        return result


if TRITON_AVAILABLE:

    @triton.jit
    def segmented_cumsum_kernel(
        data_ptr,  # *Pointer* to the flat data array
        seqlens_ptr,  # *Pointer* to the prefix-sums array of shape (S+1,)
        out_ptr,  # *Pointer* to the output array (same shape as data)
        S,  # number of segments
        BLOCK_SIZE: tl.constexpr,  # size of each block we scan
    ):
        pid = tl.program_id(0)  # which segment are we doing?
        if pid >= S:
            return

        # load segment boundaries
        start = tl.load(seqlens_ptr + pid)
        end = tl.load(seqlens_ptr + pid + 1)
        length = end - start

        # how many blocks this segment spans?
        n_blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE

        # running total from all previous blocks in this segment
        running = tl.zeros([], dtype=tl.float32)

        for block_idx in range(0, n_blocks):
            # compute the offsets within data for this block
            offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            idxs = start + offs
            # only load/store where offs < length
            mask = offs < length

            # load a block (padding out‐of‐bounds with 0)
            x = tl.load(data_ptr + idxs, mask=mask, other=0.0)

            # inclusive scan within the block
            scanned = tl.cumsum(x, 0)

            # add the running total from all previous blocks
            scanned = scanned + running

            # store back only the valid elements
            tl.store(out_ptr + idxs, scanned, mask=mask)

            # update running total to include the last element of this block
            # note: if the block was empty (mask all false) we leave running as is
            # last_idx = start + block_idx * BLOCK_SIZE + length % BLOCK_SIZE - 1
            if block_idx == n_blocks - 1:
                # final block may be shorter
                last_off = (length - 1) % BLOCK_SIZE
            else:
                last_off = BLOCK_SIZE - 1
            running = tl.load(out_ptr + start + block_idx * BLOCK_SIZE + last_off)


def segmented_cumsum_cuda(
    data: torch.Tensor, cu_seqlens: torch.Tensor, block_size: int = 1024
):
    """
    data:      (N,)  float32 on CUDA
    cu_seqlens:(S+1,) int64 on CUDA
    returns   (N,)  float32 on CUDA where each segment is cumulatively summed in-place
    """
    if not TRITON_AVAILABLE:
        raise ImportError(
            "triton is required for CUDA segmented_cumsum but is not installed"
        )
    assert data.is_cuda and cu_seqlens.is_cuda
    data.numel()
    S = cu_seqlens.numel() - 1
    out = torch.empty_like(data)
    grid = (S,)

    segmented_cumsum_kernel[grid](
        data,
        cu_seqlens.to(dtype=torch.int32),
        out,
        S,
        BLOCK_SIZE=block_size,
        num_warps=4,  # tune as needed
        num_stages=2,
    )
    return out


def segmented_cumsum_cpu(data: torch.Tensor, cu_seqlens: torch.Tensor):
    S = cu_seqlens.shape[0] - 1
    seg_cu_data = torch.concat(
        [torch.cumsum(data[cu_seqlens[i] : cu_seqlens[i + 1]], dim=0) for i in range(S)]
    )
    return seg_cu_data


def segmented_cumsum(data: torch.Tensor, cu_seqlens: torch.Tensor):
    if data.is_cpu:
        return segmented_cumsum_cpu(data, cu_seqlens)
    else:
        return segmented_cumsum_cuda(data, cu_seqlens)


def segmented_sum(data: torch.Tensor, seqlens: torch.Tensor):
    m = seqlens.shape[0]
    device = data.device
    dtype = data.dtype

    # Assign each position in data to its segment index
    segment_ids = torch.repeat_interleave(torch.arange(m, device=device), seqlens)

    # Count True values per segment
    seq_sums: torch.Tensor = torch.zeros(m, dtype=dtype, device=device)
    seq_sums.scatter_add_(0, segment_ids, data)

    return seq_sums


def approximate_ellipse(polys: gpd.array.GeometryArray):
    rects = shapely.minimum_rotated_rectangle(polys)
    xy = np.stack([np.stack(rect.exterior.coords.xy, axis=1)[:-1] for rect in rects])

    c = xy.mean(axis=1)
    v0 = xy[:, 1] - xy[:, 0]

    # lengths of adjacent edges
    w = np.linalg.norm(v0, axis=-1)  # one side length
    h = np.linalg.norm(xy[:, 2] - xy[:, 1], axis=-1)  # other side length
    theta = np.arctan2(v0[:, 1], v0[:, 0])

    a = w / 2.0
    b = h / 2.0

    S = np.empty((xy.shape[0], 2, 2))
    S[:, 0, 0] = a * np.cos(theta)
    S[:, 0, 1] = -b * np.sin(theta)
    S[:, 1, 0] = a * np.sin(theta)
    S[:, 1, 1] = b * np.cos(theta)

    R = np.empty((xy.shape[0], 2, 2))
    R[:, 0, 0] = np.cos(theta)
    R[:, 0, 1] = -np.sin(theta)
    R[:, 1, 0] = np.sin(theta)
    R[:, 1, 1] = np.cos(theta)

    D = np.zeros((xy.shape[0], 2, 2))
    D[:, 0, 0] = np.square(1.0 / a)
    D[:, 1, 1] = np.square(1.0 / b)
    return (S, R, D, c)


def rect_grid_indices(
    c: torch.Tensor, w: torch.Tensor, h: torch.Tensor, L: float, M: int
):
    """
    c: (n,2) centers
    w,h: (n,) half-width/height
    L: defines size of grid [-L,L] x [-L, L]
    M: defines resolution of grid
    Returns:
      idx: (total_points, 2) tensor of (i,j) indices
      seqlens: (n,) valid points per rectangle
    """

    n = c.shape[0]
    c0, c1 = c[:, 0], c[:, 1]

    # bounds
    ax, bx = (c0 - w) / L, (c0 + w) / L
    ay, by = (c1 - h) / L, (c1 + h) / L
    u = (M - 1 + M * ax) / 2.0
    v = (M - 1 + M * bx) / 2.0
    uy = (M - 1 + M * ay) / 2.0
    vy = (M - 1 + M * by) / 2.0

    i0 = torch.clamp(torch.ceil(u).to(torch.long), 0, M - 1)
    i1 = torch.clamp(torch.floor(v).to(torch.long), 0, M - 1)
    j0 = torch.clamp(torch.ceil(uy).to(torch.long), 0, M - 1)
    j1 = torch.clamp(torch.floor(vy).to(torch.long), 0, M - 1)

    ni = (i1 - i0 + 1).clamp(min=0)
    nj = (j1 - j0 + 1).clamp(min=0)
    seqlens = ni * nj
    total = seqlens.sum()

    # prefix sums
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0), value=0)

    # build i indices
    all_i = torch.arange(total, device=c.device)
    # which rectangle each entry belongs to
    rect_ids = torch.repeat_interleave(torch.arange(n, device=c.device), seqlens)

    # offset within rectangle
    offs = all_i - cu_seqlens[rect_ids]

    # decompose offsets into (i_offset, j_offset)
    i_offs = offs // nj[rect_ids]
    j_offs = offs % nj[rect_ids]

    ii = i0[rect_ids] + i_offs
    jj = j0[rect_ids] + j_offs

    idx = torch.stack([ii, jj], dim=1)  # (total,2)

    return idx, seqlens


def ellipse_grid_indices(S: torch.Tensor, c: torch.Tensor, L: float, M: int):
    """
    S: (n, 2, 2)
    c: (n,2)
    ------ Ellipse defined as {x : (x-c)^T (S S^T)^{-1} (x-c) <= 1} -------
    L: defines size of grid [-L,L] x [-L, L]
    M: defines resolution of grid
    Returns:
      idx: (n, k, 2) tensor of indices (padded with -1)
    """
    device = S.device

    ## Get axis-aligned bounding box rectangles per ellipse
    half_widths = torch.linalg.vector_norm(S, dim=-1)
    bbox_idx, bbox_seqlens = rect_grid_indices(
        c=c, w=half_widths[:, 0], h=half_widths[:, 1], L=L, M=M
    )
    S_inv = torch.linalg.inv(S)
    S_inv_rep = torch.repeat_interleave(S_inv, bbox_seqlens, dim=0)
    c_rep = torch.repeat_interleave(c, bbox_seqlens, dim=0)

    grid1d = L * (2 * torch.arange(M, device=device) + 1 - M) / M
    grid2d_locs = torch.stack((grid1d[bbox_idx[:, 0]], grid1d[bbox_idx[:, 1]]), dim=-1)
    grid2d_locs_proj = torch.einsum("bij, bj -> bi", S_inv_rep, (grid2d_locs - c_rep))
    locs_norms: torch.Tensor = torch.linalg.vector_norm(grid2d_locs_proj, dim=-1)
    ellipse_mask = locs_norms <= 1

    ellipse_seqlens = segmented_sum(ellipse_mask.to(torch.long), bbox_seqlens)
    ellipse_coordinates = bbox_idx[ellipse_mask]

    return (ellipse_coordinates, ellipse_seqlens)
