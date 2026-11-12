import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

from ebbf.data.utils import segmented_sum
from ebbf.model.unet import DhariwalUNet


class GridEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        img_resolution: int,
        img_range: float,
        n_channels: int,
        splat: bool = False,
    ):
        super().__init__()

        self.img_range = img_range
        self.splat = splat
        self.grid_size = img_resolution
        self.embed_table = nn.Embedding(n_items, n_channels)
        torch.nn.init.kaiming_uniform_(self.embed_table.weight)

    def nearest_grid_encode(
        self,
        xy: torch.Tensor,  ## S x 2
        X: torch.Tensor,  ## S x K
        seqlens: torch.Tensor,  ## B
    ):
        device = xy.device
        M = self.grid_size
        L = self.img_range
        if xy.dtype.is_floating_point:  ## Need to discretize
            grid_pos = (0.5 * M * (xy + L) / L).long()
            grid_pos = grid_pos.clip(min=0, max=(M - 1))
        else:
            grid_pos = xy

        dtype = X.dtype
        B = seqlens.shape[0]
        r = X.shape[1]

        # flat index within a M*M grid for each element
        i = grid_pos[:, 0]
        j = grid_pos[:, 1]
        flat_idx = i * M + j  # shape: (S,)

        # build batch index per element using
        batch_idx = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(seqlens)

        # make global index across batches to scatter into single flattened axis
        global_flat_idx = batch_idx * (M * M) + flat_idx  # shape: (S,)

        # prepare output container of shape (r, B*M*M) then reshape
        out = torch.zeros(r, B * M * M, dtype=dtype, device=device)

        # scatter_add: add X.T (r x S) into out along dim=1 at positions global_flat_idx
        out.scatter_add_(1, global_flat_idx.unsqueeze(0).expand(r, -1), X.T)

        # reshape to (r, B, M, M) and permute to (B, r, M, M)
        out = out.view(r, B, M, M).permute(1, 0, 2, 3).contiguous()
        return out

    def splat_grid_encode(
        self,
        xy: torch.Tensor,  ## S x 2
        X: torch.Tensor,  ## S x K
        seqlens: torch.Tensor,  ## B
    ):
        device = xy.device
        M = self.grid_size
        L = self.img_range

        dtype = X.dtype
        B = seqlens.shape[0]
        k = X.shape[1]

        # map to fractional grid coords
        grid_coords = (xy + L) * (M - 1) / (2 * L)
        gx, gy = grid_coords[:, 0], grid_coords[:, 1]

        ix0 = gx.floor().long()
        iy0 = gy.floor().long()
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        wx = gx - ix0.float()
        wy = gy - iy0.float()

        w00 = (1 - wx) * (1 - wy)
        w10 = wx * (1 - wy)
        w01 = (1 - wx) * wy
        w11 = wx * wy

        # batch indices per point
        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seqlens)

        # flatten output grid
        out = torch.zeros(B * M * M, k, device=device, dtype=dtype)

        def scatter_add_neighbor(ix, iy, w):
            lin_idx = batch_idx * (M * M) + iy * M + ix
            contrib = w[:, None] * X  # (n, k)
            out.scatter_add_(0, lin_idx[:, None].expand(-1, k), contrib)

        scatter_add_neighbor(ix0, iy0, w00)
        scatter_add_neighbor(ix1, iy0, w10)
        scatter_add_neighbor(ix0, iy1, w01)
        scatter_add_neighbor(ix1, iy1, w11)

        # reshape back
        out = out.view(B, M, M, k).permute(0, 3, 2, 1).contiguous()
        return out

    def forward(
        self,
        xy: torch.Tensor,  ## S x 2
        items: torch.Tensor,  ## S
        seqlens: torch.Tensor,  ## B
    ):

        X = self.embed_table(items)
        if self.splat:
            out = self.splat_grid_encode(xy, X, seqlens)
        else:
            out = self.nearest_grid_encode(xy, X, seqlens)

        return out


class GridDecoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        img_resolution: int,
        img_range: float,
        n_channels: int,
        n_outputs: int = 2,
    ):
        super().__init__()

        self.img_range = img_range
        self.grid_size = img_resolution
        self.n_outputs = n_outputs
        self.embed_table = nn.Embedding(n_items, n_channels)
        torch.nn.init.kaiming_uniform_(self.embed_table.weight)

    def splat_grid_decode(
        self,
        X: torch.Tensor,  ## B x C x K x K
        xy: torch.Tensor,  ## S x 2
        seqlens: torch.Tensor,  ## B
    ):
        L = self.img_range
        B, k, M, _ = X.shape
        device = X.device

        # cumulative seqlens -> batch ids
        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seqlens)

        # map to fractional grid coords
        grid_coords = (xy + L) * (M - 1) / (2 * L)
        gx, gy = grid_coords[:, 0], grid_coords[:, 1]

        ix0 = gx.floor().long()
        iy0 = gy.floor().long()
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        wx = (gx - ix0.float()).unsqueeze(-1)
        wy = (gy - iy0.float()).unsqueeze(-1)

        # gather corner values
        def gather(ix, iy):
            return X[batch_idx, :, ix, iy]  # (n,k)

        v00 = gather(ix0, iy0)
        v10 = gather(ix1, iy0)
        v01 = gather(ix0, iy1)
        v11 = gather(ix1, iy1)

        # bilinear interpolation
        out = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11
        return out

    def forward(
        self,
        X: torch.Tensor,  ## B x C x K x K
        xy: torch.Tensor,  ## S x 2
        items: torch.Tensor,  ## S
        seqlens: torch.Tensor,  ## B
    ):
        grid_emb = self.splat_grid_decode(X, xy, seqlens)
        grid_emb = rearrange(grid_emb, "S (O C)-> S O C", O=self.n_outputs)  ## S x O x C

        ## Take dot products against embeddings
        item_emb = self.embed_table(items).unsqueeze(-2)  ## S x 1 x C
        x = torch.sum(grid_emb * item_emb, dim=-1)  ## S x O
        return x

    def vector_field(
        self,
        X: torch.Tensor,  ## B x C x K x K
        items: torch.Tensor,  ## S
        restrict_vector: bool = True,
    ):
        """Per-item output field on the full grid: B x S x K x K x O (O=2 if restrict_vector)."""
        X_emb = rearrange(X, "B (O C) M N-> B O C M N", O=self.n_outputs)  ## B x O x C x K x K
        if restrict_vector:
            X_emb = X_emb[:, :2, ...]  ## B x 2 x C x K x K

        item_emb = self.embed_table(items)  ## S x C
        return torch.einsum("btcij,sc->bsijt", X_emb, item_emb)


class TxFlowUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.in_channels = config.model.in_channels
        self.out_channels = config.model.out_channels

        self.config = config
        self.padding_bound: float = 1.0 + (2.0 * config.data.patch_padding / config.data.patch_size)
        self.img_resolution: int = config.model.img_resolution

        self.unet = DhariwalUNet(config)
        self.pred_noise: bool = config.model.get("pred_noise", True)

        self.nuc_condition = config.model.conditioned == "nuc"
        if self.nuc_condition:
            self.nuc_channel_frac = config.model.nuc_channel_frac
            nuc_in_channels = int(self.nuc_channel_frac * self.in_channels)
            tx_in_channels = self.in_channels - nuc_in_channels
            ## A single learned embedding shared by all nuclei
            self.nuc_ellipse_encoder = GridEncoder(
                1,
                img_resolution=self.img_resolution,
                img_range=self.padding_bound,
                n_channels=nuc_in_channels,
            )
        else:
            tx_in_channels = self.in_channels

        self.tx_grid_encoder = GridEncoder(
            config.n_genes,
            img_resolution=self.img_resolution,
            img_range=self.padding_bound,
            n_channels=tx_in_channels,
            splat=config.model.get("splat", True),
        )

        n_outputs = 3 if self.pred_noise else 2
        self.tx_grid_decoder = GridDecoder(
            config.n_genes,
            img_resolution=self.img_resolution,
            img_range=self.padding_bound,
            n_channels=(self.out_channels // n_outputs),
            n_outputs=n_outputs,
        )

        self.eps = 1e-4

    def forward(
        self,
        gene_idx: torch.Tensor,  ## L
        t: torch.Tensor,  ## B
        tx_pos: torch.Tensor,  ## L x 2
        tx_seqlens: torch.Tensor | None,
        nuc_ellipse_pos: torch.Tensor | None = None,  ## E x 2
        nuc_ellipse_seqlens: torch.Tensor | None = None,  ## N
        nuc_seqlens: torch.Tensor | None = None,  ## B
    ):
        ## encode and pass through backbone (B x out_channels x K x K)
        x = self.encode_and_backbone(
            tx_pos=tx_pos,
            t=t,
            gene_idx=gene_idx,
            tx_seqlens=tx_seqlens,
            nuc_ellipse_pos=nuc_ellipse_pos,
            nuc_ellipse_seqlens=nuc_ellipse_seqlens,
            nuc_seqlens=nuc_seqlens,
        )

        ## Extract relevant per-transcript vectors
        xy = tx_pos.clamp(min=(self.eps - self.padding_bound), max=(self.padding_bound - self.eps))
        return self.tx_grid_decoder(x, xy=xy, items=gene_idx, seqlens=tx_seqlens)

    def encode_inputs(
        self,
        tx_pos: torch.Tensor,
        gene_idx: torch.Tensor,
        tx_seqlens: torch.Tensor,
        nuc_ellipse_pos: torch.Tensor | None = None,
        nuc_ellipse_seqlens: torch.Tensor | None = None,
        nuc_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rasterize transcripts (+ nuclei) to the B x in_channels x K x K input image."""
        xy = tx_pos.clamp(min=(self.eps - self.padding_bound), max=(self.padding_bound - self.eps))
        x = self.tx_grid_encoder(xy=xy, items=gene_idx, seqlens=tx_seqlens)

        if self.nuc_condition:
            batch_nuc_ellipse_seqlens = segmented_sum(nuc_ellipse_seqlens, nuc_seqlens)
            nuc_items = torch.zeros(
                nuc_ellipse_pos.shape[0], dtype=torch.long, device=nuc_ellipse_pos.device
            )
            nuc_x = self.nuc_ellipse_encoder(
                xy=nuc_ellipse_pos, items=nuc_items, seqlens=batch_nuc_ellipse_seqlens
            )
            x = torch.concat((x, nuc_x), dim=1)
        return x

    def encode_and_backbone(
        self,
        tx_pos,
        t,
        gene_idx,
        tx_seqlens,
        nuc_ellipse_pos=None,
        nuc_ellipse_seqlens=None,
        nuc_seqlens=None,
    ) -> torch.Tensor:
        x = self.encode_inputs(
            tx_pos=tx_pos,
            gene_idx=gene_idx,
            tx_seqlens=tx_seqlens,
            nuc_ellipse_pos=nuc_ellipse_pos,
            nuc_ellipse_seqlens=nuc_ellipse_seqlens,
            nuc_seqlens=nuc_seqlens,
        )
        return self.unet(x, t)

    def bottleneck(
        self,
        gene_idx,
        tx_pos,
        tx_seqlens,
        t=None,
        nuc_ellipse_pos=None,
        nuc_ellipse_seqlens=None,
        nuc_seqlens=None,
        include_mid: bool = True,
        pool: str = "none",
    ) -> torch.Tensor:
        """Bottleneck features. pool='none' -> B x C_b x m x m;
        'mean' -> B x C_b; 'meanmax' -> B x 2*C_b."""
        if t is None:
            t = torch.zeros(tx_seqlens.shape[0], device=tx_pos.device, dtype=torch.float32)
        x = self.encode_inputs(
            tx_pos=tx_pos,
            gene_idx=gene_idx,
            tx_seqlens=tx_seqlens,
            nuc_ellipse_pos=nuc_ellipse_pos,
            nuc_ellipse_seqlens=nuc_ellipse_seqlens,
            nuc_seqlens=nuc_seqlens,
        )
        h = self.unet.encode(x, t, include_mid=include_mid)
        if pool == "mean":
            h = h.mean(dim=(-2, -1))
        elif pool == "meanmax":
            h = torch.cat([h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))], dim=1)
        return h
