#!/usr/bin/env python
"""Two-panel ROI figure comparing EBBF transcript clustering to the Xenium IF stain.

Panel 1      Whole-slide DAPI at low resolution with the ROI boxed, to orient the
             reader (skip with --no-overview).
Panel 2      EBBF transcripts at their original positions, colored by cluster, each
             linked to its cluster centroid by a spoke ("spiderweb"). Centroids are
             marked: star = de novo cluster, circle = original-segmentation cluster.
             Cluster colors come from a proper graph coloring of a kNN graph over the
             centroids, so no two neighbouring cells ever share a color.
Panel 3      All 4 morphology_focus IF channels merged into one additive color
             composite, with the original Xenium cell/nucleus boundaries outlined.

Usage
-----
    # command line
    python ebbf_roi_plot.py --roi 2357.5 2451.0 4938.0 5034.0 -o my_roi

    # or from python / a notebook
    from ebbf.inference.roi_plot import plot_roi, Config
    fig = plot_roi((2357.5, 2451.0, 4938.0, 5034.0))
    fig = plot_roi(roi, Config(knn_k=12, show_spokes=False))

Coordinate systems -- two traps worth knowing about, both handled here:

1. `x`/`y` in ef.parquet are the EBBF post-diffusion positions, which are collapsed
   onto the cluster center (per-cluster sigma ~0.03 um). Plotting them gives one dot
   per cluster, not a spatial map. The original transcript positions are
   `x_initial`/`y_initial`. This module uses x_initial/y_initial for the transcripts
   and the mean of x/y for the centroids.

2. The micron<->pixel mapping for morphology_focus is just `pixel_size` from
   experiment.xenium; there is no offset or affine anywhere in the dataset. Verified
   by overlaying nucleus_boundaries.parquet on DAPI -- the polygons land exactly on
   the nuclei. Use `check_alignment()` to re-run that check on a new dataset.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import tifffile
import zarr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

ROI = tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax) in microns


@dataclass
class Config:
    """Everything tunable. Defaults reproduce the reference figure."""

    # --- inputs ---
    ef: str = "ebbf_results/ef.parquet"
    # ef.unfiltered.h5ad, not ef.h5ad: the filtered file drops cells that failed the
    # transcript-count filter, so it is missing the denovo flag for ~25% of clusters.
    ef_h5ad: str = "ebbf_results/ef.unfiltered.h5ad"
    xenium: str = "xenium_dataset"
    # Path (relative to `xenium`) of a morphology_focus OME-TIFF. Any ch000N file works
    # -- the OME series exposes all 4 channels regardless. Leave as None to auto-locate
    # the file under `<xenium>/morphology_focus/` (its exact name varies between Xenium
    # Ranger versions, e.g. morphology_focus_0000.ome.tif or ch0000_dapi.ome.tif).
    if_tif: str | None = None

    # --- axis direction (applied via set_xlim/set_ylim, so panels stay registered) ---
    y_down: bool = False  # False -> y increases upward
    x_flip: bool = True  # True  -> x increases leftward

    # --- IF composite ---
    # additive color per channel, in channel order
    # (DAPI, ATP1A1/CD45/E-Cadherin, 18S, alphaSMA/Vimentin)
    ch_colors: Sequence[tuple] = (
        (0.15, 0.35, 1.00),  # blue
        (1.00, 0.20, 0.20),  # red
        (0.20, 1.00, 0.25),  # green
        (1.00, 0.25, 1.00),  # magenta
    )
    # high low-percentile floor: summing 4 channels saturates to white very fast
    ch_percentiles: tuple[float, float] = (40.0, 99.7)
    ch_gamma: float = 0.85

    # --- cluster coloring ---
    knn_k: int = 8
    palette: Sequence[tuple] = (
        (0.90, 0.10, 0.10),
        (0.10, 0.35, 0.90),
        (0.10, 0.65, 0.20),
        (0.95, 0.55, 0.00),
        (0.60, 0.20, 0.75),
        (0.00, 0.70, 0.70),
        (0.85, 0.35, 0.65),
        (0.55, 0.40, 0.15),
        (0.40, 0.75, 0.10),
        (0.20, 0.20, 0.55),
        (0.95, 0.75, 0.10),
        (0.55, 0.55, 0.60),
    )

    # --- whole-slide orientation panel (DAPI, from the image pyramid) ---
    show_overview: bool = True
    overview_level: int | None = None  # None -> coarsest level >= overview_min_px
    overview_min_px: int = 1200
    overview_percentiles: tuple[float, float] = (50.0, 99.9)
    roi_box_color: str = "red"

    # --- layers ---
    show_spokes: bool = True
    show_boundaries: bool = True
    boundary_style: dict = field(
        default_factory=lambda: {
            "cell_boundaries.parquet": ("white", 0.7, "cell boundary"),
            "nucleus_boundaries.parquet": ("yellow", 0.6, "nucleus boundary"),
        }
    )

    # --- cosmetics (scaled for a ~100 um ROI; shrink for larger windows) ---
    tx_size: float = 4.0
    noise_size: float = 3.0
    spoke_width: float = 0.25
    spoke_alpha: float = 0.35
    panel_height: float = 7.0  # inches; panel widths follow from the aspect ratios
    figsize: tuple[float, float] | None = None  # set to override the derived size
    dpi: int = 250

    def path(self, *parts: str) -> str:
        return "/".join([self.xenium, *parts])

    @property
    def pixel_size(self) -> float:
        """Microns per pixel, from experiment.xenium."""
        with open(self.path("experiment.xenium")) as fh:
            return json.load(fh)["pixel_size"]

    @property
    def if_tif_path(self) -> str:
        """Absolute path to a morphology_focus OME-TIFF, auto-located if unset.

        Xenium Ranger writes the morphology stack under `morphology_focus/` but the
        file name has changed across versions, so fall back to a glob rather than a
        hard-coded name. Any single channel file works -- the OME series exposes all
        4 channels regardless.
        """
        if self.if_tif is not None:
            return self.path(self.if_tif)
        candidates = sorted(glob.glob(self.path("morphology_focus", "*.ome.tif")))
        if not candidates:
            # older layout: a single top-level morphology image
            candidates = sorted(glob.glob(self.path("morphology*.ome.tif")))
        if not candidates:
            raise FileNotFoundError(
                f"No morphology_focus OME-TIFF found under {self.path('morphology_focus')}"
            )
        return candidates[0]

    def has_if_tif(self) -> bool:
        """True if a morphology_focus OME-TIFF can be located under `xenium`."""
        try:
            return os.path.exists(self.if_tif_path)
        except FileNotFoundError:
            return False


# --------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------
def load_transcripts(roi: ROI, cfg: Config) -> pd.DataFrame:
    """EBBF transcripts whose ORIGINAL position falls inside the ROI."""
    xmin, xmax, ymin, ymax = roi
    flt = (
        (pc.field("x_initial") >= xmin)
        & (pc.field("x_initial") <= xmax)
        & (pc.field("y_initial") >= ymin)
        & (pc.field("y_initial") <= ymax)
    )
    cols = ["x_initial", "y_initial", "feature_name", "ef_cluster_id", "is_noise"]
    df = ds.dataset(cfg.ef, format="parquet").to_table(columns=cols, filter=flt).to_pandas()
    df["is_noise"] = df["is_noise"].fillna(False) | df["ef_cluster_id"].isna()
    return df


def cluster_centroids(cluster_ids: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Centroid of the POST-diffusion positions, plus the de novo flag.

    Averages over *every* transcript of each cluster, including ones outside the ROI,
    so clusters straddling the window edge are not pulled inward by truncation.
    """
    cen = (
        ds.dataset(cfg.ef, format="parquet")
        .to_table(
            columns=["x", "y", "ef_cluster_id"],
            filter=pc.is_in(
                pc.field("ef_cluster_id"), pa.array(np.asarray(cluster_ids, dtype="int64"))
            ),
        )
        .to_pandas()
        .groupby("ef_cluster_id")[["x", "y"]]
        .mean()
    )
    obs = ad.read_h5ad(cfg.ef_h5ad, backed="r").obs  # ef_cluster_id is unique per cell
    cen["cell_category"] = cen.index.map(obs.set_index("ef_cluster_id")["cell_category"])
    return cen


def load_if_stack(roi: ROI, cfg: Config):
    """Return (stack[C,H,W], channel_names, imshow_extent) for the ROI crop."""
    xmin, xmax, ymin, ymax = roi
    ps = cfg.pixel_size
    x0, x1 = int(xmin / ps), int(np.ceil(xmax / ps))
    y0, y1 = int(ymin / ps), int(np.ceil(ymax / ps))
    with tifffile.TiffFile(cfg.if_tif_path) as tf:
        arr = zarr.open(tf.series[0].levels[0].aszarr(), mode="r")
        if hasattr(arr, "array_keys"):  # pyramid group -> full-resolution level "0"
            arr = arr["0"]
        # multi-channel morphology is (C, H, W); a single-channel stain is (H, W)
        crop = arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[:, y0:y1, x0:x1]
        stack = np.asarray(crop)
        ome = tf.ome_metadata
    if stack.ndim == 2:  # promote single-channel to (1, H, W) so composite() can loop
        stack = stack[None]
    names = [s.split('Name="')[1].split('"')[0] for s in ome.split("<Channel ")[1:]]
    return stack, names, [x0 * ps, x1 * ps, y1 * ps, y0 * ps]


def load_overview(cfg: Config):
    """Whole-slide DAPI from the image pyramid, for the orientation panel.

    Picks the coarsest pyramid level still at least `overview_min_px` across, so this
    reads a few MB rather than the 64k x 42k full-resolution plane.
    """
    ps = cfg.pixel_size
    with tifffile.TiffFile(cfg.if_tif_path) as tf:
        levels = tf.series[0].levels
        full_h, full_w = levels[0].shape[-2:]
        lvl = cfg.overview_level
        if lvl is None:  # coarsest level that is still big enough to be legible
            big = [i for i, s in enumerate(levels) if max(s.shape[-2:]) >= cfg.overview_min_px]
            lvl = max(big) if big else 0
        arr = zarr.open(levels[lvl].aszarr(), mode="r")
        if hasattr(arr, "array_keys"):
            arr = arr[str(lvl)]
        img = np.asarray(arr[0] if arr.ndim == 3 else arr)  # DAPI only
    # scale by the true downsample factor rather than assuming powers of two
    extent = [0.0, full_w * ps, full_h * ps, 0.0]
    print(f"  overview: level {lvl} {img.shape}")
    return img, extent


def load_boundaries(fname: str, roi: ROI, cfg: Config) -> list[np.ndarray]:
    """Closed polygons with at least one vertex in the ROI, drawn in full.

    Selecting whole polygons (rather than clipping vertices) keeps cells that straddle
    the window edge from being cut mid-outline.
    """
    xmin, xmax, ymin, ymax = roi
    bd = ds.dataset(cfg.path(fname), format="parquet")
    inside = (
        bd.to_table(
            columns=["cell_id"],
            filter=(pc.field("vertex_x") >= xmin)
            & (pc.field("vertex_x") <= xmax)
            & (pc.field("vertex_y") >= ymin)
            & (pc.field("vertex_y") <= ymax),
        )
        .column("cell_id")
        .unique()
    )
    poly = bd.to_table(filter=pc.is_in(pc.field("cell_id"), inside)).to_pandas()
    segs = [g[["vertex_x", "vertex_y"]].to_numpy() for _, g in poly.groupby("cell_id")]
    return [np.vstack([p, p[:1]]) for p in segs if len(p) > 2]  # close each polygon


# --------------------------------------------------------------------------------
# data sources -- morphology image + Xenium boundaries, either straight from Xenium
# Ranger files or from an in-memory SpatialData object. Both expose the same
# micron-space outputs (micron = pixel * pixel_size), so `plot_roi` is format-agnostic.
# --------------------------------------------------------------------------------
def _boundary_element(name: str) -> str:
    """Map a boundary key ('cell_boundaries[.parquet]') to a shapes element name."""
    base = name.rsplit("/", 1)[-1]
    return base[: -len(".parquet")] if base.endswith(".parquet") else base


class XeniumRangerSource:
    """Reads the morphology OME-TIFF and boundary parquets from a Xenium dir."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def has_morphology(self) -> bool:
        return self.cfg.has_if_tif()

    @property
    def pixel_size(self) -> float:
        return self.cfg.pixel_size

    def if_stack(self, roi: ROI):
        return load_if_stack(roi, self.cfg)

    def overview(self):
        return load_overview(self.cfg)

    def boundaries(self, name: str, roi: ROI) -> list[np.ndarray]:
        fname = _boundary_element(name) + ".parquet"
        return load_boundaries(fname, roi, self.cfg)


class SpatialDataSource:
    """Serves the same crops/boundaries from an in-memory SpatialData object.

    Works for both Xenium Ranger dirs (loaded via ``spatialdata_io.xenium``) and
    written SpatialData zarr stores: the morphology image lives at
    ``images['morphology_focus']`` (multiscale, dims ``(c, y, x)``) with an identity
    transform to global pixels, and boundaries are micron-space GeoDataFrames whose
    transform to global is a pure ``Scale`` of ``1 / pixel_size``.
    """

    IMAGE_KEY = "morphology_focus"

    def __init__(self, sd, cfg: Config | None = None):
        self.sd = sd
        self.cfg = cfg or Config()

    def has_morphology(self) -> bool:
        return self.IMAGE_KEY in self.sd.images

    # --- geometry <-> pixel scale ---
    @staticmethod
    def _extract_scale(transform) -> float | None:
        from spatialdata.transformations import Scale, Sequence

        if isinstance(transform, Scale):
            return float(np.asarray(transform.scale).ravel()[0])
        if isinstance(transform, Sequence):
            for sub in transform.transformations:
                s = SpatialDataSource._extract_scale(sub)
                if s is not None:
                    return s
        return None

    @property
    def pixel_size(self) -> float:
        from spatialdata.transformations import get_transformation

        for key in ("cell_boundaries", "nucleus_boundaries"):
            if key in self.sd.shapes:
                scale = self._extract_scale(get_transformation(self.sd.shapes[key], "global"))
                if scale:
                    return 1.0 / scale
        raise ValueError("Could not derive pixel_size from SpatialData boundary transforms")

    # --- morphology image (multiscale DataTree or a single DataArray) ---
    def _levels(self):
        """Morphology levels finest-to-coarsest, each a (c, y, x) DataArray."""
        img = self.sd.images[self.IMAGE_KEY]
        if hasattr(img, "keys") and any(str(k).startswith("scale") for k in img.keys()):
            names = sorted(
                (str(k) for k in img.keys() if str(k).startswith("scale")),
                key=lambda s: int(s[5:]),
            )
            return [img[n]["image"] for n in names]
        return [img]

    def if_stack(self, roi: ROI):
        xmin, xmax, ymin, ymax = roi
        ps = self.pixel_size
        x0, x1 = int(xmin / ps), int(np.ceil(xmax / ps))
        y0, y1 = int(ymin / ps), int(np.ceil(ymax / ps))
        da = self._levels()[0]
        crop = da.isel(y=slice(y0, y1), x=slice(x0, x1))
        stack = np.asarray(crop.values)
        if stack.ndim == 2:  # promote single-channel to (1, H, W)
            stack = stack[None]
        names = [str(c) for c in np.atleast_1d(da.coords["c"].values)]
        return stack, names, [x0 * ps, x1 * ps, y1 * ps, y0 * ps]

    def overview(self):
        ps = self.pixel_size
        levels = self._levels()
        big = [da for da in levels if max(da.shape[-2:]) >= self.cfg.overview_min_px]
        chosen = big[-1] if big else levels[-1]  # coarsest level still legible
        img = np.asarray(chosen.isel(c=0).values)  # DAPI (channel 0)
        full_h, full_w = levels[0].shape[-2:]
        print(f"  overview: {img.shape}")
        return img, [0.0, full_w * ps, full_h * ps, 0.0]

    def boundaries(self, name: str, roi: ROI) -> list[np.ndarray]:
        xmin, xmax, ymin, ymax = roi
        gdf = self.sd.shapes[_boundary_element(name)]
        sub = gdf.cx[xmin:xmax, ymin:ymax]  # geometries intersecting the ROI box
        segs: list[np.ndarray] = []
        for geom in sub.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in parts:
                segs.append(np.asarray(poly.exterior.coords))
        return segs


def make_source(cfg: Config, sd=None):
    """Pick the right data source for `cfg.xenium` (or wrap a pre-loaded `sd`).

    A pre-loaded SpatialData object is used directly (this is what the reporting
    pipeline passes, so it never reads the Xenium dir twice). Otherwise a written
    SpatialData zarr is read via ``spatialdata.read_zarr`` and a Xenium Ranger dir
    is served straight from its OME-TIFF / boundary parquets.
    """
    from ebbf.data.formats import InputFormat, detect_input_format

    if sd is not None:
        return SpatialDataSource(sd, cfg)
    if detect_input_format(cfg.xenium) == InputFormat.SPATIALDATA:
        import spatialdata

        return SpatialDataSource(spatialdata.read_zarr(cfg.xenium), cfg)
    return XeniumRangerSource(cfg)


# --------------------------------------------------------------------------------
# cluster coloring
# --------------------------------------------------------------------------------
def graph_color(xy: np.ndarray, cfg: Config) -> np.ndarray:
    """Color indices such that no two kNN-adjacent cells share one (DSATUR).

    A random palette inevitably gives neighbouring cells the same hue. DSATUR -- color
    the uncolored node with the most distinctly-colored neighbours next, ties broken by
    degree -- needs far fewer colors than naive greedy (7 vs ~12 on a typical window).
    """
    k = min(cfg.knn_k, len(xy) - 1)
    _, nbr = cKDTree(xy).query(xy, k=k + 1)  # k+1: first hit is the point itself
    adj: list[set[int]] = [set() for _ in range(len(xy))]
    for i, row in enumerate(np.atleast_2d(nbr)):
        for j in row[1:]:
            adj[i].add(int(j))
            adj[int(j)].add(i)  # symmetrize: kNN is not a symmetric relation

    color_of = np.full(len(xy), -1)
    for _ in range(len(xy)):
        best, best_key = -1, None
        for v in np.flatnonzero(color_of < 0):
            key = (len({color_of[u] for u in adj[v] if color_of[u] >= 0}), len(adj[v]))
            if best_key is None or key > best_key:
                best, best_key = v, key
        used = {color_of[u] for u in adj[best] if color_of[u] >= 0}
        color_of[best] = next(c for c in range(len(xy)) if c not in used)

    bad = sum(color_of[i] == color_of[j] for i in range(len(xy)) for j in adj[i])
    assert bad == 0, f"{bad} adjacent same-color pairs"  # cheap guard against regressions
    print(f"  kNN k={k}: {color_of.max() + 1} colors, 0 adjacent same-color pairs")
    return color_of


def build_palette(n_colors: int, cfg: Config) -> np.ndarray:
    """The hand-picked palette, extended with evenly spaced hues if it runs out."""
    if n_colors <= len(cfg.palette):
        return np.asarray(cfg.palette)
    extra = n_colors - len(cfg.palette)
    return np.vstack(
        [
            cfg.palette,
            matplotlib.colors.hsv_to_rgb(
                np.column_stack(
                    [
                        np.linspace(0, 1, extra, endpoint=False),
                        np.full(extra, 0.85),
                        np.full(extra, 0.7),
                    ]
                )
            ),
        ]
    )


def composite(stack: np.ndarray, cfg: Config) -> np.ndarray:
    """Additive RGB merge, each channel contrast-stretched on its own first."""
    rgb = np.zeros(stack.shape[1:] + (3,), np.float32)
    for c in range(stack.shape[0]):
        img = stack[c].astype(np.float32)
        lo, hi = np.percentile(img, cfg.ch_percentiles)
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** cfg.ch_gamma
        rgb += img[..., None] * np.asarray(cfg.ch_colors[c], np.float32)
    return np.clip(rgb, 0, 1)


# --------------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------------
# centroid markers: (cell_category, marker, size, label); None matches missing category
CENTROID_MARKERS = [
    ("denovo", "*", 190, "de novo cluster"),
    ("original", "o", 55, "original cluster"),
    (None, "X", 70, "category unknown"),
]


def plot_roi(roi: ROI, cfg: Config | None = None, out: str | None = None, source=None):
    """Build the two-panel figure for `roi`. Returns the matplotlib Figure.

    `source` supplies the morphology image and Xenium boundaries; if omitted it is
    auto-selected from `cfg.xenium` (Xenium Ranger files or a SpatialData zarr).
    If `out` is given, writes <out>.png and <out>.pdf.
    """
    cfg = cfg or Config()
    source = source or make_source(cfg)
    xmin, xmax, ymin, ymax = roi

    df = load_transcripts(roi, cfg)
    noise = df["is_noise"].to_numpy(bool)
    sig = df[~noise]
    if sig.empty:
        raise ValueError(f"no clustered transcripts in ROI {roi}")
    clusters = np.sort(sig["ef_cluster_id"].unique())
    print(f"  {len(df):,} transcripts, {len(clusters):,} clusters, {noise.sum():,} noise")

    cen = cluster_centroids(clusters, cfg)
    color_of = graph_color(cen[["x", "y"]].to_numpy(), cfg)
    palette = build_palette(color_of.max() + 1, cfg)
    cen["color"] = list(palette[color_of])
    col = np.array([dict(zip(cen.index, palette[color_of]))[c] for c in sig["ef_cluster_id"]])

    stack, names, extent = source.if_stack(roi)

    # Panel widths follow the data aspect ratios so nothing is letterboxed: the two ROI
    # panels match the ROI shape, the overview matches the slide shape.
    ratios = [(xmax - xmin) / (ymax - ymin)] * 2
    if cfg.show_overview:
        ov_img, ov_extent = source.overview()
        ratios.insert(0, ov_extent[1] / ov_extent[2])
    figsize = cfg.figsize or (cfg.panel_height * sum(ratios) + 1.6, cfg.panel_height + 1.2)
    fig, all_axes = plt.subplots(
        1, len(ratios), figsize=figsize, gridspec_kw={"width_ratios": ratios}
    )
    axes = all_axes[1:] if cfg.show_overview else all_axes

    # --- optional: whole-slide DAPI with the ROI boxed ---
    if cfg.show_overview:
        ax = all_axes[0]
        lo, hi = np.percentile(ov_img, cfg.overview_percentiles)
        ax.imshow(
            np.clip((ov_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1),
            cmap="gray",
            extent=ov_extent,
            origin="upper",
            interpolation="bilinear",
        )
        # box, plus a marker so the ROI stays visible when it is only a few pixels wide
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=cfg.roi_box_color,
                lw=1.2,
                zorder=3,
            )
        )
        ax.plot(
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            marker="o",
            ms=22,
            mfc="none",
            mec=cfg.roi_box_color,
            mew=1.4,
            zorder=3,
        )
        ax.set_xlim(*(ov_extent[:2][::-1] if cfg.x_flip else ov_extent[:2]))
        ax.set_ylim(*(ov_extent[2:] if cfg.y_down else ov_extent[2:][::-1]))
        ax.set_aspect("equal")
        ax.set_title("whole slide (DAPI)\nred box = ROI")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    # --- left: transcripts, spokes, centroids ---
    ax = axes[0]
    ax.set_facecolor("white")
    handles = []
    if noise.any():
        ax.scatter(
            df.loc[noise, "x_initial"],
            df.loc[noise, "y_initial"],
            s=cfg.noise_size,
            c="0.75",
            lw=0,
            alpha=0.6,
        )
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                markersize=4,
                markerfacecolor="0.75",
                markeredgecolor="none",
                label=f"noise (n={noise.sum():,})",
            )
        )
    if cfg.show_spokes:
        spokes = np.stack(
            [
                sig[["x_initial", "y_initial"]].to_numpy(),
                cen.loc[sig["ef_cluster_id"], ["x", "y"]].to_numpy(),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(
                spokes, colors=col, zorder=1, linewidths=cfg.spoke_width, alpha=cfg.spoke_alpha
            )
        )
    ax.scatter(sig["x_initial"], sig["y_initial"], s=cfg.tx_size, c=col, lw=0, zorder=2)

    for cat, marker, size, label in CENTROID_MARKERS:
        m = cen["cell_category"].isna() if cat is None else (cen["cell_category"] == cat)
        if not m.any():
            continue
        ax.scatter(
            cen.loc[m, "x"],
            cen.loc[m, "y"],
            marker=marker,
            s=size,
            c=np.vstack(cen.loc[m, "color"]),
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        # neutral swatch: the plotted markers carry the per-cluster color, which would
        # otherwise make the legend look like it encodes one specific cluster
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker=marker,
                markersize=np.sqrt(size) * 0.9,
                markerfacecolor="0.55",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"{label} (n={int(m.sum())})",
            )
        )
    ax.set_title(f"EBBF transcript clusters\n{len(sig):,} transcripts, {len(clusters):,} clusters")
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)

    # --- right: IF composite + segmentation ---
    ax = axes[1]
    ax.imshow(composite(stack, cfg), extent=extent, origin="upper", interpolation="bilinear")
    ax.set_title("morphology_focus IF composite + Xenium segmentation")
    for i, (name, color) in enumerate(zip(names, cfg.ch_colors)):
        ax.text(
            0.015,
            0.975 - 0.045 * i,
            f"ch{i}  {name}",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
    if cfg.show_boundaries:
        for fname, (bcolor, blw, blabel) in cfg.boundary_style.items():
            segs = source.boundaries(fname, roi)
            ax.add_collection(
                LineCollection(segs, colors=bcolor, linewidths=blw, alpha=0.9, label=blabel)
            )
            print(f"  {fname}: {len(segs):,} polygons")
        leg = ax.legend(loc="lower left", fontsize=8, framealpha=0.55, facecolor="black")
        for txt in leg.get_texts():
            txt.set_color("white")

    for ax in axes:
        ax.set_xlim(*((xmax, xmin) if cfg.x_flip else (xmin, xmax)))
        ax.set_ylim(*((ymax, ymin) if cfg.y_down else (ymin, ymax)))
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(
        f"ROI x[{xmin:.1f}, {xmax:.1f}]  y[{ymin:.1f}, {ymax:.1f}] µm", fontsize=13, y=0.99
    )
    if out:
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=cfg.dpi, bbox_inches="tight")
        print(f"  wrote {out}.png / .pdf")
    return fig


def check_alignment(roi: ROI, cfg: Config | None = None, out: str = "alignment_check", source=None):
    """Sanity check the micron<->pixel mapping: nucleus outlines should sit on DAPI."""
    cfg = cfg or Config()
    source = source or make_source(cfg)
    stack, _, extent = source.if_stack(roi)
    lo, hi = np.percentile(stack[0], [2, 99.8])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        np.clip((stack[0] - lo) / max(hi - lo, 1e-6), 0, 1),
        cmap="gray",
        extent=extent,
        origin="upper",
    )
    ax.add_collection(
        LineCollection(source.boundaries("nucleus_boundaries", roi), colors="red", linewidths=0.7)
    )
    ax.set_xlim(*((roi[1], roi[0]) if cfg.x_flip else roi[:2]))
    ax.set_ylim(*((roi[3], roi[2]) if cfg.y_down else roi[2:]))
    ax.set_aspect("equal")
    ax.set_title("nucleus_boundaries over DAPI -- outlines should land on nuclei")
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    print(f"  wrote {out}.png")
    return fig


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--roi",
        nargs=4,
        type=float,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="ROI in microns",
    )
    p.add_argument(
        "-o",
        "--out",
        default="roi_clusters_vs_IF",
        help="output basename; writes <out>.png and <out>.pdf",
    )
    p.add_argument("--ef", help="path to ef.parquet")
    p.add_argument("--ef-h5ad", help="path to ef.unfiltered.h5ad")
    p.add_argument("--xenium", help="path to the Xenium output directory")
    p.add_argument("--if-tif", help="path (relative to --xenium) to a morphology_focus OME-TIFF")
    p.add_argument("--knn-k", type=int, help="neighbours per cell in the coloring graph")
    p.add_argument("--tx-size", type=float, help="transcript marker size")
    p.add_argument("--dpi", type=int, help="raster resolution")
    p.add_argument("--no-spokes", action="store_true", help="hide centroid spokes")
    p.add_argument("--no-boundaries", action="store_true", help="hide Xenium outlines")
    p.add_argument(
        "--no-overview", action="store_true", help="hide the whole-slide orientation panel"
    )
    p.add_argument(
        "--check-alignment",
        action="store_true",
        help="also write the nucleus-outline-over-DAPI sanity check",
    )
    a = p.parse_args()

    overrides = {
        k: v for k, v in vars(a).items() if v is not None and k in Config.__dataclass_fields__
    }
    cfg = Config(**overrides)
    cfg.show_spokes = not a.no_spokes
    cfg.show_boundaries = not a.no_boundaries
    cfg.show_overview = not a.no_overview

    roi = tuple(a.roi)
    print(f"ROI x[{roi[0]}, {roi[1]}] y[{roi[2]}, {roi[3]}] µm")
    plot_roi(roi, cfg, out=a.out)
    if a.check_alignment:
        check_alignment(roi, cfg)


if __name__ == "__main__":
    main()
