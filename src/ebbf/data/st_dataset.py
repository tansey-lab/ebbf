import concurrent.futures as cf
import logging
import os
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import shapely
from scipy.sparse import coo_array, csr_array
from shapely.geometry import Polygon
from tqdm import tqdm

from ebbf.compat import get_num_cpus
from ebbf.data.datasets import NuclearPatchDataset
from ebbf.data.utils import approximate_ellipse

logger = logging.getLogger(__name__)


def detect_input_format(input_dir: str) -> str:
    """
    Detect the input format of a directory.

    Returns:
        "xenium" if it's a Xenium Ranger output directory
        "spatialdata" if it's a SpatialData zarr directory
    """
    # Check for Xenium-specific files
    xenium_markers = [
        "transcripts.parquet",
        "nucleus_boundaries.parquet",
        "experiment.xenium",
    ]
    has_xenium_files = all(os.path.exists(os.path.join(input_dir, f)) for f in xenium_markers[:2])

    if has_xenium_files:
        return "xenium"

    # Check for SpatialData structure
    spatialdata_markers = [
        "points/transcripts",
        "shapes/nucleus_boundaries",
    ]
    has_spatialdata = all(os.path.exists(os.path.join(input_dir, p)) for p in spatialdata_markers)

    if has_spatialdata:
        return "spatialdata"

    raise ValueError(
        f"Could not detect input format for '{input_dir}'. "
        "Expected either Xenium output "
        "(transcripts.parquet, nucleus_boundaries.parquet) "
        "or SpatialData zarr (points/transcripts/, shapes/nucleus_boundaries/)."
    )


def _vertices_to_polygons(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert Xenium vertex-based boundary format to GeoDataFrame with Shapely polygons.

    Args:
        df: DataFrame with columns ['cell_id', 'vertex_x', 'vertex_y']

    Returns:
        GeoDataFrame with cell_id as index and geometry column containing polygons
    """
    polygons = []
    cell_ids = []

    for cell_id, group in df.groupby("cell_id"):
        coords = list(zip(group["vertex_x"], group["vertex_y"]))
        if len(coords) >= 3:
            # Close the polygon if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygons.append(Polygon(coords))
            cell_ids.append(cell_id)

    gdf = gpd.GeoDataFrame({"geometry": polygons}, index=cell_ids)
    gdf.index.name = "cell_id"
    return gdf


def _load_xenium_data(xenium_dir: str):
    """
    Load data from a Xenium Ranger output directory.

    Returns:
        tuple: (points_dataset, nucs_gdf, cells_gdf_or_none, column_mapping)
            - points_dataset: PyArrow dataset for transcripts
            - nucs_gdf: GeoDataFrame with nucleus polygons
            - cells_gdf_or_none: GeoDataFrame with cell polygons or None
            - column_mapping: dict mapping expected column names to actual names
    """
    transcripts_path = os.path.join(xenium_dir, "transcripts.parquet")
    nucs_path = os.path.join(xenium_dir, "nucleus_boundaries.parquet")
    cells_path = os.path.join(xenium_dir, "cell_boundaries.parquet")

    # Load nucleus boundaries and convert to polygons
    logger.info("Loading nucleus boundaries from Xenium format")
    nucs_df = pd.read_parquet(nucs_path)
    nucs_gdf = _vertices_to_polygons(nucs_df)

    # Load cell boundaries if present
    cells_gdf = None
    if os.path.exists(cells_path):
        logger.info("Loading cell boundaries from Xenium format")
        cells_df = pd.read_parquet(cells_path)
        cells_gdf = _vertices_to_polygons(cells_df)

    # Create points dataset
    points_dataset = ds.dataset(transcripts_path, format="parquet")

    # Xenium uses x_location/y_location instead of x/y
    column_mapping = {
        "x": "x_location",
        "y": "y_location",
    }

    return points_dataset, nucs_gdf, cells_gdf, column_mapping


def _load_spatialdata(zarr_dir: str):
    """
    Load data from a SpatialData zarr directory.

    Returns:
        tuple: (points_dataset, nucs_gdf, cells_gdf_or_none, column_mapping)
    """
    # Determine paths - handle both directory and file parquet formats
    points_path = os.path.join(zarr_dir, "points/transcripts/points.parquet")
    if os.path.isdir(points_path):
        points_folder = points_path
    else:
        # Try the parent directory (for partitioned parquet)
        points_folder = os.path.join(zarr_dir, "points/transcripts")

    nucs_fname = os.path.join(zarr_dir, "shapes/nucleus_boundaries/shapes.parquet")
    cells_fname = os.path.join(zarr_dir, "shapes/cell_boundaries/shapes.parquet")

    # Load nucleus boundaries
    nucs_gdf = gpd.read_parquet(nucs_fname)

    # Load cell boundaries if present
    cells_gdf = None
    if os.path.isfile(cells_fname):
        cells_gdf = gpd.read_parquet(cells_fname)

    # Create points dataset
    points_dataset = ds.dataset(points_folder, format="parquet")

    # SpatialData uses x/y directly
    column_mapping = {"x": "x", "y": "y"}

    return points_dataset, nucs_gdf, cells_gdf, column_mapping


# --- Globals set once per worker ---
_gene2idx = None
_cell_id2idx = None
_n_nucs = None
_n_genes = None


def _init_worker(gene2idx, cell_id2idx, n_nucs, n_genes):
    """Initializer, runs once per worker process."""
    global _gene2idx, _cell_id2idx, _n_nucs, _n_genes
    _gene2idx = gene2idx
    _cell_id2idx = cell_id2idx
    _n_nucs = n_nucs
    _n_genes = n_genes


def process_batch(batch):
    """Executed for each batch inside worker processes."""
    global _gene2idx, _cell_id2idx, _n_nucs, _n_genes

    genes = np.array(batch["feature_name"], dtype=object)
    cell_id = np.array(batch["cell_id"], dtype=object)

    ## Filter to nuclear tx
    overlaps_nucleus = np.array(batch["overlaps_nucleus"]) == 1
    nuc_genes = genes[overlaps_nucleus].astype(str)
    nuc_cell_ids = cell_id[overlaps_nucleus]

    ## Filter out weirdness
    mask = nuc_genes != "None"

    if not np.any(mask):
        logger.debug("EMPTY")
        num = len(batch["feature_name"])
        return {
            "local_nuc_gene_counts": csr_array((_n_nucs, _n_genes), dtype=np.float64),
            "num": num,
        }

    ## Use otypes=[object] so missing cell_ids (filtered-out nuclei) come back as
    ## None rather than tripping the dtype inference that np.vectorize defaults to.
    nuclear_tx_gene_idx = np.vectorize(_gene2idx.get, otypes=[object])(nuc_genes[mask])
    nuclear_tx_nuc_idx = np.vectorize(_cell_id2idx.get, otypes=[object])(nuc_cell_ids[mask])

    ## Drop transcripts assigned to nuclei that were filtered out upstream
    ## (e.g. centroid outside the tissue polygon).
    kept = nuclear_tx_nuc_idx != None  # noqa: E711
    nuclear_tx_gene_idx = nuclear_tx_gene_idx[kept].astype(np.int64)
    nuclear_tx_nuc_idx = nuclear_tx_nuc_idx[kept].astype(np.int64)

    local_nuc_gene_counts = coo_array(
        (np.ones_like(nuclear_tx_gene_idx), (nuclear_tx_nuc_idx, nuclear_tx_gene_idx)),
        shape=(_n_nucs, _n_genes),
    )

    num = len(batch["feature_name"])

    return {
        "local_nuc_gene_counts": local_nuc_gene_counts.tocsr(),
        "num": num,
    }


def _compute_gene_counts(
    points_dataset: ds.Dataset,
    cell_id2idx: dict[str, int],
    gene2idx: dict[str, int],
    batch_size: int = 10_000_000,
    num_workers: int | None = None,
):
    """Compute per-nucleus gene count matrix from transcript data."""
    n_rows = points_dataset.count_rows(filter=(ds.field("is_gene")))

    if num_workers is None:
        num_workers = get_num_cpus() // 2

    n_nucs = len(cell_id2idx)
    n_genes = len(gene2idx)

    # Create a generator for streaming batches
    def batch_generator():
        batches = points_dataset.to_batches(
            batch_size=batch_size,
            columns=[
                "feature_name",
                "is_gene",
                "cell_id",
                "overlaps_nucleus",
            ],
            filter=(ds.field("is_gene")),
        )
        yield from batches

    # --- Main driver ---
    nuc_gene_counts = csr_array((n_nucs, n_genes), dtype=np.int64)

    with cf.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(gene2idx, cell_id2idx, n_nucs, n_genes),
    ) as executor:
        pbar = tqdm(total=n_rows)
        for res in executor.map(process_batch, batch_generator(), chunksize=1):
            nuc_gene_counts = nuc_gene_counts + res["local_nuc_gene_counts"]
            pbar.update(res["num"])

    return nuc_gene_counts


def _compute_boxes_from_bounds(
    min_x: float, min_y: float, max_x: float, max_y: float, box_width: float
):
    """Compute a grid of boxes covering the given bounding box."""
    L_x = max_x - min_x
    L_y = max_y - min_y

    n_x = max(1, int(L_x // box_width + 1))
    n_y = max(1, int(L_y // box_width + 1))

    if n_x == 1:
        a_x = np.array([min_x])
    else:
        a_x = min_x + np.arange(n_x) * (L_x - box_width) / (n_x - 1)

    if n_y == 1:
        a_y = np.array([min_y])
    else:
        a_y = min_y + np.arange(n_y) * (L_y - box_width) / (n_y - 1)

    boxes = np.array(
        [(xmin, ymin, xmin + box_width, ymin + box_width) for xmin, ymin in product(a_x, a_y)]
    )
    return boxes


def _compute_polygon_triangulation(polygons):
    all_triangles = shapely.constrained_delaunay_triangles(polygons)

    all_triangle_xy = []
    all_triangle_probs = []
    for triangles in tqdm(all_triangles):
        triangle_xy = np.stack(
            [
                np.unique(np.stack(triangle.exterior.xy, axis=-1), axis=0)
                for triangle in triangles.geoms
            ]
        )
        all_triangle_xy.append(triangle_xy)
        triangle_areas = np.array([triangle.area for triangle in triangles.geoms])
        triangle_probs = triangle_areas / np.sum(triangle_areas)
        all_triangle_probs.append(triangle_probs)
    return (all_triangle_xy, all_triangle_probs)


def load_nucleus_centroids(zarr_dir: str, input_format: str | None = None) -> np.ndarray:
    """Load just the nucleus shapes from a vendor input dir and return centroids
    as an (N, 2) array. Used by the CLI to derive a tissue mask via DBSCAN before
    calling :func:`build_stdataset`.
    """
    if input_format is None:
        input_format = detect_input_format(zarr_dir)
    elif input_format not in ("spatialdata", "xenium"):
        raise ValueError(f"Unknown input_format: {input_format}")

    if input_format == "xenium":
        _, nucs, _, _ = _load_xenium_data(zarr_dir)
    else:
        _, nucs, _, _ = _load_spatialdata(zarr_dir)
    _, _, _, c = approximate_ellipse(nucs.geometry.values)
    return c


def build_stdataset(
    zarr_dir: str,
    box_width: float = 400,
    nuc_density_per_um2: float = 1e-4,
    input_format: str | None = None,
    tissue_polygon=None,
):
    """
    Build an STDataset from a vendor input directory.

    Supports both SpatialData zarr directories and Xenium Ranger output directories.
    Tiles each tissue region (one per polygon of ``tissue_polygon``, analogous to the
    TMA punches in nuc_diffusion) with overlapping boxes and keeps any box that meets
    the minimum nucleus density threshold. All nuclei are kept.

    Args:
        zarr_dir: Path to input directory. Can be either:
            - SpatialData zarr directory with points/transcripts/ and shapes/
            - Xenium Ranger output directory with transcripts.parquet,
              nucleus_boundaries.parquet, etc.
        box_width: Width of spatial tiles
        nuc_density_per_um2: Minimum nucleus density to keep a tile (default: 1e-4)
        input_format: Input format, either "spatialdata" or "xenium".
            If None, will auto-detect based on directory contents.
        tissue_polygon: Optional shapely (Multi)Polygon. Each of its parts is tiled
            separately. If None, the bounding box of all nuclei is tiled.
    """
    num_workers = get_num_cpus()

    # Detect or validate input format
    if input_format is None:
        input_format = detect_input_format(zarr_dir)
        logger.info("Detected input format: %s", input_format)
    elif input_format not in ("spatialdata", "xenium"):
        raise ValueError(f"Unknown input_format: {input_format}")

    # Load data based on format
    if input_format == "xenium":
        points_dataset, nucs, cells, column_mapping = _load_xenium_data(zarr_dir)
    else:
        points_dataset, nucs, cells, column_mapping = _load_spatialdata(zarr_dir)

    # Get points folder path for polars operations
    if input_format == "xenium":
        points_folder = os.path.join(zarr_dir, "transcripts.parquet")
    else:
        points_path = os.path.join(zarr_dir, "points/transcripts/points.parquet")
        if os.path.isdir(points_path):
            points_folder = points_path
        else:
            points_folder = os.path.join(zarr_dir, "points/transcripts")

    ## Read gene panel from transcripts
    df = pl.scan_parquet(points_folder)
    df = df.filter(pl.col("is_gene"))
    df = df.select(["feature_name"]).unique().collect().to_pandas()
    panel = np.unique(df["feature_name"].values)
    gene2idx = {x: i for i, x in enumerate(panel)}

    ## Process nucleus shapes
    nuc_cell_id = nucs.index.values
    S, _, _, c = approximate_ellipse(nucs.geometry.values)

    cell_id2idx = {x: i for i, x in enumerate(nuc_cell_id)}

    ## Process cell shapes (optional)
    if cells is not None:
        logger.info("Computing cell polygons")
        polygons = shapely.make_valid(cells.loc[nuc_cell_id]["geometry"].values)
        cell_triangle_xy, cell_triangle_probs = _compute_polygon_triangulation(polygons)
        cell_triangle_num = np.array([p.shape[0] for p in cell_triangle_probs])
        cell_triangle_xy = np.concatenate(cell_triangle_xy, axis=0)
        cell_triangle_probs = np.concatenate(cell_triangle_probs, axis=0)
    else:
        cell_triangle_xy, cell_triangle_probs = None, None
        cell_triangle_num = None

    ## Count transcripts per cell
    df = pl.scan_parquet(points_folder)
    df = df.filter((pl.col("cell_id") != "UNASSIGNED") & pl.col("is_gene"))
    cell_counts = df.group_by("cell_id").len().collect().to_pandas()
    nuc_cell_counts = np.zeros(nuc_cell_id.shape[0], dtype=np.int64)
    for cid, count in zip(cell_counts["cell_id"].values, cell_counts["len"].values):
        if cid in cell_id2idx:
            nuc_cell_counts[cell_id2idx[cid]] = count

    logger.info("Computing gene count matrix")
    nuc_gene_counts: csr_array = _compute_gene_counts(
        points_dataset,
        cell_id2idx,
        gene2idx,
        num_workers=(num_workers // 2),
    )

    ## Tile each tissue region with boxes (one region per polygon part)
    if tissue_polygon is not None:
        regions = shapely.get_parts(tissue_polygon)
    else:
        regions = [shapely.box(*c.min(axis=0), *c.max(axis=0))]
    boxes = np.concatenate(
        [_compute_boxes_from_bounds(*region.bounds, box_width) for region in regions]
    )

    ## Filter out sparse tiles based on nucleus density
    nuc_count_thresh = nuc_density_per_um2 * (box_width**2)
    n_nucs_per_box = np.array(
        [
            np.count_nonzero(
                (c[:, 0] > box[0]) & (c[:, 1] > box[1]) & (c[:, 0] < box[2]) & (c[:, 1] < box[3])
            )
            for box in boxes
        ]
    )
    boxes = boxes[n_nucs_per_box >= nuc_count_thresh]
    logger.info(
        "Kept %d tiles with >= %.1f nuclei (density=%s)",
        len(boxes),
        nuc_count_thresh,
        nuc_density_per_um2,
    )

    ## Finalize stats
    nuc_total_tx_counts = nuc_gene_counts.sum(axis=1)
    nuc_gene_counts: coo_array = nuc_gene_counts.tocoo()

    stdata = STDataset(
        boxes=boxes,
        nuc_cell_id=nuc_cell_id,
        nuc_total_tx_counts=nuc_total_tx_counts,
        nuc_cell_counts=nuc_cell_counts,
        nuc_xy=c,
        nuc_ellipse_mat=S,
        nuc_countmat_nuc_idx=nuc_gene_counts.row,
        nuc_countmat_gene_idx=nuc_gene_counts.col,
        nuc_countmat_counts=nuc_gene_counts.data,
        genes=panel,
        cell_triangle_num=cell_triangle_num,
        cell_triangle_xy=cell_triangle_xy,
        cell_triangle_probs=cell_triangle_probs,
    )

    return stdata, points_folder, column_mapping


class STDataset:
    def __init__(
        self,
        boxes: np.ndarray,
        nuc_cell_id: np.ndarray,
        nuc_total_tx_counts: np.ndarray,
        nuc_cell_counts: np.ndarray,
        nuc_xy: np.ndarray,
        nuc_ellipse_mat: np.ndarray,
        nuc_countmat_nuc_idx: np.ndarray,
        nuc_countmat_gene_idx: np.ndarray,
        nuc_countmat_counts: np.ndarray,
        genes: np.ndarray,
        cell_triangle_num: list[np.ndarray] | None = None,
        cell_triangle_xy: list[np.ndarray] | None = None,
        cell_triangle_probs: list[np.ndarray] | None = None,
        **kwargs,
    ):

        self.boxes = boxes
        self.nuc_cell_id = nuc_cell_id
        self.nuc_total_tx_counts = nuc_total_tx_counts
        self.nuc_xy = nuc_xy
        self.nuc_ellipse_mat = nuc_ellipse_mat
        self.nuc_countmat_nuc_idx = nuc_countmat_nuc_idx
        self.nuc_countmat_gene_idx = nuc_countmat_gene_idx
        self.nuc_countmat_counts = nuc_countmat_counts
        self.genes = genes
        self.nuc_cell_counts = nuc_cell_counts

        if cell_triangle_xy is None:
            self.cell_triangle_xy = np.array([])
            self.cell_triangle_probs = np.array([])
            self.cell_triangle_num = np.array([])
        else:
            self.cell_triangle_num = cell_triangle_num
            self.cell_triangle_xy = cell_triangle_xy
            self.cell_triangle_probs = cell_triangle_probs

        self.__dict__.update(kwargs)

    def save(self, fname: str):
        # Don't let np.savez coerce None into an object scalar
        data = {key: val for key, val in self.__dict__.items() if val is not None}
        np.savez_compressed(fname, **data)

    @classmethod
    def load(cls, fname: str):
        res = np.load(fname, allow_pickle=True)
        return cls(**res)

    def get_nuc_dataset(
        self, patch_size: float, patch_padding: float, dataset_size: int, min_nucs: int
    ):

        n_nucs = self.nuc_xy.shape[0]
        n_genes = self.genes.shape[0]

        nuc2genes = csr_array(
            (
                self.nuc_countmat_counts,
                (self.nuc_countmat_nuc_idx, self.nuc_countmat_gene_idx),
            ),
            shape=(n_nucs, n_genes),
        )
        nuc_total_sums = nuc2genes.sum(axis=1)[:, np.newaxis]
        nuc2gene_probs = (nuc2genes / nuc_total_sums).tocsr()

        dataset = NuclearPatchDataset(
            nuc2xy=self.nuc_xy,
            nuc2total_gene_count=self.nuc_total_tx_counts,
            nuc2gene_probs=nuc2gene_probs,
            nuc2cell_count=self.nuc_cell_counts,
            punch_bounds=self.boxes,
            patch_size=patch_size,
            patch_padding=patch_padding,
            dataset_size=dataset_size,
            min_nucs=min_nucs,
            nuc2ellipse_mat=self.nuc_ellipse_mat,
            cell_triangle_num=(
                self.cell_triangle_num if self.cell_triangle_num.shape[0] > 0 else None
            ),
            cell_triangle_xy=(
                self.cell_triangle_xy if self.cell_triangle_xy.shape[0] > 0 else None
            ),
            cell_triangle_probs=(
                self.cell_triangle_probs if self.cell_triangle_probs.shape[0] > 0 else None
            ),
        )
        return dataset
