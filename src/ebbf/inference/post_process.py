import argparse
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import uuid

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ebbf.compat import get_num_cpus
from ebbf.data.st_dataset import STDataset

logger = logging.getLogger(__name__)


def remove(path):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError(f"file {path} is not a file or dir.")


class DBSCANWorker(mp.Process):
    def __init__(self, in_q: mp.Queue, out_q: mp.Queue, epsilon: float, output_path: str):

        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.epsilon = epsilon
        self.output_path = output_path

    def run(self):
        # Construct once, persists for lifetime of process
        dbscan = DBSCAN(eps=self.epsilon, min_samples=1)

        while True:
            fragment: ds.ParquetFileFragment = self.in_q.get()

            if fragment is None:
                break

            table = fragment.to_table()
            x = np.array(table["x"], dtype=np.float64)
            y = np.array(table["y"], dtype=np.float64)
            xy = np.stack([x, y], axis=-1)
            cluster_id = dbscan.fit_predict(xy)

            cluster_field = pa.field("cluster_id", pa.int64())
            output_table = table.append_column(cluster_field, pa.array(cluster_id))

            path_names = fragment.path.split("/")
            out_fname = os.path.join(self.output_path, *path_names[2:])
            out_folder = os.path.join(self.output_path, *path_names[2:-1])
            os.makedirs(out_folder, exist_ok=True)
            pq.write_table(output_table, out_fname)

            self.out_q.put(out_fname)


###############
### STEP 1 ###
##############
def filter_unique_and_remap_tx(input_path: str | list[str], output_path: str, idx2genes: dict):

    df = pl.scan_parquet(input_path)

    ## Get unique values
    min_vals = df.group_by("tx_idx").agg(pl.min("tx_dist"))
    df = df.join(min_vals, how="inner", on=["tx_idx", "tx_dist"])

    ## Rename to be consistent with everything else
    df = df.rename(
        {
            "tx_idx": "transcript_id",
            "tx_x": "x",
            "tx_y": "y",
            "tx_initial_x": "x_initial",
            "tx_initial_y": "y_initial",
        }
    )

    ## Replace gene indices with gene names
    df = df.with_columns(
        pl.col("tx_gene_idx").replace_strict(idx2genes, default="None").alias("feature_name")
    )

    ## Ship it
    df.sink_parquet(output_path)
    return "Success!"


###############
###############


###############
### STEP 2 ###
##############
def group_labels_min_size(x, s):
    """
    Assign group labels so each group of x accumulates at least s total
    before starting the next group. Order is preserved.

    Returns an array of shape (n,) of integer group labels.
    """
    n = len(x)
    labels = np.empty(n, dtype=int)

    g = 0  # current group label
    running = 0  # running group sum

    for i, v in enumerate(x):
        labels[i] = g
        running += v
        if running >= s:
            g += 1
            running = 0

    return labels


def _add_grid_columns(df: pl.DataFrame, box_width: float, offset: float) -> pl.DataFrame:
    """Add grid_x, grid_y, and grid_name columns to a DataFrame."""
    return df.with_columns(
        ((pl.col("x") / box_width) + offset).floor().cast(pl.Int32).alias("grid_x"),
        ((pl.col("y") / box_width) + offset).floor().cast(pl.Int32).alias("grid_y"),
    ).with_columns(
        (pl.col("grid_x").cast(pl.String) + "_" + pl.col("grid_y").cast(pl.String)).alias(
            "grid_name"
        )
    )


def double_grid_and_group(
    input_path: str,
    output_path: str,
    cluster_distance: float,
    box_width: float | None = 50.0,
    min_size: int | None = 25_000,
):

    ## Ensure box_width is large enough
    box_width = max(box_width, 2 * cluster_distance)
    os.makedirs(output_path, exist_ok=True)

    ## Stage part files in a temp directory inside output_path;
    ## TemporaryDirectory ensures cleanup on error.
    with tempfile.TemporaryDirectory(dir=output_path, prefix=".staging_") as staging_dir:
        for grid_version, offset in [(0, 0.0), (1, 0.5)]:
            ## Phase 1: compute grid_name -> super_grid_label mapping (small aggregation)
            ldf = pl.scan_parquet(input_path)
            grid_ldf = ldf.with_columns(
                ((pl.col("x") / box_width) + offset).floor().cast(pl.Int32).alias("grid_x"),
                ((pl.col("y") / box_width) + offset).floor().cast(pl.Int32).alias("grid_y"),
            ).with_columns(
                (pl.col("grid_x").cast(pl.String) + "_" + pl.col("grid_y").cast(pl.String)).alias(
                    "grid_name"
                ),
            )
            counts = (
                grid_ldf.group_by("grid_name")
                .len()
                .collect(engine="streaming")
                .to_pandas()
                .sort_values(by="len")
            )
            counts["label"] = group_labels_min_size(counts["len"].values, min_size)
            mapping_df = pl.DataFrame(
                {
                    "grid_name": counts["grid_name"].to_list(),
                    "super_grid_label": counts["label"].to_list(),
                }
            )

            ## Phase 2: stream input in batches, write per-partition chunks to staging
            file_counts: dict[int, int] = {}
            parquet_file = pq.ParquetFile(input_path)
            for record_batch in parquet_file.iter_batches():
                batch = pl.from_arrow(record_batch)

                batch = _add_grid_columns(batch, box_width, offset)
                batch = batch.join(mapping_df, on="grid_name", how="left")
                batch = batch.with_columns(
                    pl.lit(grid_version).cast(pl.Int32).alias("grid_version"),
                )

                for group_df in batch.partition_by("super_grid_label"):
                    label = group_df["super_grid_label"][0]
                    part_dir = os.path.join(staging_dir, f"gv{grid_version}_sgl{label}")
                    os.makedirs(part_dir, exist_ok=True)
                    idx = file_counts.get(label, 0)
                    group_df.write_parquet(os.path.join(part_dir, f"part-{idx}.parquet"))
                    file_counts[label] = idx + 1

            ## Phase 3: consolidate each partition into a single file in output_path
            ## (Step 3 runs DBSCAN per fragment, so each partition must be one file)
            for label in sorted(file_counts.keys()):
                staging_part_dir = os.path.join(staging_dir, f"gv{grid_version}_sgl{label}")
                final_part_dir = os.path.join(
                    output_path,
                    f"grid_version={grid_version}",
                    f"super_grid_label={label}",
                )
                os.makedirs(final_part_dir, exist_ok=True)
                final_path = os.path.join(final_part_dir, "data.parquet")
                n_parts = file_counts[label]
                if n_parts == 1:
                    shutil.move(os.path.join(staging_part_dir, "part-0.parquet"), final_path)
                else:
                    part_files = [
                        os.path.join(staging_part_dir, f"part-{i}.parquet") for i in range(n_parts)
                    ]
                    pl.scan_parquet(part_files).sink_parquet(final_path)

    return "Success!"


###############
###############


###############
### STEP 3 ###
##############


def grid_level_cluster(input_path: str, output_path: str, epsilon: float, n_workers: int):

    ## Read in the parquet dataset
    source_dataset = ds.dataset(input_path, format="parquet")
    fragments = list(source_dataset.get_fragments())
    n_frags = len(fragments)

    ## Spin up the parallel processing context
    ctx = mp.get_context("spawn")
    in_q, out_q = ctx.Queue(), ctx.Queue()

    ## Spin up the workers
    workers = [DBSCANWorker(in_q, out_q, epsilon, output_path) for _ in range(n_workers)]
    for w in workers:
        w.start()

    ## Place all the data in the queue
    for fragment in fragments:
        in_q.put(fragment)

    ## Place Nones in the queue to spin down the children
    for _ in workers:
        in_q.put(None)

    ## Extract outputs from the queue
    with tqdm(total=n_frags) as pbar:
        for _ in range(n_frags):
            out_q.get()  # block until a task finishes
            pbar.update(1)

    ## Spin down the workers
    for w in workers:
        w.join()

    return "Success!"


###############
###############


###############
### STEP 4 ###
##############


def grid_connected_components(input_path: str, output_path: str):

    df = pl.scan_parquet(input_path)
    df = df.with_columns(
        (
            pl.col("super_grid_label").cast(pl.String)
            + "_"
            + pl.col("grid_version").cast(pl.String)
            + "_"
            + pl.col("cluster_id").cast(pl.String)
        ).alias("full_cluster_id")
    )
    df0 = (
        df.filter(pl.col("grid_version") == 0)
        .select(["transcript_id", "full_cluster_id"])
        .rename({"full_cluster_id": "full_cluster_id_0"})
    )
    df1 = (
        df.filter(pl.col("grid_version") == 1)
        .select(["transcript_id", "full_cluster_id"])
        .rename({"full_cluster_id": "full_cluster_id_1"})
    )
    edges = (
        (df0.join(df1, on="transcript_id"))
        .select(["full_cluster_id_0", "full_cluster_id_1"])
        .unique()
        .lazy()
    )
    edges = edges.collect(engine="streaming").to_pandas()

    unique_cluster_id_0, idx_0 = np.unique(edges["full_cluster_id_0"], return_inverse=True)
    unique_cluster_id_1, idx_1 = np.unique(edges["full_cluster_id_1"], return_inverse=True)
    n_0 = unique_cluster_id_0.shape[0]
    idx_1 = idx_1 + n_0

    unique_cluster_id = np.concatenate([unique_cluster_id_0, unique_cluster_id_1])
    n = unique_cluster_id.shape[0]
    n_edges = edges.shape[0]
    csgraph = coo_array((np.ones(n_edges), (idx_0, idx_1)), shape=(n, n))
    _, labels = connected_components(csgraph, directed=False)
    cluster_id2final_label = dict(zip(unique_cluster_id, labels))

    df_mapped = df.filter(pl.col("grid_version") == 0).with_columns(
        pl.col("full_cluster_id").replace_strict(cluster_id2final_label).alias("ef_cluster_id")
    )
    df_mapped.sink_parquet(output_path)
    return "Success!"


###############
###############


###############
### STEP 4 ###
##############


def map_clusters_to_cells(
    input_path: str,
    output_path: str,
    nuc_centroid: np.ndarray,
    nuc_ellipse_mat: np.ndarray,
    nuc_cell_id: np.ndarray,
):

    ## Build tree
    tree = KDTree(nuc_centroid)

    ## Read in data
    df = pl.scan_parquet(input_path)

    ## Get means of clusters
    cluster_df = (
        df.group_by(["ef_cluster_id"])
        .agg(pl.col("x").mean(), pl.col("y").mean())
        .collect(engine="streaming")
        .to_pandas()
    )
    cluster_id = cluster_df["ef_cluster_id"].values
    cluster_xy = cluster_df[["x", "y"]].values

    ## Look up closest nucleus centroid
    centroid_distance, centroid_idx = tree.query(cluster_xy)
    centroid_cell_id = nuc_cell_id[centroid_idx]

    ## Compute mahalanobis distance of cluster to closest centroid ellipse
    diff = cluster_xy - nuc_centroid[centroid_idx]
    Q = np.linalg.inv(np.matmul(nuc_ellipse_mat, nuc_ellipse_mat.mT))
    mahalanobis_distance = np.sqrt(np.einsum("...i,...ij,...j->...", diff, Q[centroid_idx], diff))

    ## Map everything back onto the lazyframe
    cluster_id2centroid_cell_id = dict(zip(cluster_id, centroid_cell_id))
    cluster_id2centroid_distance = dict(zip(cluster_id, centroid_distance))
    cluster_id2mahalanobis_distance = dict(zip(cluster_id, mahalanobis_distance))

    df = df.with_columns(
        pl.col("ef_cluster_id").replace_strict(cluster_id2centroid_cell_id).alias("cell_id")
    )
    df = df.with_columns(
        pl.col("ef_cluster_id")
        .replace_strict(cluster_id2centroid_distance)
        .alias("centroid_distance")
    )
    df = df.with_columns(
        pl.col("ef_cluster_id")
        .replace_strict(cluster_id2mahalanobis_distance)
        .alias("mahalanobis_distance")
    )

    ## Filter out the columns to just what we want
    df = df.select(
        [
            "transcript_id",
            "feature_name",
            "cell_id",
            "x",
            "y",
            "x_initial",
            "y_initial",
            "centroid_distance",
            "mahalanobis_distance",
            "ef_cluster_id",
        ]
    )

    ## Write it out
    df.sink_parquet(output_path)

    return "Success!"


###############
###############


###############
### STEP 6 ###
##############


def calculate_tx_distances(
    input_path: str,
    output_path: str,
    original_tx_fname: str,
):
    """Enrich transcript table with overlaps_nucleus and distance to original position.

    Args:
        input_path: Path to the cluster-to-cell mapped parquet file
        output_path: Path to write the enriched parquet file
        original_tx_fname: Path to the original transcripts parquet file
    """
    ## Read in data
    df = pl.scan_parquet(input_path)

    ## Read in original tx data for overlaps_nucleus
    tx_df = pl.scan_parquet(original_tx_fname)
    tx_df = tx_df.select(["transcript_id", "overlaps_nucleus"])

    ## Merge current data with original
    df = df.with_columns(pl.col("transcript_id").cast(pl.UInt64))
    df = df.join(tx_df, on="transcript_id", how="inner")

    ## Compute distance between diffused and original positions
    df = df.with_columns(
        distance=(
            (pl.col("x") - pl.col("x_initial")).pow(2) + (pl.col("y") - pl.col("y_initial")).pow(2)
        ).sqrt()
    )

    ## Write it out
    df.sink_parquet(output_path)
    return "Success!"


###############
###############


###############
### STEP 7 ###
##############


def _create_anndata_from_lazyframe(
    ldf: pl.LazyFrame,
    cell_id2idx: dict,
    cell_id_col: str = "cell_id",
    gene_col: str = "feature_name",
):
    """Build an AnnData count matrix from a transcript LazyFrame.

    Args:
        ldf: LazyFrame with transcript-level data
        cell_id2idx: Mapping from cell identifiers to row indices
        cell_id_col: Column name for cell identifiers
        gene_col: Column name for gene/feature names
    """
    cell_ids = list(cell_id2idx.keys())
    sub_ldf = ldf.filter(pl.col(cell_id_col).is_in(cell_ids))

    X = sub_ldf.group_by([cell_id_col, gene_col]).len(name="count").collect().to_pandas()

    row_idx = X[cell_id_col].map(cell_id2idx).values
    row_mask = ~np.isnan(row_idx)

    genes = X[gene_col].unique()
    gene2idx = {x: i for i, x in enumerate(genes)}
    col_idx = X[gene_col].map(gene2idx).values
    col_mask = ~np.isnan(col_idx)

    ## Filter out the nans
    mask = row_mask & col_mask
    row_idx = row_idx[mask].astype(int)
    col_idx = col_idx[mask].astype(int)

    counts = X["count"].values[mask]
    n_genes = max(gene2idx.values()) + 1
    n_cells = max(cell_id2idx.values()) + 1

    mat = csr_matrix((counts, (row_idx, col_idx)), shape=(n_cells, n_genes))

    ## Genes
    idx2gene = {v: k for k, v in gene2idx.items()}
    var = pd.DataFrame(index=np.arange(n_genes))
    var["feature_name"] = var.index.map(idx2gene)
    var = var.set_index("feature_name")

    ## Cell ids
    idx2cell_id = {v: k for k, v in cell_id2idx.items()}
    obs = pd.DataFrame(index=np.arange(n_cells))
    obs["cell_id"] = obs.index.map(idx2cell_id)
    obs = obs.set_index("cell_id")

    ## Create anndata object
    adata = anndata.AnnData(X=mat, var=var, obs=obs)
    return adata


def create_anndata(
    ef_tx_fname: str,
    output_fname: str,
    dataset: STDataset,
    cd_cutoff: float = 2.0,
):
    """Create AnnData file from transcript assignment table.

    Cells are classified into three categories:
    1. Original cells: mahalanobis_distance <= 1 and centroid_distance <= cd_cutoff
    2. Split cells: mahalanobis_distance <= 1 and centroid_distance > cd_cutoff
    3. De novo cells: mahalanobis_distance > 1

    Args:
        ef_tx_fname: Path to the transcript assignment parquet file
        output_fname: Path to output h5ad file
        dataset: STDataset with nucleus spatial data
        cd_cutoff: Centroid distance cutoff for cell classification
    """
    ## Get relevant spatial locations
    cell_id2x_centroid = dict(zip(dataset.nuc_cell_id, dataset.nuc_xy[:, 0]))
    cell_id2y_centroid = dict(zip(dataset.nuc_cell_id, dataset.nuc_xy[:, 1]))

    ## Read in data
    ldf = pl.scan_parquet(ef_tx_fname)

    # Classify each ef_cluster_id into a cell category at the cluster level
    cluster_info = (
        ldf.select(
            [
                "ef_cluster_id",
                "cell_id",
                "x",
                "y",
                "centroid_distance",
                "mahalanobis_distance",
            ]
        )
        .unique(subset=["ef_cluster_id"])
        .collect()
    )

    # Classify clusters
    cluster_info = cluster_info.with_columns(
        [
            pl.when(pl.col("mahalanobis_distance") > 1)
            .then(pl.lit("denovo"))
            .when(pl.col("centroid_distance") > cd_cutoff)
            .then(pl.lit("split"))
            .otherwise(pl.lit("original"))
            .alias("cell_type")
        ]
    )

    # Assign unique cell identifiers
    # Original: use cell_id directly (multiple clusters merge)
    # Split: "{cell_id}_split_{ef_cluster_id}"
    # De novo: "denovo_{ef_cluster_id}"
    cluster_info = cluster_info.with_columns(
        [
            pl.when(pl.col("cell_type") == "original")
            .then(pl.col("cell_id").cast(pl.Utf8))
            .when(pl.col("cell_type") == "split")
            .then(
                pl.col("cell_id").cast(pl.Utf8)
                + pl.lit("_split_")
                + pl.col("ef_cluster_id").cast(pl.Utf8)
            )
            .otherwise(pl.lit("denovo_") + pl.col("ef_cluster_id").cast(pl.Utf8))
            .alias("unified_cell_id")
        ]
    )

    # Build mapping from ef_cluster_id -> unified_cell_id
    cluster_to_cell = dict(
        zip(
            cluster_info["ef_cluster_id"].to_list(),
            cluster_info["unified_cell_id"].to_list(),
        )
    )

    # Build spatial and metadata mappings per unified_cell_id
    cell_id_to_xy = {}
    cell_id_to_type_label = {}
    cell_id_to_parent = {}

    for row in cluster_info.iter_rows(named=True):
        uid = row["unified_cell_id"]
        ctype = row["cell_type"]

        if ctype == "original":
            orig_id = row["cell_id"]
            cell_id_to_xy[uid] = (
                cell_id2x_centroid[orig_id],
                cell_id2y_centroid[orig_id],
            )
            cell_id_to_parent[uid] = str(orig_id)
        elif ctype == "split":
            cell_id_to_xy[uid] = (row["x"], row["y"])
            cell_id_to_parent[uid] = str(row["cell_id"])
        else:  # denovo
            cell_id_to_xy[uid] = (row["x"], row["y"])
            cell_id_to_parent[uid] = None

        cell_id_to_type_label[uid] = ctype

    # Annotate transcripts with unified_cell_id via ef_cluster_id join
    cluster_mapping_df = pl.DataFrame(
        {
            "ef_cluster_id": list(cluster_to_cell.keys()),
            "unified_cell_id": list(cluster_to_cell.values()),
        }
    )

    annotated_ldf = ldf.join(
        cluster_mapping_df.lazy(),
        on="ef_cluster_id",
        how="inner",
    )

    # Build cell_id2idx ensuring all original cells are included
    all_cell_ids = sorted(
        set(cell_id_to_xy.keys()) | set(str(k) for k in cell_id2x_centroid.keys())
    )
    cell_id2idx = {cid: i for i, cid in enumerate(all_cell_ids)}

    # Populate spatial info and metadata for empty original cells
    for orig_id in cell_id2x_centroid.keys():
        uid = str(orig_id)
        if uid not in cell_id_to_xy:
            cell_id_to_xy[uid] = (
                cell_id2x_centroid[orig_id],
                cell_id2y_centroid[orig_id],
            )
            cell_id_to_type_label[uid] = "original"
            cell_id_to_parent[uid] = uid

    # Build the count matrix
    adata = _create_anndata_from_lazyframe(
        annotated_ldf,
        cell_id2idx=cell_id2idx,
        cell_id_col="unified_cell_id",
        gene_col="feature_name",
    )

    # Add spatial coordinates and metadata to obs
    adata.obs["cell_category"] = adata.obs.index.map(cell_id_to_type_label)
    adata.obs["parent_cell_id"] = adata.obs.index.map(cell_id_to_parent)
    adata.obs["x_centroid"] = adata.obs.index.map(lambda uid: cell_id_to_xy[uid][0])
    adata.obs["y_centroid"] = adata.obs.index.map(lambda uid: cell_id_to_xy[uid][1])
    adata.obs["n_tx"] = np.array(adata.X.sum(axis=1)).flatten().astype(int)

    # Store spatial coordinates in obsm for compatibility with spatial analysis tools
    adata.obsm["spatial"] = np.column_stack(
        [
            adata.obs["x_centroid"].values.astype(float),
            adata.obs["y_centroid"].values.astype(float),
        ]
    )

    adata.write(output_fname, compression="lzf")
    return adata


###############
###############


def post_process(
    input_files: list[str],
    dataset: STDataset,
    cluster_distance: float,
    output_fname: str = "ef.parquet",
    anndata_fname: str | None = "ef.h5ad",
    cd_cutoff: float = 2.0,
    original_tx_fname: str | None = None,
    work_dir: str = ".",
    n_workers: int | None = None,
):
    ## Get number of parallel workers
    if n_workers is None:
        n_workers = get_num_cpus()

    ## Ensure work_dir exists
    os.makedirs(work_dir, exist_ok=True)

    ## Get mappings for genes
    idx2genes = {i: x for i, x in enumerate(dataset.genes)}

    ## Step 1: Make tx unique
    logger.info("Step 1: getting unique tx...")
    temp_fname1 = "temp1-" + uuid.uuid4().hex + ".parquet"
    temp_path1 = os.path.join(work_dir, temp_fname1)
    filter_unique_and_remap_tx(input_path=input_files, output_path=temp_path1, idx2genes=idx2genes)

    ## Step 2: Build double grid
    logger.info("Step 2: building double grid...")
    temp_fname2 = "temp2-" + uuid.uuid4().hex
    temp_path2 = os.path.join(work_dir, temp_fname2)
    double_grid_and_group(
        input_path=temp_path1, output_path=temp_path2, cluster_distance=cluster_distance
    )

    ## Remove first intermediate file
    remove(temp_path1)

    ## Step 3: Cluster within each grid section
    logger.info("Step 3: clustering within grid sections...")
    temp_fname3 = "temp3-" + uuid.uuid4().hex
    temp_path3 = os.path.join(work_dir, temp_fname3)
    grid_level_cluster(
        input_path=temp_path2,
        output_path=temp_path3,
        epsilon=cluster_distance,
        n_workers=n_workers,
    )

    ## Remove second intermediate file
    remove(temp_path2)

    ## Step 4: Link clusters across grid versions
    logger.info("Step 4: linking clusters across grids...")
    temp_fname4 = "temp4-" + uuid.uuid4().hex + ".parquet"
    temp_path4 = os.path.join(work_dir, temp_fname4)
    grid_connected_components(input_path=temp_path3, output_path=temp_path4)

    ## Remove third intermediate file
    remove(temp_path3)

    ## Step 5: Map clusters to cells
    logger.info("Step 5: mapping clusters to cells...")
    temp_fname5 = "temp5-" + uuid.uuid4().hex + ".parquet"
    temp_path5 = os.path.join(work_dir, temp_fname5)
    map_clusters_to_cells(
        input_path=temp_path4,
        output_path=temp_path5,
        nuc_centroid=dataset.nuc_xy,
        nuc_ellipse_mat=dataset.nuc_ellipse_mat,
        nuc_cell_id=dataset.nuc_cell_id,
    )

    ## Remove fourth intermediate file
    remove(temp_path4)

    ## Step 6: Compute distances to original positions
    if original_tx_fname is not None:
        logger.info("Step 6: computing distances to original positions...")
        calculate_tx_distances(
            input_path=temp_path5,
            output_path=output_fname,
            original_tx_fname=original_tx_fname,
        )
        ## Remove fifth intermediate file
        remove(temp_path5)
    else:
        logger.info("Step 6: skipping distance computation (no original transcripts provided)...")
        os.rename(temp_path5, output_fname)

    ## Step 7: Create AnnData file
    if anndata_fname is not None:
        logger.info("Step 7: creating AnnData file...")
        create_anndata(
            ef_tx_fname=output_fname,
            output_fname=anndata_fname,
            dataset=dataset,
            cd_cutoff=cd_cutoff,
        )

    logger.info("DONE!")

    return "Success!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input parquet files from inference",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to stdata.npz file")
    parser.add_argument(
        "--cluster_distance",
        type=float,
        default=1.5,
        help="Linkage distance for clustering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ef.parquet",
        help="Output file path (default: ef.parquet)",
    )
    parser.add_argument(
        "--original_transcripts",
        type=str,
        default=None,
        help="Path to original transcripts parquet file for distance computation",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Directory for temporary files (default: current directory)",
    )

    args = parser.parse_args()
    stdata = STDataset.load(args.dataset)

    post_process(
        input_files=args.input_files,
        work_dir=args.work_dir,
        dataset=stdata,
        cluster_distance=args.cluster_distance,
        output_fname=args.output,
        original_tx_fname=args.original_transcripts,
    )
