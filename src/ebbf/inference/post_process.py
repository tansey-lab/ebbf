import argparse
import multiprocessing as mp
import os
import shutil
import sys
import uuid

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ebbf.data.tma import TMA


def remove(path):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError(f"file {path} is not a file or dir.")


class DBSCANWorker(mp.Process):
    def __init__(
        self, in_q: mp.Queue, out_q: mp.Queue, epsilon: float, output_path: str
    ):

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
def filter_unique_and_remap_tx(input_path: str, output_path: str, idx2genes: dict):

    df = pl.scan_parquet(input_path)

    ## Get unique values
    min_vals = df.group_by("tx_idx").agg(pl.min("tx_dist"))
    df = df.join(min_vals, how="inner", on=["tx_idx", "tx_dist"])

    ## Rename to be consistent with everything else
    df = df.rename({"tx_idx": "transcript_id", "tx_x": "x", "tx_y": "y"})

    ## Replace gene indices with gene names
    df = df.with_columns(
        pl.col("tx_gene_idx")
        .replace_strict(idx2genes, default="None")
        .alias("feature_name")
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


def grid_group_lazyframe(
    ldf: pl.LazyFrame, box_width: float, min_size: int, offset: float
):

    grid_ldf = ldf.with_columns(
        [
            ((pl.col("x") / box_width) + offset).floor().cast(pl.Int32).alias("grid_x"),
            ((pl.col("y") / box_width) + offset).floor().cast(pl.Int32).alias("grid_y"),
        ]
    )

    grid_ldf = grid_ldf.with_columns(
        (
            pl.col("grid_x").cast(pl.String) + "_" + pl.col("grid_y").cast(pl.String)
        ).alias("grid_name")
    )
    counts = (
        grid_ldf.group_by(["grid_name"])
        .len()
        .collect()
        .to_pandas()
        .sort_values(by="len")
    )
    counts["label"] = group_labels_min_size(counts["len"].values, min_size)
    grid_name2label = dict(zip(counts["grid_name"], counts["label"]))
    grid_ldf = grid_ldf.with_columns(
        pl.col("grid_name").replace_strict(grid_name2label).alias("super_grid_label")
    )
    return grid_ldf


def double_grid_and_group(
    input_path: str,
    output_path: str,
    cluster_distance: float,
    box_width: float | None = 50.0,
    min_size: int | None = 25_000,
):

    ## Ensure box_width is large enough
    box_width = max(box_width, 2 * cluster_distance)

    ## Read data
    df = pl.scan_parquet(input_path)

    ## Build two offset grids
    grid_a_df = grid_group_lazyframe(
        df, box_width=box_width, min_size=min_size, offset=0.0
    )
    grid_a_df = grid_a_df.with_columns(pl.lit(0).alias("grid_version"))
    grid_b_df = grid_group_lazyframe(
        df, box_width=box_width, min_size=min_size, offset=0.5
    )
    grid_b_df = grid_b_df.with_columns(pl.lit(1).alias("grid_version"))

    ## Concatenate grids and save, partitioning by grid version and label
    grid_df = pl.concat([grid_a_df, grid_b_df])
    out_folder = pl.PartitionByKey(output_path, by=["grid_version", "super_grid_label"])
    grid_df.sink_parquet(out_folder, mkdir=True)
    return "Success!"


###############
###############


###############
### STEP 3 ###
##############


def grid_level_cluster(
    input_path: str, output_path: str, epsilon: float, n_workers: int
):

    ## Read in the parquet dataset
    source_dataset = ds.dataset(input_path, format="parquet")
    fragments = list(source_dataset.get_fragments())
    n_frags = len(fragments)

    ## Spin up the parallel processing context
    ctx = mp.get_context("spawn")
    in_q, out_q = ctx.Queue(), ctx.Queue()

    ## Spin up the workers
    workers = [
        DBSCANWorker(in_q, out_q, epsilon, output_path) for _ in range(n_workers)
    ]
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

    unique_cluster_id_0, idx_0 = np.unique(
        edges["full_cluster_id_0"], return_inverse=True
    )
    unique_cluster_id_1, idx_1 = np.unique(
        edges["full_cluster_id_1"], return_inverse=True
    )
    n_0 = unique_cluster_id_0.shape[0]
    idx_1 = idx_1 + n_0

    unique_cluster_id = np.concatenate([unique_cluster_id_0, unique_cluster_id_1])
    n = unique_cluster_id.shape[0]
    n_edges = edges.shape[0]
    csgraph = coo_array((np.ones(n_edges), (idx_0, idx_1)), shape=(n, n))
    _, labels = connected_components(csgraph, directed=False)
    cluster_id2final_label = dict(zip(unique_cluster_id, labels))

    df_mapped = df.filter(pl.col("grid_version") == 0).with_columns(
        pl.col("full_cluster_id")
        .replace_strict(cluster_id2final_label)
        .alias("ef_cluster_id")
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
    mahalanobis_distance = np.sqrt(
        np.einsum("...i,...ij,...j->...", diff, Q[centroid_idx], diff)
    )

    ## Map everything back onto the lazyframe
    cluster_id2centroid_cell_id = dict(zip(cluster_id, centroid_cell_id))
    cluster_id2centroid_distance = dict(zip(cluster_id, centroid_distance))
    cluster_id2mahalanobis_distance = dict(zip(cluster_id, mahalanobis_distance))

    df = df.with_columns(
        pl.col("ef_cluster_id")
        .replace_strict(cluster_id2centroid_cell_id)
        .alias("cell_id")
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


def post_process(
    directory: str,
    tma: TMA,
    cluster_distance: float,
    input_fname: str | None = None,
    output_fname: str | None = None,
    n_workers: int | None = None,
):

    if input_fname is None:
        input_fname = "tx"

    if output_fname is None:
        output_fname = "ef.parquet"

    ## Get number of parallel workers
    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0))

    ## Get mappings for genes
    idx2genes = {i: x for i, x in enumerate(tma.genes)}

    ## Step 1: Make tx unique
    print("Step 1: getting unique tx...", file=sys.stderr)
    input_path = os.path.join(directory, input_fname)
    temp_fname1 = "temp1-" + uuid.uuid4().hex + ".parquet"
    temp_path1 = os.path.join(directory, temp_fname1)
    filter_unique_and_remap_tx(
        input_path=input_path, output_path=temp_path1, idx2genes=idx2genes
    )

    ## Step 2: Build double grid
    print("Step 2: building double grid...", file=sys.stderr)
    temp_fname2 = "temp2-" + uuid.uuid4().hex
    temp_path2 = os.path.join(directory, temp_fname2)
    double_grid_and_group(
        input_path=temp_path1, output_path=temp_path2, cluster_distance=cluster_distance
    )

    ## Remove first intermediate file
    remove(temp_path1)

    ## Step 3: Cluster within each grid section
    print("Step 3: clustering within grid sections...", file=sys.stderr)
    temp_fname3 = "temp3-" + uuid.uuid4().hex
    temp_path3 = os.path.join(directory, temp_fname3)
    grid_level_cluster(
        input_path=temp_path2,
        output_path=temp_path3,
        epsilon=cluster_distance,
        n_workers=n_workers,
    )

    ## Remove second intermediate file
    remove(temp_path2)

    ## Step 4: Link clusters across grid versions
    print("Step 4: linking clusters across grids...", file=sys.stderr)
    temp_fname4 = "temp4-" + uuid.uuid4().hex + ".parquet"
    temp_path4 = os.path.join(directory, temp_fname4)
    grid_connected_components(input_path=temp_path3, output_path=temp_path4)

    ## Remove third intermediate file
    remove(temp_path3)

    ## Step 5: Map clusters to cells
    print("Step 5: mapping clusters to cells...", file=sys.stderr)
    output_path = os.path.join(directory, output_fname)
    map_clusters_to_cells(
        input_path=temp_path4,
        output_path=output_path,
        nuc_centroid=tma.nuc_xy,
        nuc_ellipse_mat=tma.nuc_ellipse_mat,
        nuc_cell_id=tma.nuc_cell_id,
    )

    ## Remove fourth intermediate file
    remove(temp_path4)

    print("DONE!", file=sys.stderr)

    return "Success!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", type=str, help="Model directory.")
    parser.add_argument("--tma_directory", type=str, help="TMA directory.")
    parser.add_argument(
        "--cluster_distance",
        type=float,
        default=1.5,
        help="Linkage distance for clustering",
    )

    args = parser.parse_args()
    tma = TMA.load(os.path.join(args.tma_directory, "tma.npz"))

    post_process(
        directory=args.model_directory, tma=tma, cluster_distance=args.cluster_distance
    )
