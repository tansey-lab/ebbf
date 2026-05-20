import argparse
import logging
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import polars as pl
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


def _path_nbytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += os.path.getsize(os.path.join(root, fname))
    return total


def _path_nrows(path: str) -> int | None:
    try:
        return ds.dataset(path, format="parquet").count_rows()
    except Exception:
        return None


def _log_step(step_name: str, started_at: float, output_path: str, timings: dict | None = None):
    elapsed = time.perf_counter() - started_at
    if timings is not None:
        timings[step_name] = elapsed
    nbytes = _path_nbytes(output_path)
    nrows = _path_nrows(output_path)
    logger.info(
        "%s complete in %.2fs | rows=%s | size=%.2f GB | path=%s",
        step_name,
        elapsed,
        (str(nrows) if nrows is not None else "NA"),
        nbytes / (1024**3),
        output_path,
    )


def remove(path):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError(f"file {path} is not a file or dir.")


def _dbscan_worker(in_q, out_q, epsilon: float, output_path: str):
    dbscan = DBSCAN(eps=epsilon, min_samples=1)

    while True:
        fragment: ds.ParquetFileFragment = in_q.get()

        if fragment is None:
            break

        # Each file may contain multiple super_grid_labels — run DBSCAN per label
        df = pl.from_arrow(fragment.to_table())
        results = []
        for group_df in df.partition_by("super_grid_label", maintain_order=False):
            xy = group_df.select(["x", "y"]).to_numpy().astype(np.float64)
            cluster_ids = dbscan.fit_predict(xy)
            results.append(
                group_df.with_columns(pl.Series("cluster_id", cluster_ids, dtype=pl.Int64))
            )
        output_table = pl.concat(results).to_arrow()

        path_names = fragment.path.split("/")
        out_fname = os.path.join(output_path, *path_names[2:])
        out_folder = os.path.join(output_path, *path_names[2:-1])
        os.makedirs(out_folder, exist_ok=True)
        pq.write_table(output_table, out_fname)

        out_q.put(out_fname)


###############
### STEP 1 ###
##############
def filter_unique_and_remap_tx(input_path: str | list[str], output_path: str, idx2genes: dict):
    logger.info("filter_unique_and_remap_tx: deduplicating transcripts and remapping gene indices")

    df = (
        pl.scan_parquet(input_path)
        .filter(pl.col("tx_dist") == pl.col("tx_dist").min().over("tx_idx"))
        .unique(subset="tx_idx", keep="first")
    )

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

    ## Add noise columns (always present in output, with graceful fallback)
    schema = df.collect_schema()
    if "tx_noise_prob_1" in schema:
        df = df.with_columns((pl.col("tx_noise_prob_1") > 0.5).alias("is_noise"))
    else:
        df = df.with_columns(pl.lit(False).alias("is_noise"))
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("tx_noise_prob_1"))

    if "tx_noise_prob_0" not in schema:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("tx_noise_prob_0"))

    ## Ship it
    df.sink_parquet(output_path)
    logger.info("filter_unique_and_remap_tx: done → %s", output_path)
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
    batch_size: int = 50_000_000,
    n_file_groups: int = 10_000,
):
    """Stream ``input_path`` in large batches and write hive-partitioned output.

    Instead of one file per ``super_grid_label`` partition (~14k files), partitions
    are packed into ``n_file_groups`` files per grid version (200 files total).
    Step 3 (``grid_level_cluster``) splits each file by ``super_grid_label`` in
    memory before running DBSCAN, so correctness is unchanged.

    ``batch_size`` controls peak memory per batch and, critically, the total number
    of write calls: larger batches → fewer write calls → faster on HPC shared
    filesystems where per-write metadata overhead dominates.  The default of 50 M
    rows uses ~4–6 GB RAM per batch and reduces write calls ~10× versus 5 M rows.
    Each batch is converted to Arrow once, then partitioned via zero-copy slicing
    rather than 100 separate Polars ``to_arrow()`` calls.
    """
    ## Ensure box_width is large enough
    box_width = max(box_width, 2 * cluster_distance)
    os.makedirs(output_path, exist_ok=True)

    for grid_version, offset in tqdm([(0, 0.0), (1, 0.5)], desc="double_grid_and_group"):
        ## Phase 1: compute grid_name -> super_grid_label mapping (small aggregation)
        logger.info(
            "double_grid_and_group: grid_version=%d phase 1 – computing partition sizes",
            grid_version,
        )
        ldf = pl.scan_parquet(input_path)

        counts = (
            _add_grid_columns(ldf, box_width, offset)
            .group_by("grid_name")
            .len()
            .collect(engine="streaming")
            .sort(by="len")
        )
        logger.info(
            "double_grid_and_group: grid_version=%d phase 1 – computing partition super groups",
            grid_version,
        )
        labels = group_labels_min_size(counts["len"].to_numpy(), min_size)
        mapping_df = pl.DataFrame(
            {
                "grid_name": counts["grid_name"].to_list(),
                "super_grid_label": labels,
            }
        )
        n_partitions = int(mapping_df["super_grid_label"].max()) + 1
        n_groups = min(n_file_groups, n_partitions)
        mapping_df = mapping_df.with_columns(
            (pl.col("super_grid_label") % n_groups).cast(pl.Int32).alias("file_group")
        )
        logger.info(
            "double_grid_and_group: grid_version=%d – %d partitions packed into %d files",
            grid_version,
            n_partitions,
            n_groups,
        )

        ## Phase 2: stream in large batches; one Arrow conversion + zero-copy slices
        ## per batch so write calls = n_batches × n_groups instead of the
        ## n_batches × n_groups × (to_arrow overhead) of the naive approach.
        logger.info(
            "double_grid_and_group: grid_version=%d phase 2 – streaming with batch_size=%d",
            grid_version,
            batch_size,
        )
        writers: dict[int, pq.ParquetWriter] = {}
        parquet_file = pq.ParquetFile(input_path)
        try:
            for record_batch in parquet_file.iter_batches(batch_size=batch_size):
                batch = (
                    pl.from_arrow(record_batch)
                    .pipe(lambda df: _add_grid_columns(df, box_width, offset))
                    .join(mapping_df, on="grid_name", how="left")
                    .with_columns(pl.lit(grid_version).cast(pl.Int32).alias("grid_version"))
                    .sort("file_group")  # contiguous runs → O(1) Arrow slices below
                )
                table = batch.to_arrow()  # one conversion for the whole batch
                fg_col = batch["file_group"].to_numpy()

                # Identify contiguous runs of each file_group value
                boundaries = np.where(np.diff(fg_col))[0] + 1
                starts = np.concatenate([[0], boundaries])
                ends = np.concatenate([boundaries, [len(batch)]])

                for start, end in zip(starts, ends):
                    fg = int(fg_col[start])
                    final_part_dir = os.path.join(
                        output_path, f"grid_version={grid_version}", f"file_group={fg}"
                    )
                    os.makedirs(final_part_dir, exist_ok=True)
                    final_path = os.path.join(final_part_dir, "data.parquet")
                    slice_table = table.slice(start, end - start)  # zero-copy
                    writer = writers.get(fg)
                    if writer is None:
                        writer = pq.ParquetWriter(final_path, slice_table.schema)
                        writers[fg] = writer
                    writer.write_table(slice_table)
        finally:
            for writer in writers.values():
                writer.close()

        logger.info("double_grid_and_group: grid_version=%d done", grid_version)

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

    ## Spin up the workers, use ctx.Process so spawn is respected
    ## avoids fork-after-polars/Rayon deadlock
    workers = [
        ctx.Process(target=_dbscan_worker, args=(in_q, out_q, epsilon, output_path))
        for _ in range(n_workers)
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

# Cluster key packing: encode (super_grid_label, grid_version, cluster_id) into one Int64.
#
# Bit layout (63 bits, fits in signed Int64):
#   bits [62:32]  super_grid_label  31 bits  max 2,147,483,647
#   bit  [31]     grid_version       1 bit   0 or 1
#   bits [30: 0]  cluster_id        31 bits  max 2,147,483,647
_CK_SGL_SHIFT = 32  # super_grid_label occupies bits 62..32
_CK_GV_SHIFT = 31  # grid_version occupies bit 31
_CK_MASK_31 = (1 << 31) - 1  # low-31-bit mask for cluster_id


def ck_encode() -> pl.Expr:
    """Encode columns (super_grid_label, grid_version, cluster_id) → packed Int64 '_ck'."""
    return (
        (pl.col("super_grid_label").cast(pl.Int64) * (1 << _CK_SGL_SHIFT))
        | (pl.col("grid_version").cast(pl.Int64) * (1 << _CK_GV_SHIFT))
        | pl.col("cluster_id").cast(pl.Int64)
    ).alias("_ck")


def ck_decode_super_grid_label(expr: pl.Expr = pl.col("_ck")) -> pl.Expr:
    """Recover super_grid_label from a packed cluster-key expression."""
    return (expr // (1 << _CK_SGL_SHIFT)).alias("super_grid_label")


def ck_decode_grid_version(expr: pl.Expr = pl.col("_ck")) -> pl.Expr:
    """Recover grid_version (0 or 1) from a packed cluster-key expression."""
    return ((expr // (1 << _CK_GV_SHIFT)) & 1).alias("grid_version")


def ck_decode_cluster_id(expr: pl.Expr = pl.col("_ck")) -> pl.Expr:
    """Recover cluster_id from a packed cluster-key expression."""
    return (expr & _CK_MASK_31).alias("cluster_id")


def grid_connected_components(input_path: str, output_path: str, n_parts: int = 50):

    # Partitioned join: each pass loads only transcript_ids where transcript_id % n_parts == k,
    # capping peak memory to roughly 2 × (total_rows / n_parts) × 16 bytes instead of
    # loading all ~458M rows per side at once.
    logger.info("grid_connected_components: collecting edges (%d partitions)", n_parts)
    edge_chunks: list[pl.DataFrame] = []
    for k in tqdm(range(n_parts), desc="grid_connected_components"):
        part = (
            pl.scan_parquet(input_path)
            .filter((pl.col("transcript_id") % n_parts) == k)
            .with_columns(ck_encode())
            .select(["transcript_id", "grid_version", "_ck"])
            .collect(engine="streaming")
        )
        v0 = part.filter(pl.col("grid_version") == 0).select(
            ["transcript_id", pl.col("_ck").alias("ck0")]
        )
        v1 = part.filter(pl.col("grid_version") == 1).select(
            ["transcript_id", pl.col("_ck").alias("ck1")]
        )
        edge_chunks.append(v0.join(v1, on="transcript_id").select(["ck0", "ck1"]).unique())

    edges = pl.concat(edge_chunks).unique().to_pandas()
    logger.info(
        "grid_connected_components: %d unique edges across %d clusters",
        len(edges),
        len(set(edges["ck0"]) | set(edges["ck1"])),
    )

    logger.info("grid_connected_components: running connected_components")
    unique_ck0, idx_0 = np.unique(edges["ck0"], return_inverse=True)
    unique_ck1, idx_1 = np.unique(edges["ck1"], return_inverse=True)
    n_0 = unique_ck0.shape[0]
    idx_1 = idx_1 + n_0

    unique_cluster_id = np.concatenate([unique_ck0, unique_ck1])
    n = unique_cluster_id.shape[0]
    n_edges = edges.shape[0]
    csgraph = coo_array((np.ones(n_edges), (idx_0, idx_1)), shape=(n, n))
    _, labels = connected_components(csgraph, directed=False)
    n_components = int(labels.max()) + 1
    logger.info(
        "grid_connected_components: %d connected components from %d nodes, %d edges",
        n_components,
        n,
        n_edges,
    )
    cluster_id2final_label = dict(zip(unique_cluster_id.tolist(), labels.tolist()))

    logger.info("grid_connected_components: writing output → %s", output_path)
    (
        pl.scan_parquet(input_path)
        .with_columns(ck_encode())
        .filter(pl.col("grid_version") == 0)
        .with_columns(pl.col("_ck").replace_strict(cluster_id2final_label).alias("ef_cluster_id"))
        .sink_parquet(output_path)
    )
    logger.info("grid_connected_components: done")
    return "Success!"


###############
###############


###############
### STEP 4 ###
##############


def compute_cluster_to_cell_mapping(
    cluster_id: np.ndarray,
    cluster_xy: np.ndarray,
    nuc_centroid: np.ndarray,
    nuc_ellipse_mat: np.ndarray,
    nuc_cell_id: np.ndarray,
) -> pl.DataFrame:
    """Assign each cluster to its nearest nucleus and compute distances.

    For each cluster center, finds the nearest nucleus centroid (Euclidean)
    and computes the Mahalanobis distance to that nucleus' ellipse. The
    ellipse is parameterized by ``nuc_ellipse_mat`` ``M``: ``M`` maps the
    unit circle to the ellipse, so the Mahalanobis metric uses
    ``Q = inv(M @ M.T)``.

    Returns a DataFrame with columns
    ``ef_cluster_id, cell_id, centroid_distance, mahalanobis_distance``.
    """
    tree = KDTree(nuc_centroid)
    centroid_distance, centroid_idx = tree.query(cluster_xy)
    centroid_cell_id = nuc_cell_id[centroid_idx]

    diff = cluster_xy - nuc_centroid[centroid_idx]
    Q = np.linalg.inv(np.matmul(nuc_ellipse_mat, np.swapaxes(nuc_ellipse_mat, -1, -2)))
    mahalanobis_distance = np.sqrt(np.einsum("...i,...ij,...j->...", diff, Q[centroid_idx], diff))

    return pl.DataFrame(
        {
            "ef_cluster_id": cluster_id,
            "cell_id": centroid_cell_id,
            "centroid_distance": centroid_distance,
            "mahalanobis_distance": mahalanobis_distance,
        }
    )


def map_clusters_to_cells(
    input_path: str,
    nuc_centroid: np.ndarray,
    nuc_ellipse_mat: np.ndarray,
    nuc_cell_id: np.ndarray,
    worker_id: int | None = None,
    worker_n: int | None = None,
) -> pl.LazyFrame:
    """Map clustered transcripts to nucleus cell IDs.

    Returns a LazyFrame keyed by transcript with cell assignments and
    distance columns. The caller chooses how to materialize the result
    (e.g. ``.sink_parquet(path)`` or ``.collect()``).
    """
    df = pl.scan_parquet(input_path)

    cluster_df = (
        df.group_by(["ef_cluster_id"])
        .agg(pl.col("x").mean(), pl.col("y").mean())
        .collect(engine="streaming")
    )
    cluster_id = cluster_df["ef_cluster_id"].to_numpy()
    cluster_xy = cluster_df.select(["x", "y"]).to_numpy()

    mapping_df = compute_cluster_to_cell_mapping(
        cluster_id=cluster_id,
        cluster_xy=cluster_xy,
        nuc_centroid=nuc_centroid,
        nuc_ellipse_mat=nuc_ellipse_mat,
        nuc_cell_id=nuc_cell_id,
    )

    if (worker_id is not None) and (worker_n is not None):
        mapping_df = mapping_df.filter((pl.col("ef_cluster_id") % worker_n) == worker_id)

    return df.join(mapping_df.lazy(), on="ef_cluster_id", how="inner").select(
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


def post_process_prepare(
    input_files: list[str],
    dataset: STDataset,
    cluster_distance: float,
    output_fname: str,
    noise_meta_fname: str | None = None,
    work_dir: str = ".",
    n_workers: int | None = None,
):
    """Run steps 1–4: dedup, grid, DBSCAN, connected components.

    If ``noise_meta_fname`` is provided, noise transcripts are separated before
    DBSCAN and their metadata (transcript_id, is_noise, tx_noise_prob_*) is
    written there for ``post_process_merge`` to re-attach.
    """
    if n_workers is None:
        n_workers = get_num_cpus()
    os.makedirs(work_dir, exist_ok=True)
    idx2genes = {i: x for i, x in enumerate(dataset.genes)}
    timings: dict[str, float] = {}

    temp_path1 = os.path.join(work_dir, "temp1-" + uuid.uuid4().hex + ".parquet")
    temp_path2 = os.path.join(work_dir, "temp2-" + uuid.uuid4().hex)
    temp_path3 = os.path.join(work_dir, "temp3-" + uuid.uuid4().hex)

    logger.info("Step 1: deduplicating and remapping transcripts")
    step_start = time.perf_counter()
    filter_unique_and_remap_tx(input_path=input_files, output_path=temp_path1, idx2genes=idx2genes)
    _log_step("Step 1", step_start, temp_path1, timings)

    ## Save noise metadata and filter to non-noise before DBSCAN.
    ## noise_meta contains ALL rows with noise probability columns (for joining back onto
    ## non-noise rows later) plus full positional data for noise rows (for appending later).
    if noise_meta_fname is not None:
        _step1_df = pl.scan_parquet(temp_path1)

        ## All rows: just the noise probability columns for joining back later
        ## Noise rows: also keep positions / feature_name so we can append them to ef.parquet
        _step1_df.with_columns(
            ## ensure x_initial / y_initial are present for noise rows (they came from infer)
            pl.col("x").alias("x"),
            pl.col("y").alias("y"),
        ).select(
            [
                "transcript_id",
                "feature_name",
                "x",
                "y",
                "x_initial",
                "y_initial",
                "is_noise",
                "tx_noise_prob_0",
                "tx_noise_prob_1",
            ]
        ).sink_parquet(noise_meta_fname)
        logger.info("Noise metadata written → %s", noise_meta_fname)

        temp_path1b = os.path.join(work_dir, "temp1b-" + uuid.uuid4().hex + ".parquet")
        _step1_df.filter(~pl.col("is_noise")).drop(
            ["is_noise", "tx_noise_prob_0", "tx_noise_prob_1"]
        ).sink_parquet(temp_path1b)
        remove(temp_path1)
        temp_path1 = temp_path1b

    logger.info("Step 2: building double grid (n_workers=%d)", n_workers)
    step_start = time.perf_counter()
    double_grid_and_group(
        input_path=temp_path1, output_path=temp_path2, cluster_distance=cluster_distance
    )
    _log_step("Step 2", step_start, temp_path2, timings)
    remove(temp_path1)

    logger.info("Step 3: clustering within grid sections (n_workers=%d)", n_workers)
    step_start = time.perf_counter()
    grid_level_cluster(
        input_path=temp_path2, output_path=temp_path3, epsilon=cluster_distance, n_workers=n_workers
    )
    _log_step("Step 3", step_start, temp_path3, timings)
    remove(temp_path2)

    logger.info("Step 4: linking clusters across grid versions")
    step_start = time.perf_counter()
    grid_connected_components(input_path=temp_path3, output_path=output_fname)
    _log_step("Step 4", step_start, output_fname, timings)
    remove(temp_path3)

    logger.info("post_process_prepare complete. Step timing summary (seconds): %s", timings)
    return "Success!"


def post_process_merge(
    mapped_files: list[str],
    dataset: STDataset,
    output_fname: str = "ef.parquet",
    anndata_fname: str | None = "ef.h5ad",
    original_tx_fname: str | None = None,
    noise_meta_fname: str | None = None,
    work_dir: str = ".",
    denovo_min_tx_pctile: float = 25.0,
    denovo_filter_report_fname: str | None = None,
):
    os.makedirs(work_dir, exist_ok=True)
    temp_path = os.path.join(work_dir, "temp-merge-" + uuid.uuid4().hex + ".parquet")
    pl.scan_parquet(mapped_files).sink_parquet(temp_path)
    if original_tx_fname is not None:
        calculate_tx_distances(
            input_path=temp_path,
            output_path=output_fname,
            original_tx_fname=original_tx_fname,
        )
        remove(temp_path)
    else:
        os.rename(temp_path, output_fname)

    if anndata_fname is not None:
        create_anndata(
            ef_tx_fname=output_fname,
            output_fname=anndata_fname,
            dataset=dataset,
            denovo_min_tx_pctile=denovo_min_tx_pctile,
            denovo_filter_report_fname=denovo_filter_report_fname,
        )

    ## Re-attach noise probability columns and append noise rows to ef.parquet
    if noise_meta_fname is not None:
        logger.info("Re-attaching noise metadata from %s", noise_meta_fname)
        noise_meta = pl.scan_parquet(noise_meta_fname).with_columns(
            pl.col("transcript_id").cast(pl.UInt64)
        )

        ## Step A: join is_noise + prob cols back onto the non-noise ef rows
        probs_df = noise_meta.select(
            ["transcript_id", "is_noise", "tx_noise_prob_0", "tx_noise_prob_1"]
        )
        ef_ldf = pl.scan_parquet(output_fname).join(probs_df, on="transcript_id", how="left")

        ## Step B: build noise rows (positions are already in noise_meta from step1)
        noise_rows = noise_meta.filter(pl.col("is_noise"))
        if original_tx_fname is not None:
            tx_df = (
                pl.scan_parquet(original_tx_fname)
                .select(["transcript_id", "overlaps_nucleus"])
                .with_columns(pl.col("transcript_id").cast(pl.UInt64))
            )
            noise_rows = noise_rows.join(tx_df, on="transcript_id", how="left")
            noise_rows = noise_rows.with_columns(
                distance=(
                    (pl.col("x") - pl.col("x_initial")).pow(2)
                    + (pl.col("y") - pl.col("y_initial")).pow(2)
                ).sqrt()
            )

        combined = pl.concat([ef_ldf, noise_rows], how="diagonal")
        tmp_out = output_fname + ".tmp"
        combined.sink_parquet(tmp_out)
        os.replace(tmp_out, output_fname)
        logger.info("Noise rows merged into %s", output_fname)

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
    denovo_min_tx_pctile: float = 25.0,
    denovo_filter_report_fname: str | None = None,
):
    """Create AnnData file from transcript assignment table.

    Cells are classified into two categories based on Mahalanobis distance:
    1. Original cells: mahalanobis_distance <= 1
    2. De novo cells: mahalanobis_distance > 1

    De novo cells are further filtered against the input segmentation:
    clusters with transcript count below the ``denovo_min_tx_pctile``
    percentile of input cell transcript counts are dropped. Original cells are
    never filtered out by this threshold. Set the percentile to 0 to disable.

    Args:
        ef_tx_fname: Path to the transcript assignment parquet file
        output_fname: Path to output h5ad file
        dataset: STDataset with nucleus spatial data
        denovo_min_tx_pctile: Percentile of input cell transcript counts used
            as the minimum-size threshold for de novo cells (default: 25.0).
        denovo_filter_report_fname: Parquet path for the per-de-novo-cluster
            filter report (transcript count, threshold, and pass/fail flags).
            If None, derived as
            ``{output_fname stem}.denovo_filter_report.parquet``.
    """
    stem = str(Path(output_fname).with_suffix(""))
    if denovo_filter_report_fname is None:
        denovo_filter_report_fname = stem + ".denovo_filter_report.parquet"
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

    # Classify clusters by Mahalanobis distance only.
    cluster_info = cluster_info.with_columns(
        [
            pl.when(pl.col("mahalanobis_distance") > 1)
            .then(pl.lit("denovo"))
            .otherwise(pl.lit("original"))
            .alias("cell_type")
        ]
    )

    # Filter de novo clusters against an input segmentation transcript-count
    # percentile. Zeros are dropped from the reference so empty segmentations
    # don't collapse the percentile to 0. A per-cluster report is always
    # written (empty if no candidate clusters exist) so downstream consumers
    # can rely on its presence.
    candidate_mask = pl.col("cell_type") == "denovo"
    n_candidate_before = int(cluster_info.filter(candidate_mask).height)
    filter_active = denovo_min_tx_pctile > 0

    input_tx = np.asarray(dataset.nuc_cell_counts)
    input_tx_nz = input_tx[input_tx > 0]
    tx_threshold = (
        float(np.percentile(input_tx_nz, denovo_min_tx_pctile))
        if denovo_min_tx_pctile > 0 and input_tx_nz.size > 0
        else 0.0
    )

    report_schema = {
        "ef_cluster_id": cluster_info.schema["ef_cluster_id"],
        "cell_id": cluster_info.schema["cell_id"],
        "cell_type": pl.Utf8,
        "x": cluster_info.schema["x"],
        "y": cluster_info.schema["y"],
        "centroid_distance": cluster_info.schema["centroid_distance"],
        "mahalanobis_distance": cluster_info.schema["mahalanobis_distance"],
        "n_tx": pl.Int64,
        "tx_threshold": pl.Float64,
        "tx_pctile": pl.Float64,
        "passes_tx_filter": pl.Boolean,
        "kept": pl.Boolean,
    }

    if n_candidate_before > 0:
        candidate_ids = cluster_info.filter(candidate_mask)["ef_cluster_id"].to_list()
        candidate_tx = (
            ldf.filter(pl.col("ef_cluster_id").is_in(candidate_ids))
            .select(["ef_cluster_id", "x", "y"])
            .collect()
        )

        n_tx_per_cluster: dict[int, int] = {}
        for key, group in candidate_tx.group_by("ef_cluster_id"):
            cid = key[0] if isinstance(key, tuple) else key
            n_tx_per_cluster[cid] = group.height

        stats_df = pl.DataFrame(
            {
                "ef_cluster_id": list(n_tx_per_cluster.keys()),
                "_candidate_n_tx": list(n_tx_per_cluster.values()),
            }
        )
        cluster_info = cluster_info.join(stats_df, on="ef_cluster_id", how="left")

        pass_tx = pl.col("_candidate_n_tx").fill_null(0) >= tx_threshold

        report = (
            cluster_info.filter(candidate_mask)
            .with_columns(
                [
                    pl.col("_candidate_n_tx").fill_null(0).cast(pl.Int64).alias("n_tx"),
                    pl.lit(tx_threshold).alias("tx_threshold"),
                    pl.lit(denovo_min_tx_pctile).alias("tx_pctile"),
                    pass_tx.alias("passes_tx_filter"),
                    pass_tx.alias("kept"),
                ]
            )
            .select(list(report_schema.keys()))
        )

        cluster_info = cluster_info.drop("_candidate_n_tx")
    else:
        report = pl.DataFrame(schema=report_schema)

    report.write_parquet(denovo_filter_report_fname)

    # Build per-cluster maps from the candidate report; non-candidate
    # (original) clusters are not filtered and default to passing.
    ef_to_passes_tx: dict[int, bool] = {}
    ef_to_kept: dict[int, bool] = {}
    if report.height > 0:
        for ef, p_tx, kept in zip(
            report["ef_cluster_id"].to_list(),
            report["passes_tx_filter"].to_list(),
            report["kept"].to_list(),
        ):
            ef_to_passes_tx[ef] = p_tx
            ef_to_kept[ef] = kept

    n_candidate_after = (
        sum(1 for v in ef_to_kept.values() if v) if filter_active else n_candidate_before
    )
    logger.info(
        "De novo filter: %d → %d (tx_count >= %.1f [p%.0f]); parquet → %s (%d rows)",
        n_candidate_before,
        n_candidate_after,
        tx_threshold,
        denovo_min_tx_pctile,
        denovo_filter_report_fname,
        report.height,
    )

    # Assign unique cell identifiers
    # Original: use cell_id directly (multiple clusters merge)
    # De novo: "denovo_{ef_cluster_id}"
    cluster_info = cluster_info.with_columns(
        [
            pl.when(pl.col("cell_type") == "original")
            .then(pl.col("cell_id").cast(pl.Utf8))
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

    # Per-cluster centroids from transcripts: post-diffusion (x, y) and
    # pre-diffusion (x_initial, y_initial) means.
    cluster_centroids = (
        ldf.with_columns(
            (
                (pl.col("x") - pl.col("x_initial")).pow(2)
                + (pl.col("y") - pl.col("y_initial")).pow(2)
            )
            .sqrt()
            .alias("_displacement")
        )
        .group_by("ef_cluster_id")
        .agg(
            [
                pl.col("x").mean().alias("x_post"),
                pl.col("y").mean().alias("y_post"),
                pl.col("x_initial").mean().alias("x_pre"),
                pl.col("y_initial").mean().alias("y_pre"),
                pl.col("_displacement").mean().alias("mean_displacement"),
            ]
        )
        .collect()
    )
    ef_to_post_xy = {
        ef: (xp, yp)
        for ef, xp, yp in zip(
            cluster_centroids["ef_cluster_id"].to_list(),
            cluster_centroids["x_post"].to_list(),
            cluster_centroids["y_post"].to_list(),
        )
    }
    ef_to_pre_xy = {
        ef: (xp, yp)
        for ef, xp, yp in zip(
            cluster_centroids["ef_cluster_id"].to_list(),
            cluster_centroids["x_pre"].to_list(),
            cluster_centroids["y_pre"].to_list(),
        )
    }
    ef_to_mean_displacement = dict(
        zip(
            cluster_centroids["ef_cluster_id"].to_list(),
            cluster_centroids["mean_displacement"].to_list(),
        )
    )

    # Build spatial and metadata mappings per unified_cell_id
    cell_id_to_post_xy = {}
    cell_id_to_pre_xy = {}
    cell_id_to_mean_displacement = {}
    cell_id_to_type_label = {}
    cell_id_to_parent = {}
    # Debug fields: the nearest nucleus and the distances used to classify
    # this cluster. Empty original cells (no transcripts) leave these as None.
    cell_id_to_nearest_centroid_cell_id = {}
    cell_id_to_nearest_centroid_x = {}
    cell_id_to_nearest_centroid_y = {}
    cell_id_to_centroid_distance = {}
    cell_id_to_mahalanobis_distance = {}
    cell_id_to_ef_cluster_id = {}

    for row in cluster_info.iter_rows(named=True):
        uid = row["unified_cell_id"]
        ctype = row["cell_type"]
        ef = row["ef_cluster_id"]

        cell_id_to_post_xy[uid] = ef_to_post_xy[ef]
        cell_id_to_pre_xy[uid] = ef_to_pre_xy[ef]
        cell_id_to_mean_displacement[uid] = ef_to_mean_displacement[ef]

        if ctype == "original":
            cell_id_to_parent[uid] = str(row["cell_id"])
        else:  # denovo
            cell_id_to_parent[uid] = None

        cell_id_to_type_label[uid] = ctype

        nearest_id = row["cell_id"]
        cell_id_to_nearest_centroid_cell_id[uid] = str(nearest_id)
        cell_id_to_nearest_centroid_x[uid] = cell_id2x_centroid.get(nearest_id)
        cell_id_to_nearest_centroid_y[uid] = cell_id2y_centroid.get(nearest_id)
        cell_id_to_centroid_distance[uid] = row["centroid_distance"]
        cell_id_to_mahalanobis_distance[uid] = row["mahalanobis_distance"]
        cell_id_to_ef_cluster_id[uid] = row["ef_cluster_id"]

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

    # Only include cells that received at least one transcript; nuclei whose
    # transcripts all diffused away (no surviving original cluster) are
    # dropped rather than emitted as empty rows.
    all_cell_ids = sorted(cell_id_to_post_xy.keys())
    cell_id2idx = {cid: i for i, cid in enumerate(all_cell_ids)}

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
    adata.obs["x_centroid"] = adata.obs.index.map(lambda uid: cell_id_to_post_xy[uid][0])
    adata.obs["y_centroid"] = adata.obs.index.map(lambda uid: cell_id_to_post_xy[uid][1])
    adata.obs["mean_displacement"] = adata.obs.index.map(cell_id_to_mean_displacement).astype(float)
    adata.obs["n_tx"] = np.array(adata.X.sum(axis=1)).flatten().astype(int)

    # Debug fields: classifier inputs for this cluster (None for empty originals)
    adata.obs["nearest_centroid_cell_id"] = adata.obs.index.map(cell_id_to_nearest_centroid_cell_id)
    adata.obs["nearest_centroid_x"] = adata.obs.index.map(cell_id_to_nearest_centroid_x)
    adata.obs["nearest_centroid_y"] = adata.obs.index.map(cell_id_to_nearest_centroid_y)
    adata.obs["centroid_distance"] = adata.obs.index.map(cell_id_to_centroid_distance)
    adata.obs["mahalanobis_distance"] = adata.obs.index.map(cell_id_to_mahalanobis_distance)
    adata.obs["ef_cluster_id"] = adata.obs.index.map(cell_id_to_ef_cluster_id)

    # Store spatial coordinates in obsm for compatibility with spatial analysis
    # tools. ``spatial`` is the post-diffusion cluster centroid;
    # ``prediffusion_centroid`` is the mean of the cluster's transcripts at
    # their pre-diffusion (input) positions.
    adata.obsm["spatial"] = np.column_stack(
        [
            np.array([cell_id_to_post_xy[uid][0] for uid in adata.obs.index], dtype=float),
            np.array([cell_id_to_post_xy[uid][1] for uid in adata.obs.index], dtype=float),
        ]
    )
    adata.obsm["prediffusion_centroid"] = np.column_stack(
        [
            np.array([cell_id_to_pre_xy[uid][0] for uid in adata.obs.index], dtype=float),
            np.array([cell_id_to_pre_xy[uid][1] for uid in adata.obs.index], dtype=float),
        ]
    )

    # Attach filter outcomes to obs. ``original`` clusters are never filtered
    # so they pass by default; de novo candidates inherit pass/fail from the
    # denovo filter report.
    def _passes_tx_for(uid: str) -> bool:
        ef = cell_id_to_ef_cluster_id[uid]
        return ef_to_passes_tx.get(ef, True)

    def _kept_for(uid: str) -> bool:
        ef = cell_id_to_ef_cluster_id[uid]
        # Non-candidate (original) clusters aren't in the report; they're kept.
        return ef_to_kept.get(ef, True)

    adata.obs["passes_tx_filter"] = adata.obs.index.map(_passes_tx_for).astype(bool)
    adata.obs["kept"] = adata.obs.index.map(_kept_for).astype(bool)

    # Save the pre-filter AnnData alongside the filtered one so downstream
    # consumers can inspect dropped clusters.
    unfiltered_fname = stem + ".unfiltered.h5ad"
    adata.write(unfiltered_fname, compression="lzf")
    logger.info("Unfiltered AnnData → %s (%d cells)", unfiltered_fname, adata.n_obs)

    if filter_active:
        adata = adata[adata.obs["kept"].values].copy()
    adata.write(output_fname, compression="lzf")
    logger.info("Filtered AnnData → %s (%d cells)", output_fname, adata.n_obs)
    return adata


###############
###############


def post_process(
    input_files: list[str],
    dataset: STDataset,
    cluster_distance: float,
    output_fname: str = "ef.parquet",
    anndata_fname: str | None = "ef.h5ad",
    original_tx_fname: str | None = None,
    work_dir: str = ".",
    n_workers: int | None = None,
    denovo_min_tx_pctile: float = 25.0,
    denovo_filter_report_fname: str | None = None,
):
    timings: dict[str, float] = {}

    ## Get number of parallel workers
    if n_workers is None:
        n_workers = get_num_cpus()

    ## Ensure work_dir exists
    os.makedirs(work_dir, exist_ok=True)

    ## Get mappings for genes
    idx2genes = {i: x for i, x in enumerate(dataset.genes)}

    ## Step 1: Make tx unique
    logger.info("Step 1: getting unique tx...")
    step_start = time.perf_counter()
    temp_fname1 = "temp1-" + uuid.uuid4().hex + ".parquet"
    temp_path1 = os.path.join(work_dir, temp_fname1)
    filter_unique_and_remap_tx(input_path=input_files, output_path=temp_path1, idx2genes=idx2genes)
    _log_step("Step 1", step_start, temp_path1, timings)

    ## Split noise rows from non-noise before Step 2 (DBSCAN runs only on real transcripts)
    _step1_df = pl.scan_parquet(temp_path1)

    temp_noise_meta_path = os.path.join(
        work_dir, "temp_noise_meta-" + uuid.uuid4().hex + ".parquet"
    )
    _step1_df.select(
        ["transcript_id", "is_noise", "tx_noise_prob_0", "tx_noise_prob_1"]
    ).sink_parquet(temp_noise_meta_path)

    temp_noise_rows_path = os.path.join(
        work_dir, "temp_noise_rows-" + uuid.uuid4().hex + ".parquet"
    )
    _step1_df.filter(pl.col("is_noise")).select(
        [
            "transcript_id",
            "feature_name",
            "x",
            "y",
            "x_initial",
            "y_initial",
            "is_noise",
            "tx_noise_prob_0",
            "tx_noise_prob_1",
        ]
    ).sink_parquet(temp_noise_rows_path)

    temp_path1b = os.path.join(work_dir, "temp1b-" + uuid.uuid4().hex + ".parquet")
    _step1_df.filter(~pl.col("is_noise")).drop(
        ["is_noise", "tx_noise_prob_0", "tx_noise_prob_1"]
    ).sink_parquet(temp_path1b)

    remove(temp_path1)

    ## Step 2: Build double grid (non-noise transcripts only)
    logger.info("Step 2: building double grid...")
    step_start = time.perf_counter()
    temp_fname2 = "temp2-" + uuid.uuid4().hex
    temp_path2 = os.path.join(work_dir, temp_fname2)
    double_grid_and_group(
        input_path=temp_path1b, output_path=temp_path2, cluster_distance=cluster_distance
    )
    _log_step("Step 2", step_start, temp_path2, timings)

    ## Remove non-noise intermediate file
    remove(temp_path1b)

    ## Step 3: Cluster within each grid section
    logger.info("Step 3: clustering within grid sections...")
    step_start = time.perf_counter()
    temp_fname3 = "temp3-" + uuid.uuid4().hex
    temp_path3 = os.path.join(work_dir, temp_fname3)
    grid_level_cluster(
        input_path=temp_path2,
        output_path=temp_path3,
        epsilon=cluster_distance,
        n_workers=n_workers,
    )
    _log_step("Step 3", step_start, temp_path3, timings)

    ## Remove second intermediate file
    remove(temp_path2)

    ## Step 4: Link clusters across grid versions
    logger.info("Step 4: linking clusters across grids...")
    step_start = time.perf_counter()
    temp_fname4 = "temp4-" + uuid.uuid4().hex + ".parquet"
    temp_path4 = os.path.join(work_dir, temp_fname4)
    grid_connected_components(input_path=temp_path3, output_path=temp_path4)
    _log_step("Step 4", step_start, temp_path4, timings)

    ## Remove third intermediate file
    remove(temp_path3)

    ## Step 5: Map clusters to cells
    logger.info("Step 5: mapping clusters to cells...")
    step_start = time.perf_counter()
    temp_fname5 = "temp5-" + uuid.uuid4().hex + ".parquet"
    temp_path5 = os.path.join(work_dir, temp_fname5)
    map_clusters_to_cells(
        input_path=temp_path4,
        nuc_centroid=dataset.nuc_xy,
        nuc_ellipse_mat=dataset.nuc_ellipse_mat,
        nuc_cell_id=dataset.nuc_cell_id,
    ).sink_parquet(temp_path5)
    _log_step("Step 5", step_start, temp_path5, timings)

    ## Remove fourth intermediate file
    remove(temp_path4)

    ## Step 6: Compute distances to original positions (non-noise transcripts)
    if original_tx_fname is not None:
        logger.info("Step 6: computing distances to original positions...")
        step_start = time.perf_counter()
        calculate_tx_distances(
            input_path=temp_path5,
            output_path=output_fname,
            original_tx_fname=original_tx_fname,
        )
        _log_step("Step 6", step_start, output_fname, timings)
        remove(temp_path5)

        ## Also compute distances for noise rows so they have consistent columns
        temp_noise_with_dist_path = os.path.join(
            work_dir, "temp_noise_with_dist-" + uuid.uuid4().hex + ".parquet"
        )
        calculate_tx_distances(
            input_path=temp_noise_rows_path,
            output_path=temp_noise_with_dist_path,
            original_tx_fname=original_tx_fname,
        )
        remove(temp_noise_rows_path)
        temp_noise_rows_path = temp_noise_with_dist_path
    else:
        logger.info("Step 6: skipping distance computation (no original transcripts provided)...")
        os.rename(temp_path5, output_fname)

    ## Step 7: Create AnnData file (from non-noise transcripts only)
    if anndata_fname is not None:
        logger.info("Step 7: creating AnnData file...")
        step_start = time.perf_counter()
        create_anndata(
            ef_tx_fname=output_fname,
            output_fname=anndata_fname,
            dataset=dataset,
            denovo_min_tx_pctile=denovo_min_tx_pctile,
            denovo_filter_report_fname=denovo_filter_report_fname,
        )
        _log_step("Step 7", step_start, anndata_fname, timings)

    ## Amend ef.parquet: join noise columns back onto non-noise rows, then append noise rows
    logger.info("Amending ef.parquet with noise columns and appending noise rows...")
    noise_meta = pl.scan_parquet(temp_noise_meta_path).with_columns(
        pl.col("transcript_id").cast(pl.UInt64)
    )
    ef_ldf = pl.scan_parquet(output_fname).join(noise_meta, on="transcript_id", how="left")
    noise_rows_ldf = pl.scan_parquet(temp_noise_rows_path).with_columns(
        pl.col("transcript_id").cast(pl.UInt64)
    )
    combined_ldf = pl.concat([ef_ldf, noise_rows_ldf], how="diagonal")
    tmp_out = output_fname + ".tmp"
    combined_ldf.sink_parquet(tmp_out)
    os.replace(tmp_out, output_fname)

    remove(temp_noise_meta_path)
    remove(temp_noise_rows_path)

    logger.info("Step timing summary (seconds): %s", timings)
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
