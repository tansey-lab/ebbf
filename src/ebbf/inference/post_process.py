import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import shutil
import time
import uuid

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from tqdm import tqdm

from ebbf.compat import get_num_cpus
from ebbf.data.st_dataset import STDataset
from ebbf.inference.infer import DEDUP_SORT_COLS

logger = logging.getLogger(__name__)

NOISE_COLS = ["is_noise", "tx_noise_prob_0", "tx_noise_prob_1"]


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


_CLUSTER_EPSILON = None
_CLUSTER_OUTPUT_PATH = None
_CLUSTER_INPUT_PATH = None


def _init_cluster_worker(epsilon: float, output_path: str, input_path: str):
    global _CLUSTER_EPSILON, _CLUSTER_OUTPUT_PATH, _CLUSTER_INPUT_PATH
    _CLUSTER_EPSILON = epsilon
    _CLUSTER_OUTPUT_PATH = output_path
    _CLUSTER_INPUT_PATH = input_path


def _single_linkage(xy: np.ndarray, epsilon: float) -> np.ndarray:
    """Connected components of the epsilon-neighborhood graph (single-linkage clusters)."""
    n = xy.shape[0]
    pairs = KDTree(xy).query_pairs(epsilon, output_type="ndarray")
    if pairs.size == 0:
        return np.arange(n, dtype=np.int64)
    graph = coo_array(
        (np.ones(pairs.shape[0], dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, cluster_id = connected_components(graph, directed=False)
    return cluster_id.astype(np.int64, copy=False)


def _cluster_fragment(fragment_path: str) -> str:
    # Each file may contain multiple super_grid_labels -- cluster each label separately
    df = pl.read_parquet(fragment_path)
    results = []
    for group_df in df.partition_by("super_grid_label", maintain_order=False):
        xy = group_df.select(["x", "y"]).to_numpy().astype(np.float64)
        cluster_ids = _single_linkage(xy, _CLUSTER_EPSILON)
        results.append(group_df.with_columns(pl.Series("cluster_id", cluster_ids, dtype=pl.Int64)))

    out_fname = os.path.join(
        _CLUSTER_OUTPUT_PATH, os.path.relpath(fragment_path, _CLUSTER_INPUT_PATH)
    )
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    pl.concat(results).write_parquet(out_fname)
    return out_fname


###############
### STEP 1 ###
##############
def filter_unique_and_remap_tx(input_path: str | list[str], output_path: str, idx2genes: dict):
    logger.info("filter_unique_and_remap_tx: deduplicating transcripts and remapping gene indices")

    ## Keep the copy of each transcript from the patch whose center it is closest to,
    ## with the same tie-breaking as inference
    df = (
        pl.scan_parquet(input_path)
        .sort(DEDUP_SORT_COLS)
        .unique(subset="tx_idx", keep="first")
        .drop(["tx_initial_x", "tx_initial_y", *DEDUP_SORT_COLS[1:]])
    )

    ## Rename to be consistent with everything else
    df = df.rename({"tx_idx": "transcript_id", "tx_x": "x", "tx_y": "y"})

    ## Replace gene indices with gene names
    df = df.with_columns(
        pl.col("tx_gene_idx").replace_strict(idx2genes, default="None").alias("feature_name")
    )

    ## Noise call from the noise head at t=1
    df = df.with_columns((pl.col("tx_noise_prob_1") > 0.5).alias("is_noise"))

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
    """Single-linkage clustering (KDTree pairs + connected components) per super_grid_label."""
    source_dataset = ds.dataset(input_path, format="parquet")
    fragment_paths = [f.path for f in source_dataset.get_fragments()]

    ## spawn avoids fork-after-polars/KDTree deadlocks
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_cluster_worker,
        initargs=(epsilon, output_path, input_path),
    ) as executor:
        futures = [executor.submit(_cluster_fragment, p) for p in fragment_paths]
        with tqdm(total=len(futures)) as pbar:
            for fut in cf.as_completed(futures):
                fut.result()  # re-raises worker exceptions in the main process
                pbar.update(1)

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
    """Run steps 1–4: dedup, grid, clustering, connected components.

    If ``noise_meta_fname`` is provided, noise transcripts are separated before
    clustering and their metadata (positions, is_noise, tx_noise_prob_*) is
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

    ## Save noise metadata and filter to non-noise before clustering.
    if noise_meta_fname is not None:
        _step1_df = pl.scan_parquet(temp_path1)
        _step1_df.select(["transcript_id", "feature_name", "x", "y", *NOISE_COLS]).sink_parquet(
            noise_meta_fname
        )
        logger.info("Noise metadata written → %s", noise_meta_fname)

        temp_path1b = os.path.join(work_dir, "temp1b-" + uuid.uuid4().hex + ".parquet")
        _step1_df.filter(~pl.col("is_noise")).drop(NOISE_COLS).sink_parquet(temp_path1b)
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
    original_tx_fname: str,
    output_fname: str = "ef.parquet",
    anndata_fname: str | None = "ef.h5ad",
    noise_meta_fname: str | None = None,
    work_dir: str = ".",
    cd_cutoff: float = 2.0,
    denovo_min_tx_pctile: float = 0.0,
    min_tx_count: int = 50,
):
    """Steps 6–7: attach original positions, classify clusters, write ef.h5ad and ef.parquet.

    ``ef.parquet`` gets an ``ef_cell_id`` column (the cell each transcript is assigned
    to; ``UNASSIGNED`` for noise transcripts). If ``noise_meta_fname`` is given, noise
    probabilities are joined back on and the noise transcripts are appended.
    """
    os.makedirs(work_dir, exist_ok=True)
    temp_path = os.path.join(work_dir, "temp-merge-" + uuid.uuid4().hex + ".parquet")
    calculate_tx_distances(pl.scan_parquet(mapped_files), original_tx_fname).sink_parquet(temp_path)

    cluster_info, tx_threshold = classify_clusters(
        ef_tx_fname=temp_path,
        nuc_cell_counts=dataset.nuc_cell_counts,
        cd_cutoff=cd_cutoff,
        denovo_min_tx_pctile=denovo_min_tx_pctile,
        min_tx_count=min_tx_count,
    )
    logger.info(
        "Cell categories: %s (de novo threshold: %.1f transcripts)",
        dict(cluster_info["cell_category"].value_counts().iter_rows()),
        tx_threshold,
    )

    if anndata_fname is not None:
        create_anndata(
            ef_tx_fname=temp_path,
            output_fname=anndata_fname,
            dataset=dataset,
            cluster_info=cluster_info,
            tx_threshold=tx_threshold,
            cd_cutoff=cd_cutoff,
            denovo_min_tx_pctile=denovo_min_tx_pctile,
            min_tx_count=min_tx_count,
        )

    ef_ldf = pl.scan_parquet(temp_path)
    if noise_meta_fname is not None:
        noise_meta = pl.scan_parquet(noise_meta_fname).with_columns(
            pl.col("transcript_id").cast(pl.UInt64)
        )
        ef_ldf = ef_ldf.join(
            noise_meta.select(["transcript_id", *NOISE_COLS]), on="transcript_id", how="left"
        )
        noise_rows = calculate_tx_distances(
            noise_meta.filter(pl.col("is_noise")), original_tx_fname
        )
        ef_ldf = pl.concat([ef_ldf, noise_rows], how="diagonal")

    cluster_to_cell = cluster_info.select(
        "ef_cluster_id", pl.col("unified_cell_id").alias("ef_cell_id")
    ).lazy()
    ef_ldf = ef_ldf.join(cluster_to_cell, on="ef_cluster_id", how="left").with_columns(
        pl.col("ef_cell_id").fill_null(pl.lit("UNASSIGNED"))
    )
    ef_ldf.sink_parquet(output_fname)
    remove(temp_path)
    return "Success!"


###############
###############


###############
### STEP 6 ###
##############


def calculate_tx_distances(ldf: pl.LazyFrame, original_tx_fname: str) -> pl.LazyFrame:
    """Attach original positions (x_initial, y_initial), overlaps_nucleus and the
    distance between flowed and original positions."""
    tx_df = (
        pl.scan_parquet(original_tx_fname)
        .select(["transcript_id", "x", "y", "overlaps_nucleus"])
        .rename({"x": "x_initial", "y": "y_initial"})
    )

    return (
        ldf.with_columns(pl.col("transcript_id").cast(pl.UInt64))
        .join(tx_df, on="transcript_id", how="inner")
        .with_columns(
            distance=(
                (pl.col("x") - pl.col("x_initial")).pow(2)
                + (pl.col("y") - pl.col("y_initial")).pow(2)
            ).sqrt()
        )
    )


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


def classify_clusters(
    ef_tx_fname: str,
    nuc_cell_counts: np.ndarray,
    cd_cutoff: float = 2.0,
    denovo_min_tx_pctile: float = 0.0,
    min_tx_count: int = 50,
) -> tuple[pl.DataFrame, float]:
    """Classify each ef_cluster_id and give it a cell identifier.

    1. nucleus_anchored: mahalanobis_distance <= 1 and centroid_distance <= cd_cutoff.
       All such clusters of a nucleus merge into one cell named by the nucleus' cell_id.
    2. Every other cluster is its own cell, split by size against
       ``max(denovo_min_tx_pctile-th percentile of input-segmentation cell counts, min_tx_count)``:
       ``denovo_{ef_cluster_id}`` if n_tx >= threshold, else ``fragment_{ef_cluster_id}``.

    Returns per-cluster aggregates (n_tx and coordinate / displacement sums) and the threshold.
    """
    input_tx_nz = np.asarray(nuc_cell_counts)
    input_tx_nz = input_tx_nz[input_tx_nz > 0]
    pctile_value = (
        float(np.percentile(input_tx_nz, denovo_min_tx_pctile))
        if denovo_min_tx_pctile > 0 and input_tx_nz.size > 0
        else 0.0
    )
    tx_threshold = max(pctile_value, float(min_tx_count))

    anchored = (pl.col("mahalanobis_distance") <= 1) & (pl.col("centroid_distance") <= cd_cutoff)
    cluster_info = (
        pl.scan_parquet(ef_tx_fname)
        .group_by("ef_cluster_id")
        .agg(
            pl.col("cell_id").first(),
            pl.col("centroid_distance").first(),
            pl.col("mahalanobis_distance").first(),
            pl.len().alias("n_tx"),
            pl.col("x").sum().alias("x_sum"),
            pl.col("y").sum().alias("y_sum"),
            pl.col("x_initial").sum().alias("x_initial_sum"),
            pl.col("y_initial").sum().alias("y_initial_sum"),
            pl.col("distance").sum().alias("distance_sum"),
        )
        .with_columns(
            pl.when(anchored)
            .then(pl.lit("nucleus_anchored"))
            .when(pl.col("n_tx") >= tx_threshold)
            .then(pl.lit("denovo"))
            .otherwise(pl.lit("fragment"))
            .alias("cell_category")
        )
        .with_columns(
            pl.when(pl.col("cell_category") == "nucleus_anchored")
            .then(pl.col("cell_id").cast(pl.Utf8))
            .otherwise(pl.col("cell_category") + "_" + pl.col("ef_cluster_id").cast(pl.Utf8))
            .alias("unified_cell_id")
        )
        .collect(engine="streaming")
    )
    return cluster_info, tx_threshold


def create_anndata(
    ef_tx_fname: str,
    output_fname: str,
    dataset: STDataset,
    cluster_info: pl.DataFrame,
    tx_threshold: float,
    cd_cutoff: float = 2.0,
    denovo_min_tx_pctile: float = 0.0,
    min_tx_count: int = 50,
):
    """Create the cell x gene AnnData from the transcript assignment table.

    Cells are the ``unified_cell_id`` of ``classify_clusters``; every nucleus in the
    dataset gets a row (empty ones as nucleus_anchored with zero counts). Nothing is
    filtered: select on ``obs["cell_category"]`` after loading.

    obs: cell_category; parent_cell_id (the nucleus for nucleus_anchored cells);
    x_centroid / y_centroid (nucleus centroid for nucleus_anchored cells, mean flowed
    position otherwise); n_tx; mean_displacement; n_clusters; and, from the cell's
    largest cluster, ef_cluster_id, nearest_centroid_cell_id / _x / _y,
    centroid_distance and mahalanobis_distance.
    obsm: spatial (= x/y_centroid), prediffusion_centroid (mean original position).
    """
    nuc_id2xy = dict(zip(map(str, dataset.nuc_cell_id), np.asarray(dataset.nuc_xy)))

    ## Per-cell aggregates; debug fields come from each cell's largest cluster
    cells = (
        cluster_info.sort(["n_tx", "ef_cluster_id"], descending=[True, False])
        .group_by("unified_cell_id", maintain_order=True)
        .agg(
            pl.col("cell_category").first(),
            pl.col("ef_cluster_id").first(),
            pl.col("cell_id").first().cast(pl.Utf8).alias("nearest_centroid_cell_id"),
            pl.col("centroid_distance").first(),
            pl.col("mahalanobis_distance").first(),
            pl.len().alias("n_clusters"),
            pl.col("n_tx").sum().alias("_n_tx"),
            pl.col("x_sum").sum(),
            pl.col("y_sum").sum(),
            pl.col("x_initial_sum").sum(),
            pl.col("y_initial_sum").sum(),
            pl.col("distance_sum").sum(),
        )
        .with_columns(
            (pl.col(c + "_sum") / pl.col("_n_tx")).alias(c)
            for c in ["x", "y", "x_initial", "y_initial", "distance"]
        )
        .to_pandas()
        .set_index("unified_cell_id")
    )

    ## Every nucleus gets a row, including those that received no transcripts
    all_cell_ids = sorted(set(cells.index) | set(nuc_id2xy))
    cell_id2idx = {cid: i for i, cid in enumerate(all_cell_ids)}

    ## Build the count matrix
    annotated_ldf = pl.scan_parquet(ef_tx_fname).join(
        cluster_info.select(["ef_cluster_id", "unified_cell_id"]).lazy(),
        on="ef_cluster_id",
        how="inner",
    )
    adata = _create_anndata_from_lazyframe(
        annotated_ldf,
        cell_id2idx=cell_id2idx,
        cell_id_col="unified_cell_id",
        gene_col="feature_name",
    )

    obs = cells.reindex(adata.obs.index)
    is_empty_nucleus = obs["cell_category"].isna()
    obs.loc[is_empty_nucleus, "cell_category"] = "nucleus_anchored"
    is_anchored = (obs["cell_category"] == "nucleus_anchored").to_numpy()

    nuc_xy = np.array([nuc_id2xy.get(cid, (np.nan, np.nan)) for cid in obs.index], dtype=float)
    x_centroid = np.where(is_anchored, nuc_xy[:, 0], obs["x"].to_numpy(dtype=float))
    y_centroid = np.where(is_anchored, nuc_xy[:, 1], obs["y"].to_numpy(dtype=float))
    nearest_xy = np.array(
        [nuc_id2xy.get(cid, (np.nan, np.nan)) for cid in obs["nearest_centroid_cell_id"]],
        dtype=float,
    )

    adata.obs["cell_category"] = obs["cell_category"].to_numpy()
    adata.obs["parent_cell_id"] = np.where(is_anchored, obs.index.to_numpy(), None)
    adata.obs["x_centroid"] = x_centroid
    adata.obs["y_centroid"] = y_centroid
    adata.obs["n_tx"] = np.asarray(adata.X.sum(axis=1)).flatten().astype(int)
    adata.obs["mean_displacement"] = obs["distance"].to_numpy(dtype=float)
    adata.obs["n_clusters"] = obs["n_clusters"].fillna(0).to_numpy(dtype=int)
    adata.obs["ef_cluster_id"] = obs["ef_cluster_id"].to_numpy(dtype=float)
    adata.obs["nearest_centroid_cell_id"] = obs["nearest_centroid_cell_id"].to_numpy()
    adata.obs["nearest_centroid_x"] = nearest_xy[:, 0]
    adata.obs["nearest_centroid_y"] = nearest_xy[:, 1]
    adata.obs["centroid_distance"] = obs["centroid_distance"].to_numpy(dtype=float)
    adata.obs["mahalanobis_distance"] = obs["mahalanobis_distance"].to_numpy(dtype=float)

    adata.obsm["spatial"] = np.column_stack([x_centroid, y_centroid])
    adata.obsm["prediffusion_centroid"] = np.column_stack(
        [obs["x_initial"].to_numpy(dtype=float), obs["y_initial"].to_numpy(dtype=float)]
    )

    adata.uns["denovo_tx_threshold"] = float(tx_threshold)
    adata.uns["denovo_tx_pctile"] = float(denovo_min_tx_pctile)
    adata.uns["min_tx_count"] = int(min_tx_count)
    adata.uns["cd_cutoff"] = float(cd_cutoff)

    adata.write(output_fname, compression="lzf")
    logger.info("AnnData → %s (%d cells)", output_fname, adata.n_obs)
    return adata


###############
###############


def post_process(
    input_files: list[str],
    dataset: STDataset,
    cluster_distance: float,
    original_tx_fname: str,
    output_fname: str = "ef.parquet",
    anndata_fname: str | None = "ef.h5ad",
    work_dir: str = ".",
    n_workers: int | None = None,
    cd_cutoff: float = 2.0,
    denovo_min_tx_pctile: float = 0.0,
    min_tx_count: int = 50,
):
    """Full single-node pipeline: prepare (steps 1–4), map (step 5), merge (steps 6–7)."""
    os.makedirs(work_dir, exist_ok=True)
    tag = uuid.uuid4().hex
    step4_path = os.path.join(work_dir, f"temp4-{tag}.parquet")
    noise_meta_path = os.path.join(work_dir, f"temp_noise_meta-{tag}.parquet")
    mapped_path = os.path.join(work_dir, f"temp5-{tag}.parquet")

    post_process_prepare(
        input_files=input_files,
        dataset=dataset,
        cluster_distance=cluster_distance,
        output_fname=step4_path,
        noise_meta_fname=noise_meta_path,
        work_dir=work_dir,
        n_workers=n_workers,
    )

    logger.info("Step 5: mapping clusters to cells...")
    map_clusters_to_cells(
        input_path=step4_path,
        nuc_centroid=dataset.nuc_xy,
        nuc_ellipse_mat=dataset.nuc_ellipse_mat,
        nuc_cell_id=dataset.nuc_cell_id,
    ).sink_parquet(mapped_path)
    remove(step4_path)

    post_process_merge(
        mapped_files=[mapped_path],
        dataset=dataset,
        original_tx_fname=original_tx_fname,
        output_fname=output_fname,
        anndata_fname=anndata_fname,
        noise_meta_fname=noise_meta_path,
        work_dir=work_dir,
        cd_cutoff=cd_cutoff,
        denovo_min_tx_pctile=denovo_min_tx_pctile,
        min_tx_count=min_tx_count,
    )
    remove(mapped_path)
    remove(noise_meta_path)

    logger.info("DONE!")
    return "Success!"
