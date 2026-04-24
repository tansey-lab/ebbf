"""Diagnostic: does `double_grid_and_group` leak, or does the allocator just retain pages?

Generates a synthetic parquet file, runs the streaming loop from
`double_grid_and_group` with a small batch_size so we see many iterations, and
reports three metrics per batch:

  - RSS          : resident set size (what the OS sees — what `top` shows)
  - py_heap      : bytes tracked by Python's tracemalloc (true Python-live bytes)
  - rss_after_trim: RSS after malloc_trim(0) on Linux (NA on macOS)

Interpretation:
  * RSS grows unbounded + py_heap flat    → allocator retention (not a leak)
  * RSS grows unbounded + py_heap grows   → real reference leak
  * rss_after_trim << RSS                 → glibc is holding freed pages

Usage:
  python scripts/memcheck_double_grid.py --n-rows 5_000_000 --batch-size 500_000
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import os
import sys
import tempfile
import tracemalloc

import numpy as np
import polars as pl
import psutil


def _load_malloc_trim():
    """Return a callable that invokes malloc_trim(0) on glibc, else None."""
    if sys.platform != "linux":
        return None
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return None
    libc = ctypes.CDLL(libc_name)
    if not hasattr(libc, "malloc_trim"):
        return None
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    return lambda: libc.malloc_trim(0)


def _make_input(path: str, n_rows: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    # Spatial extent large enough to produce many grid cells at box_width=50.
    df = pl.DataFrame(
        {
            "transcript_id": np.arange(n_rows, dtype=np.int64),
            "x": rng.uniform(0, 5000, n_rows).astype(np.float32),
            "y": rng.uniform(0, 5000, n_rows).astype(np.float32),
            "feature_name": rng.choice([f"Gene{i}" for i in range(50)], n_rows),
        }
    )
    df.write_parquet(path)


def run(n_rows: int, batch_size: int) -> None:
    proc = psutil.Process()
    malloc_trim = _load_malloc_trim()

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.parquet")
        out_path = os.path.join(tmp, "out")
        os.makedirs(out_path, exist_ok=True)

        print(f"Generating {n_rows:,} rows → {in_path}", flush=True)
        _make_input(in_path, n_rows)
        gc.collect()

        tracemalloc.start()
        rss0 = proc.memory_info().rss
        py0, _ = tracemalloc.get_traced_memory()
        print(f"Baseline: RSS={rss0 / 1e6:8.1f} MB  py_heap={py0 / 1e6:8.1f} MB", flush=True)

        # Import here so baseline doesn't include module import memory.
        import pyarrow.parquet as pq

        from ebbf.inference.post_process import _add_grid_columns, group_labels_min_size

        box_width = 50.0
        min_size = 25_000
        n_file_groups = 100
        offset = 0.0
        grid_version = 0

        # --- Phase 1: compute mapping (one-shot; we don't instrument this part) ---
        ldf = pl.scan_parquet(in_path)
        counts = (
            _add_grid_columns(ldf, box_width, offset)
            .group_by("grid_name")
            .len()
            .collect(engine="streaming")
            .sort(by="len")
        )
        labels = group_labels_min_size(counts["len"].to_numpy(), min_size)
        mapping_df = pl.DataFrame(
            {"grid_name": counts["grid_name"].to_list(), "super_grid_label": labels}
        )
        n_partitions = int(mapping_df["super_grid_label"].max()) + 1
        n_groups = min(n_file_groups, n_partitions)
        mapping_df = mapping_df.with_columns(
            (pl.col("super_grid_label") % n_groups).cast(pl.Int32).alias("file_group")
        )
        del counts, labels, ldf
        gc.collect()

        rss1 = proc.memory_info().rss
        py1, _ = tracemalloc.get_traced_memory()
        print(
            f"After phase1: RSS={rss1 / 1e6:8.1f} MB  py_heap={py1 / 1e6:8.1f} MB",
            flush=True,
        )

        # --- Phase 2: streaming loop, identical to post_process.py ---
        writers: dict[int, pq.ParquetWriter] = {}
        parquet_file = pq.ParquetFile(in_path)
        print(
            f"\n{'batch':>5} {'rows':>10} {'RSS_MB':>10} {'py_heap_MB':>12} "
            f"{'delta_RSS':>10} {'RSS_trim':>10}",
            flush=True,
        )
        import pyarrow as pa

        def run_batch_old(record_batch):
            batch = (
                pl.from_arrow(record_batch)
                .pipe(lambda df: _add_grid_columns(df, box_width, offset))
                .join(mapping_df, on="grid_name", how="left")
                .with_columns(pl.lit(grid_version).cast(pl.Int32).alias("grid_version"))
                .sort("file_group")
            )
            table = batch.to_arrow()
            fg_col = batch["file_group"].to_numpy()
            boundaries = np.where(np.diff(fg_col))[0] + 1
            starts = np.concatenate([[0], boundaries])
            ends = np.concatenate([boundaries, [len(batch)]])
            for start, end in zip(starts, ends):
                fg = int(fg_col[start])
                final_part_dir = os.path.join(
                    out_path, f"grid_version={grid_version}", f"file_group={fg}"
                )
                os.makedirs(final_part_dir, exist_ok=True)
                final_path = os.path.join(final_part_dir, "data.parquet")
                slice_table = table.slice(start, end - start)
                writer = writers.get(fg)
                if writer is None:
                    writer = pq.ParquetWriter(final_path, slice_table.schema)
                    writers[fg] = writer
                writer.write_table(slice_table)

        def run_batch_new(record_batch):
            batch = (
                pl.from_arrow(record_batch)
                .pipe(lambda df: _add_grid_columns(df, box_width, offset))
                .join(mapping_df, on="grid_name", how="left")
                .with_columns(pl.lit(grid_version).cast(pl.Int32).alias("grid_version"))
            )
            fg_col = batch["file_group"].to_numpy()
            table = batch.to_arrow()
            del batch
            order = np.argsort(fg_col, kind="stable")
            fg_sorted = fg_col[order]
            boundaries = np.where(np.diff(fg_sorted))[0] + 1
            starts = np.concatenate([[0], boundaries])
            ends = np.concatenate([boundaries, [len(fg_sorted)]])
            for start, end in zip(starts, ends):
                fg = int(fg_sorted[start])
                final_part_dir = os.path.join(
                    out_path, f"grid_version={grid_version}", f"file_group={fg}"
                )
                os.makedirs(final_part_dir, exist_ok=True)
                final_path = os.path.join(final_part_dir, "data.parquet")
                slice_table = table.take(pa.array(order[start:end]))
                writer = writers.get(fg)
                if writer is None:
                    writer = pq.ParquetWriter(final_path, slice_table.schema)
                    writers[fg] = writer
                writer.write_table(slice_table)

        mode = os.environ.get("MEMCHECK_MODE", "new")
        run_batch = run_batch_new if mode == "new" else run_batch_old
        print(f"Mode: {mode}", flush=True)

        try:
            prev_rss = proc.memory_info().rss
            peak_rss_intra = 0
            for i, record_batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                # Sample RSS mid-batch to catch the transient peak.
                run_batch(record_batch)
                peak_now = proc.memory_info().rss
                if peak_now > peak_rss_intra:
                    peak_rss_intra = peak_now
                del record_batch
                gc.collect()

                rss = proc.memory_info().rss
                py, _ = tracemalloc.get_traced_memory()
                rss_trim = "-"
                if malloc_trim is not None:
                    malloc_trim()
                    rss_trim = f"{proc.memory_info().rss / 1e6:.1f}"
                print(
                    f"{i:>5} {batch_size:>10,} {rss / 1e6:>10.1f} "
                    f"{py / 1e6:>12.1f} {(rss - prev_rss) / 1e6:>+10.1f} {rss_trim:>10}",
                    flush=True,
                )
                prev_rss = rss
            print(f"\nPeak RSS observed during loop: {peak_rss_intra / 1e6:.1f} MB", flush=True)
        finally:
            for writer in writers.values():
                writer.close()

        gc.collect()
        rss_end = proc.memory_info().rss
        py_end, py_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("")
        print(
            f"End:       RSS={rss_end / 1e6:8.1f} MB  py_heap={py_end / 1e6:8.1f} MB  "
            f"py_peak={py_peak / 1e6:.1f} MB",
            flush=True,
        )
        print(
            f"Net growth: RSS +{(rss_end - rss0) / 1e6:.1f} MB  "
            f"py_heap +{(py_end - py0) / 1e6:.1f} MB",
            flush=True,
        )
        print("")
        print("Interpretation:")
        print("  py_heap ~flat but RSS grows   → allocator retention (not a leak)")
        print("  py_heap grows with RSS        → real reference leak")
        if malloc_trim is not None:
            print("  RSS_trim << RSS               → glibc holds freed pages")
        else:
            print("  (malloc_trim unavailable on this platform — macOS uses its own allocator)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=5_000_000)
    p.add_argument("--batch-size", type=int, default=500_000)
    args = p.parse_args()
    run(args.n_rows, args.batch_size)
