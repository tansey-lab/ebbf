"""Tests for post-processing stages."""

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import pytest

from ebbf.inference.post_process import double_grid_and_group, map_clusters_to_cells


def _make_input_parquet(path: str, n_rows: int = 100_000, seed: int = 42):
    """Create a synthetic input parquet file mimicking Step 1 output.

    Generates transcript-like data scattered across a spatial region large
    enough to produce multiple grid cells at the default box_width=50.
    """
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {
            "transcript_id": np.arange(n_rows),
            "x": rng.uniform(0, 500, n_rows).astype(np.float32),
            "y": rng.uniform(0, 500, n_rows).astype(np.float32),
            "feature_name": rng.choice(["GeneA", "GeneB", "GeneC"], n_rows),
        }
    )
    df.write_parquet(path)
    return df


class TestDoubleGridAndGroup:
    """Tests for the chunked, disk-spilling double_grid_and_group function."""

    @pytest.fixture
    def work_dir(self, tmp_path):
        """Provide a temporary working directory with a synthetic input file."""
        input_path = str(tmp_path / "input.parquet")
        _make_input_parquet(input_path)
        return tmp_path, input_path

    def test_small_min_size_produces_more_partitions(self, work_dir):
        """Lowering min_size should produce more super_grid_label partitions."""
        tmp_path, input_path = work_dir

        output_large = str(tmp_path / "grid_large_min")
        output_small = str(tmp_path / "grid_small_min")

        double_grid_and_group(
            input_path=input_path,
            output_path=output_large,
            cluster_distance=1.5,
            min_size=50_000,
        )
        double_grid_and_group(
            input_path=input_path,
            output_path=output_small,
            cluster_distance=1.5,
            min_size=1_000,
        )

        frags_large = len(list(ds.dataset(output_large, format="parquet").get_fragments()))
        frags_small = len(list(ds.dataset(output_small, format="parquet").get_fragments()))

        assert frags_small >= frags_large


def _make_cluster_input_parquet(path: str):
    """Create synthetic Step 4 output with multiple row groups."""
    rows = 4000
    ef_cluster_id = np.repeat(np.arange(20), rows // 20)
    x_base = np.tile(np.linspace(0.0, 100.0, rows // 20), 20)
    y_base = np.tile(np.linspace(50.0, 150.0, rows // 20), 20)
    df = pl.DataFrame(
        {
            "transcript_id": np.arange(rows, dtype=np.int64),
            "feature_name": np.where(np.arange(rows) % 2 == 0, "GeneA", "GeneB"),
            "x": x_base + (ef_cluster_id * 0.2),
            "y": y_base + (ef_cluster_id * 0.1),
            "x_initial": x_base,
            "y_initial": y_base,
            "ef_cluster_id": ef_cluster_id.astype(np.int64),
        }
    )
    df.write_parquet(path, row_group_size=250)


def test_map_clusters_to_cells_parallel_matches_single(tmp_path):
    """Parallel fragment mapping should match single-worker output exactly."""
    input_path = str(tmp_path / "cluster_input.parquet")
    out_single = str(tmp_path / "mapped_single.parquet")
    out_multi = str(tmp_path / "mapped_multi.parquet")
    _make_cluster_input_parquet(input_path)

    nuc_centroid = np.array([[10.0, 60.0], [40.0, 90.0], [80.0, 130.0]])
    # Axis-aligned ellipses represented by linear transforms
    nuc_ellipse_mat = np.array(
        [
            [[5.0, 0.0], [0.0, 4.0]],
            [[6.0, 0.0], [0.0, 5.0]],
            [[7.0, 0.0], [0.0, 6.0]],
        ]
    )
    nuc_cell_id = np.array([101, 102, 103])

    map_clusters_to_cells(
        input_path=input_path,
        output_path=out_single,
        nuc_centroid=nuc_centroid,
        nuc_ellipse_mat=nuc_ellipse_mat,
        nuc_cell_id=nuc_cell_id,
        n_workers=1,
    )
    map_clusters_to_cells(
        input_path=input_path,
        output_path=out_multi,
        nuc_centroid=nuc_centroid,
        nuc_ellipse_mat=nuc_ellipse_mat,
        nuc_cell_id=nuc_cell_id,
        n_workers=2,
    )

    sort_cols = ["transcript_id", "ef_cluster_id"]
    df_single = pl.read_parquet(out_single).sort(sort_cols)
    df_multi = pl.read_parquet(out_multi).sort(sort_cols)
    for col in df_single.columns:
        if df_single[col].dtype.is_float():
            np.testing.assert_allclose(
                df_single[col].to_numpy(),
                df_multi[col].to_numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            assert df_single[col].equals(df_multi[col])
