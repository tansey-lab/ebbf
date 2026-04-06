"""Tests for the double_grid_and_group stage of post-processing."""

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import pytest

from ebbf.inference.post_process import double_grid_and_group


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
