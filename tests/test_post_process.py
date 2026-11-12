"""Tests for post-processing stages."""

import anndata
import numpy as np
import polars as pl
import pyarrow.dataset as ds
import pytest

from ebbf.data.st_dataset import STDataset
from ebbf.inference.post_process import (
    calculate_tx_distances,
    ck_decode_cluster_id,
    ck_decode_grid_version,
    ck_decode_super_grid_label,
    ck_encode,
    classify_clusters,
    compute_cluster_to_cell_mapping,
    double_grid_and_group,
    map_clusters_to_cells,
    post_process_merge,
)


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


class TestClusterKeyPacking:
    """Round-trip and boundary tests for the cluster-key bit-packing helpers."""

    def _make_df(self, sgl, gv, cid) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "super_grid_label": pl.Series(sgl, dtype=pl.Int64),
                "grid_version": pl.Series(gv, dtype=pl.Int32),
                "cluster_id": pl.Series(cid, dtype=pl.Int64),
            }
        )

    def _roundtrip(self, df: pl.DataFrame) -> pl.DataFrame:
        encoded = df.with_columns(ck_encode())
        return encoded.with_columns(
            ck_decode_super_grid_label(),
            ck_decode_grid_version(),
            ck_decode_cluster_id(),
        )

    def test_zero_values(self):
        df = self._make_df([0], [0], [0])
        rt = self._roundtrip(df)
        assert rt["super_grid_label"][0] == 0
        assert rt["grid_version"][0] == 0
        assert rt["cluster_id"][0] == 0

    def test_typical_values(self):
        sgl = [0, 14_500, 1, 2_000_000]
        gv = [0, 0, 1, 1]
        cid = [0, 99, 42, 2_097_151]
        df = self._make_df(sgl, gv, cid)
        rt = self._roundtrip(df)
        assert rt["super_grid_label"].to_list() == sgl
        assert rt["grid_version"].to_list() == gv
        assert rt["cluster_id"].to_list() == cid

    def test_max_values(self):
        """All fields at their maximum supported value must survive a round-trip."""
        max31 = (1 << 31) - 1
        df = self._make_df([max31], [1], [max31])
        rt = self._roundtrip(df)
        assert rt["super_grid_label"][0] == max31
        assert rt["grid_version"][0] == 1
        assert rt["cluster_id"][0] == max31

    def test_packed_value_fits_in_signed_int64(self):
        """Packed key must be non-negative (sign bit clear)."""
        max31 = (1 << 31) - 1
        df = self._make_df([max31], [1], [max31])
        packed = df.with_columns(ck_encode())["_ck"][0]
        assert 0 <= packed <= (1 << 63) - 1

    def test_different_grid_versions_produce_different_keys(self):
        """Same (sgl, cid) but different grid_version must produce distinct keys."""
        df = self._make_df([5, 5], [0, 1], [10, 10])
        keys = df.with_columns(ck_encode())["_ck"].to_list()
        assert keys[0] != keys[1]

    def test_different_sgl_same_cid_produce_different_keys(self):
        df = self._make_df([0, 1], [0, 0], [42, 42])
        keys = df.with_columns(ck_encode())["_ck"].to_list()
        assert keys[0] != keys[1]

    def test_different_cid_same_sgl_produce_different_keys(self):
        df = self._make_df([7, 7], [1, 1], [0, 1])
        keys = df.with_columns(ck_encode())["_ck"].to_list()
        assert keys[0] != keys[1]

    def test_decode_accepts_arbitrary_expr(self):
        """Decode helpers should work when passed a non-default expression."""
        df = self._make_df([3], [0], [77])
        packed = df.with_columns(ck_encode()).rename({"_ck": "my_key"})
        sgl = packed.select(ck_decode_super_grid_label(pl.col("my_key")))["super_grid_label"][0]
        gv = packed.select(ck_decode_grid_version(pl.col("my_key")))["grid_version"][0]
        cid = packed.select(ck_decode_cluster_id(pl.col("my_key")))["cluster_id"][0]
        assert (sgl, gv, cid) == (3, 0, 77)


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
            "ef_cluster_id": ef_cluster_id.astype(np.int64),
        }
    )
    df.write_parquet(path, row_group_size=250)


def test_map_clusters_to_cells_sharded_matches_single(tmp_path):
    """Sharded (worker_id/worker_n) mapping should cover all rows when merged,
    matching single-worker output."""
    input_path = str(tmp_path / "cluster_input.parquet")
    out_single = str(tmp_path / "mapped_single.parquet")
    out_shard0 = str(tmp_path / "mapped_shard0.parquet")
    out_shard1 = str(tmp_path / "mapped_shard1.parquet")
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
    common_kwargs = dict(
        input_path=input_path,
        nuc_centroid=nuc_centroid,
        nuc_ellipse_mat=nuc_ellipse_mat,
        nuc_cell_id=nuc_cell_id,
    )

    map_clusters_to_cells(**common_kwargs).sink_parquet(out_single)
    map_clusters_to_cells(worker_id=0, worker_n=2, **common_kwargs).sink_parquet(out_shard0)
    map_clusters_to_cells(worker_id=1, worker_n=2, **common_kwargs).sink_parquet(out_shard1)

    sort_cols = ["transcript_id", "ef_cluster_id"]
    df_single = pl.read_parquet(out_single).sort(sort_cols)
    df_sharded = pl.concat([pl.read_parquet(out_shard0), pl.read_parquet(out_shard1)]).sort(
        sort_cols
    )

    # Shards should not overlap
    assert df_sharded["transcript_id"].n_unique() == df_sharded.height

    for col in df_single.columns:
        if df_single[col].dtype.is_float():
            np.testing.assert_allclose(
                df_single[col].to_numpy(),
                df_sharded[col].to_numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            assert df_single[col].equals(df_sharded[col])


class TestComputeClusterToCellMapping:
    """Unit tests for the cluster→cell distance metric.

    ``nuc_ellipse_mat[i]`` (``M``) maps the unit circle to the ellipse for
    nucleus ``i``, so the Mahalanobis quadratic form uses
    ``Q = inv(M @ M.T)``. For a diagonal ``M = diag(a, b)`` this reduces to
    ``sqrt((dx/a)^2 + (dy/b)^2)``, which is what we exercise below.
    """

    def test_nearest_centroid_assignment(self):
        """Each cluster is paired with the closest nucleus centroid."""
        nuc_centroid = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        nuc_ellipse_mat = np.tile(np.eye(2), (3, 1, 1))
        nuc_cell_id = np.array([100, 200, 300])

        cluster_id = np.array([1, 2, 3])
        cluster_xy = np.array([[0.5, 0.1], [9.0, 0.2], [0.3, 9.5]])

        out = compute_cluster_to_cell_mapping(
            cluster_id=cluster_id,
            cluster_xy=cluster_xy,
            nuc_centroid=nuc_centroid,
            nuc_ellipse_mat=nuc_ellipse_mat,
            nuc_cell_id=nuc_cell_id,
        )

        assert out["ef_cluster_id"].to_list() == [1, 2, 3]
        assert out["cell_id"].to_list() == [100, 200, 300]

    def test_centroid_distance_is_euclidean(self):
        nuc_centroid = np.array([[0.0, 0.0], [100.0, 100.0]])
        nuc_ellipse_mat = np.tile(np.eye(2), (2, 1, 1))
        nuc_cell_id = np.array([1, 2])

        cluster_xy = np.array([[3.0, 4.0], [103.0, 104.0]])
        out = compute_cluster_to_cell_mapping(
            cluster_id=np.array([10, 20]),
            cluster_xy=cluster_xy,
            nuc_centroid=nuc_centroid,
            nuc_ellipse_mat=nuc_ellipse_mat,
            nuc_cell_id=nuc_cell_id,
        )

        np.testing.assert_allclose(out["centroid_distance"].to_numpy(), [5.0, 5.0])

    def test_mahalanobis_equals_euclidean_for_identity_ellipse(self):
        """With M = I, Mahalanobis collapses to Euclidean distance."""
        nuc_centroid = np.array([[0.0, 0.0]])
        nuc_ellipse_mat = np.array([np.eye(2)])
        nuc_cell_id = np.array([7])

        cluster_xy = np.array([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]])
        out = compute_cluster_to_cell_mapping(
            cluster_id=np.array([1, 2, 3]),
            cluster_xy=cluster_xy,
            nuc_centroid=nuc_centroid,
            nuc_ellipse_mat=nuc_ellipse_mat,
            nuc_cell_id=nuc_cell_id,
        )

        np.testing.assert_allclose(
            out["mahalanobis_distance"].to_numpy(),
            out["centroid_distance"].to_numpy(),
        )
        np.testing.assert_allclose(out["mahalanobis_distance"].to_numpy(), [5.0, 2.0, 1.0])

    def test_mahalanobis_diagonal_ellipse_scales_axes(self):
        """For M = diag(a, b), Mahalanobis distance = sqrt((dx/a)^2 + (dy/b)^2)."""
        a, b = 2.0, 5.0
        nuc_centroid = np.array([[0.0, 0.0]])
        nuc_ellipse_mat = np.array([[[a, 0.0], [0.0, b]]])
        nuc_cell_id = np.array([1])

        cluster_xy = np.array([[a, 0.0], [0.0, b], [a, b], [2 * a, 0.0]])
        out = compute_cluster_to_cell_mapping(
            cluster_id=np.arange(4),
            cluster_xy=cluster_xy,
            nuc_centroid=nuc_centroid,
            nuc_ellipse_mat=nuc_ellipse_mat,
            nuc_cell_id=nuc_cell_id,
        )

        # Points exactly on the ellipse boundary have Mahalanobis = 1
        # (a,0), (0,b) → 1; (a,b) → sqrt(2); (2a,0) → 2
        np.testing.assert_allclose(
            out["mahalanobis_distance"].to_numpy(),
            [1.0, 1.0, np.sqrt(2.0), 2.0],
        )

    def test_mahalanobis_rotated_ellipse(self):
        """A 45°-rotated ellipse: points along the major axis stay on boundary at d=1."""
        a, b = 4.0, 1.0
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S = np.array([[a, 0.0], [0.0, b]])
        M = R @ S  # maps unit circle to rotated ellipse

        nuc_centroid = np.array([[0.0, 0.0]])
        nuc_ellipse_mat = np.array([M])
        nuc_cell_id = np.array([42])

        # Tip of major axis is R @ (a, 0) = (a/√2, a/√2)
        major_tip = R @ np.array([a, 0.0])
        # Tip of minor axis is R @ (0, b)
        minor_tip = R @ np.array([0.0, b])
        # Center
        center = np.array([0.0, 0.0])

        cluster_xy = np.stack([center, major_tip, minor_tip, 2 * major_tip])
        out = compute_cluster_to_cell_mapping(
            cluster_id=np.arange(4),
            cluster_xy=cluster_xy,
            nuc_centroid=nuc_centroid,
            nuc_ellipse_mat=nuc_ellipse_mat,
            nuc_cell_id=nuc_cell_id,
        )

        np.testing.assert_allclose(
            out["mahalanobis_distance"].to_numpy(),
            [0.0, 1.0, 1.0, 2.0],
            atol=1e-12,
        )

    def test_mahalanobis_uses_assigned_centroids_ellipse(self):
        """Each cluster's Mahalanobis distance uses the ellipse of its nearest centroid."""
        nuc_centroid = np.array([[0.0, 0.0], [100.0, 0.0]])
        # Nucleus 0: narrow in x. Nucleus 1: narrow in y.
        nuc_ellipse_mat = np.array(
            [
                [[1.0, 0.0], [0.0, 10.0]],
                [[10.0, 0.0], [0.0, 1.0]],
            ]
        )
        nuc_cell_id = np.array([0, 1])

        # Cluster A near nuc 0 offset by (1, 0): d = 1.0 (using nuc 0 ellipse)
        # Cluster B near nuc 1 offset by (1, 0): d = 0.1 (using nuc 1 ellipse)
        cluster_xy = np.array([[1.0, 0.0], [101.0, 0.0]])
        out = compute_cluster_to_cell_mapping(
            cluster_id=np.array([0, 1]),
            cluster_xy=cluster_xy,
            nuc_centroid=nuc_centroid,
            nuc_ellipse_mat=nuc_ellipse_mat,
            nuc_cell_id=nuc_cell_id,
        )

        np.testing.assert_allclose(out["mahalanobis_distance"].to_numpy(), [1.0, 0.1])
        assert out["cell_id"].to_list() == [0, 1]


def _make_mapped_tx() -> pl.DataFrame:
    """Step 5 output: three clusters near nucleus 10, one far from both nuclei."""
    return pl.DataFrame(
        {
            "transcript_id": np.arange(7, dtype=np.uint64),
            "feature_name": ["GeneA", "GeneA", "GeneB", "GeneB", "GeneA", "GeneB", "GeneB"],
            "cell_id": [10, 10, 10, 10, 20, 20, 20],
            "x": [0.0, 0.2, 1.0, 1.2, 5.0, 5.2, 5.4],
            "y": [0.0, 0.1, 1.0, 1.1, 5.0, 5.2, 5.4],
            "centroid_distance": [0.1, 0.1, 99.0, 99.0, 2.0, 2.0, 2.0],
            "mahalanobis_distance": [0.5, 0.5, 0.8, 0.8, 1.2, 1.2, 1.2],
            "ef_cluster_id": [1, 1, 2, 2, 3, 3, 3],
        }
    )


def _make_dataset() -> STDataset:
    return STDataset(
        boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
        nuc_cell_id=np.array([10, 20, 30]),
        nuc_total_tx_counts=np.array([10, 10, 10]),
        nuc_cell_counts=np.array([10, 10, 10]),
        nuc_xy=np.array([[0.0, 0.0], [5.0, 5.0], [50.0, 50.0]]),
        nuc_ellipse_mat=np.tile(np.eye(2), (3, 1, 1)),
        nuc_countmat_nuc_idx=np.array([], dtype=int),
        nuc_countmat_gene_idx=np.array([], dtype=int),
        nuc_countmat_counts=np.array([], dtype=int),
        genes=np.array(["GeneA", "GeneB"]),
    )


def _make_original_tx(path):
    pl.DataFrame(
        {
            "transcript_id": np.arange(9, dtype=np.uint64),
            "x": np.arange(9, dtype=np.float64),
            "y": np.zeros(9),
            "overlaps_nucleus": np.zeros(9, dtype=np.uint8),
        }
    ).write_parquet(path)


def test_classify_clusters_categories(tmp_path):
    """Anchored needs mahalanobis <= 1 AND centroid distance <= cd_cutoff; the rest
    are denovo / fragment by n_tx against max(percentile, min_tx_count)."""
    tx_path = tmp_path / "ef.parquet"
    _make_original_tx(tmp_path / "orig.parquet")
    calculate_tx_distances(_make_mapped_tx().lazy(), str(tmp_path / "orig.parquet")).sink_parquet(
        tx_path
    )

    info, thr = classify_clusters(
        str(tx_path),
        nuc_cell_counts=np.array([10, 10, 10]),
        denovo_min_tx_pctile=0.0,
        min_tx_count=3,
    )
    cats = dict(zip(info["ef_cluster_id"].to_list(), info["cell_category"].to_list()))
    ids = dict(zip(info["ef_cluster_id"].to_list(), info["unified_cell_id"].to_list()))
    assert thr == 3.0
    assert cats == {1: "nucleus_anchored", 2: "fragment", 3: "denovo"}
    assert ids == {1: "10", 2: "fragment_2", 3: "denovo_3"}

    ## Percentile wins over the floor when it is larger
    _, thr = classify_clusters(
        str(tx_path),
        nuc_cell_counts=np.array([10, 10, 10]),
        denovo_min_tx_pctile=50.0,
        min_tx_count=3,
    )
    assert thr == 10.0


def test_post_process_merge_outputs(tmp_path):
    """ef.parquet carries ef_cell_id (UNASSIGNED for noise) and x_initial from the original
    transcripts; ef.h5ad keeps every cell, including empty nuclei."""
    mapped = tmp_path / "mapped.parquet"
    orig = tmp_path / "orig.parquet"
    noise_meta = tmp_path / "noise_meta.parquet"
    _make_mapped_tx().write_parquet(mapped)
    _make_original_tx(orig)
    pl.DataFrame(
        {
            "transcript_id": np.arange(9, dtype=np.int64),
            "feature_name": ["GeneA"] * 9,
            "x": np.full(9, 7.0),
            "y": np.full(9, 7.0),
            "is_noise": [False] * 7 + [True, True],
            "tx_noise_prob_0": np.full(9, 0.1, dtype=np.float32),
            "tx_noise_prob_1": np.array([0.1] * 7 + [0.9, 0.9], dtype=np.float32),
        }
    ).write_parquet(noise_meta)

    post_process_merge(
        mapped_files=[str(mapped)],
        dataset=_make_dataset(),
        original_tx_fname=str(orig),
        output_fname=str(tmp_path / "ef.parquet"),
        anndata_fname=str(tmp_path / "ef.h5ad"),
        noise_meta_fname=str(noise_meta),
        work_dir=str(tmp_path),
        denovo_min_tx_pctile=0.0,
        min_tx_count=3,
    )

    ef = pl.read_parquet(tmp_path / "ef.parquet").sort("transcript_id")
    assert ef.height == 9
    assert (
        ef["ef_cell_id"].to_list()
        == ["10", "10"] + ["fragment_2"] * 2 + ["denovo_3"] * 3 + ["UNASSIGNED"] * 2
    )
    np.testing.assert_array_equal(ef["x_initial"].to_numpy(), np.arange(9, dtype=float))

    adata = anndata.read_h5ad(tmp_path / "ef.h5ad")
    obs = adata.obs
    assert set(obs.index) == {"10", "20", "30", "fragment_2", "denovo_3"}
    assert obs.loc["10", "cell_category"] == "nucleus_anchored"
    assert obs.loc["30", "cell_category"] == "nucleus_anchored"  # empty nucleus kept
    assert obs.loc["30", "n_tx"] == 0
    assert obs.loc["fragment_2", "cell_category"] == "fragment"
    assert obs.loc["denovo_3", "cell_category"] == "denovo"
    ## Anchored cells sit at their nucleus centroid; others at their mean flowed position
    assert (obs.loc["10", "x_centroid"], obs.loc["10", "y_centroid"]) == (0.0, 0.0)
    np.testing.assert_allclose(obs.loc["denovo_3", "x_centroid"], 5.2)
    assert adata.uns["denovo_tx_threshold"] == 3.0
