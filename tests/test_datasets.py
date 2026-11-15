"""Tests for the datasets module, particularly patch sampling behavior."""

import pickle

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch
from scipy.sparse import csr_array

from ebbf.data import datasets
from ebbf.data.datasets import NuclearPatchDataset, TxPatchDataset
from ebbf.data.utils import RTree


# Module-level helper functions for multiprocessing tests
def _rtree_lookup_in_subprocess(xy_data):
    """Helper for test_rtree_survives_fork."""
    rtree = RTree(xy_data)
    return len(rtree.lookup(xmin=50, xmax=150, ymin=50, ymax=150))


def _sample_patch_in_subprocess(dataset):
    """Helper for test_sample_patch_works_after_fork."""
    patch = dataset._sample_patch()
    return patch["nuc2xy"].shape[0]


class TestRTree:
    """Tests for the RTree spatial index."""

    def test_rtree_survives_fork(self):
        """Test that RTree works correctly in forked subprocess.

        This is a critical test - the rtree library uses C extensions
        that don't survive fork() by default. Creating a new RTree
        in the subprocess should work.
        """
        import multiprocessing as mp

        xy = np.array(
            [
                [100.0, 100.0],
                [200.0, 200.0],
            ]
        )

        # Test in subprocess - creating RTree from scratch works
        ctx = mp.get_context("fork")
        with ctx.Pool(1) as pool:
            result = pool.apply(_rtree_lookup_in_subprocess, (xy,))
            assert result == 1, "RTree should find 1 point in forked subprocess"

    def test_rtree_basic_lookup(self):
        """Test that RTree can find points within a bounding box."""
        xy = np.array(
            [
                [100.0, 100.0],
                [200.0, 200.0],
                [300.0, 300.0],
            ]
        )
        rtree = RTree(xy)

        # Should find point at (100, 100)
        indices = rtree.lookup(xmin=50, xmax=150, ymin=50, ymax=150)
        assert len(indices) == 1
        assert indices[0].item() == 0

        # Should find all points
        indices = rtree.lookup(xmin=0, xmax=400, ymin=0, ymax=400)
        assert len(indices) == 3

        # Should find no points
        indices = rtree.lookup(xmin=400, xmax=500, ymin=400, ymax=500)
        assert len(indices) == 0

    def test_rtree_center_radius_lookup(self):
        """Test lookup_center_radius method."""
        xy = np.array(
            [
                [100.0, 100.0],
                [200.0, 200.0],
            ]
        )
        rtree = RTree(xy)

        # Radius of 50 centered at (100, 100) should find first point
        indices = rtree.lookup_center_radius(x_center=100, y_center=100, radius=50)
        assert len(indices) == 1
        assert indices[0].item() == 0

        # Radius of 200 centered at (150, 150) should find both points
        indices = rtree.lookup_center_radius(x_center=150, y_center=150, radius=200)
        assert len(indices) == 2


class TestNuclearPatchDataset:
    """Tests for NuclearPatchDataset patch sampling behavior."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        # Create nuclei close together so patches can contain 2+
        nuc_xy = np.array(
            [
                [100.0, 100.0],
                [120.0, 100.0],
                [100.0, 120.0],
                [120.0, 120.0],
            ]
        )
        nuc2total_gene_count = np.array([10, 20, 30, 40])
        nuc2ctype = np.array([0, 0, 1, 1])
        nuc2cell_count = np.array([5, 10, 15, 20])

        # Simple sparse matrix for gene probs
        nuc2gene_probs = csr_array(np.eye(4, 10))

        return NuclearPatchDataset(
            nuc2xy=nuc_xy,
            nuc2total_gene_count=nuc2total_gene_count,
            nuc2gene_probs=nuc2gene_probs,
            nuc2ctype=nuc2ctype,
            nuc2cell_count=nuc2cell_count,
            patch_size=100,
            patch_padding=10,
            dataset_size=10,
            min_nucs=2,
        )

    def test_dataset_length(self, simple_dataset):
        """Test that dataset length matches dataset_size."""
        assert len(simple_dataset) == 10

    def test_sample_patch_finds_nuclei(self, simple_dataset):
        """Test that _sample_patch can find patches with nuclei."""
        # With nuclei at (50,50), (150,50), (50,150), (150,150)
        # and patch_size=100, we should be able to find patches
        patch = simple_dataset._sample_patch()

        assert "nuc2xy" in patch
        assert patch["nuc2xy"].shape[0] >= simple_dataset.min_nucs

    def test_data_bounds_computed_correctly(self, simple_dataset):
        """Test that data_min and data_max are computed from nuclei positions."""
        # nuc_xy = [[100, 100], [120, 100], [100, 120], [120, 120]]
        assert torch.allclose(simple_dataset.data_min, torch.tensor([100.0, 100.0]))
        assert torch.allclose(simple_dataset.data_max, torch.tensor([120.0, 120.0]))

    def test_patch_center_sampling_in_bounds(self, simple_dataset):
        """Test that sampled patch centers fall within data bounds."""
        for _ in range(100):
            patch_center = torch.rand(1, 2)
            patch_center = (
                simple_dataset.data_max - simple_dataset.data_min
            ) * patch_center + simple_dataset.data_min

            # patch_center should be within [100, 120] for both x and y
            assert 100 <= patch_center[0, 0].item() <= 120
            assert 100 <= patch_center[0, 1].item() <= 120

    def test_sample_patch_works_after_fork(self, simple_dataset):
        """Test that _sample_patch works in forked subprocess.

        This tests the critical fix for RTree not surviving fork().
        """
        import multiprocessing as mp

        # First sample in main process to initialize rtree
        patch = simple_dataset._sample_patch()
        main_nuclei = patch["nuc2xy"].shape[0]
        assert main_nuclei >= simple_dataset.min_nucs

        # Then sample in forked subprocess
        ctx = mp.get_context("fork")
        with ctx.Pool(1) as pool:
            sub_nuclei = pool.apply(_sample_patch_in_subprocess, (simple_dataset,))
            assert sub_nuclei >= simple_dataset.min_nucs, (
                "Forked subprocess should find nuclei (RTree must work after fork)"
            )


class TestSparseDataset:
    """Tests for datasets with clustered nuclei."""

    @pytest.fixture
    def sparse_dataset(self):
        """Create a dataset with clustered nuclei."""
        nuc_xy = np.array(
            [
                [400.0, 300.0],
                [420.0, 310.0],
                [450.0, 320.0],
                [600.0, 370.0],
                [620.0, 380.0],
            ]
        )
        nuc2total_gene_count = np.ones(5)
        nuc2ctype = np.zeros(5, dtype=np.int64)
        nuc2cell_count = np.ones(5)
        nuc2gene_probs = csr_array(np.eye(5, 10))

        return NuclearPatchDataset(
            nuc2xy=nuc_xy,
            nuc2total_gene_count=nuc2total_gene_count,
            nuc2gene_probs=nuc2gene_probs,
            nuc2ctype=nuc2ctype,
            nuc2cell_count=nuc2cell_count,
            patch_size=100,
            patch_padding=10,
            dataset_size=10,
            min_nucs=2,
        )

    def test_data_bounds_match_nuclei(self, sparse_dataset):
        """Test that data bounds are computed from actual nuclei positions."""
        # nuc_xy x range: 400-620, y range: 300-380
        assert torch.allclose(sparse_dataset.data_min, torch.tensor([400.0, 300.0]))
        assert torch.allclose(sparse_dataset.data_max, torch.tensor([620.0, 380.0]))

    def test_rtree_contains_all_nuclei(self, sparse_dataset):
        """Verify RTree contains all nuclei from the dataset."""
        # Use data_min/data_max which are computed from nuclei positions
        xmin, ymin = sparse_dataset.data_min.tolist()
        xmax, ymax = sparse_dataset.data_max.tolist()

        # Query for all nuclei using the bounding box
        all_indices = sparse_dataset.rtree.lookup(
            xmin=xmin - 1, xmax=xmax + 1, ymin=ymin - 1, ymax=ymax + 1
        )

        assert len(all_indices) == sparse_dataset.nuc2xy.shape[0], "RTree should contain all nuclei"

    def test_sample_patch_eventually_succeeds(self, sparse_dataset):
        """Test that _sample_patch eventually finds a valid patch."""
        # Even with sparse data, _sample_patch should succeed
        # (it loops until it finds a valid patch)
        patch = sparse_dataset._sample_patch()

        assert patch is not None
        assert patch["nuc2xy"].shape[0] >= sparse_dataset.min_nucs


class _CountingDataset:
    """Count the filters run against a pyarrow dataset"""

    def __init__(self, dataset: ds.Dataset):
        self.dataset = dataset
        self.n_filters = 0

    def filter(self, expression: ds.Expression) -> ds.Dataset:
        self.n_filters += 1
        return self.dataset.filter(expression)


class TestTxPatchDataset:
    """Tests for TxPatchDataset transcript loading."""

    GENES = np.array(["A", "B", "C"])
    BOXES = np.array([[0.0, 0.0, 200.0, 200.0], [250.0, 100.0, 450.0, 300.0]])

    @pytest.fixture
    def tx_dataset(self, tmp_path) -> ds.Dataset:
        rng = np.random.default_rng(0)
        n = 20_000
        table = pa.table(
            {
                "transcript_id": np.arange(n, dtype=np.uint64),
                "x_location": rng.uniform(0, 500, n).astype(np.float32),
                "y_location": rng.uniform(0, 500, n).astype(np.float32),
                "feature_name": rng.choice(self.GENES, n),
                "is_gene": rng.uniform(size=n) > 0.1,
            }
        )
        path = tmp_path / "transcripts.parquet"
        pq.write_table(table, path, row_group_size=1_000)
        return ds.dataset(path, format="parquet")

    def _make(self, tx_dataset) -> TxPatchDataset:
        ## Nuclei on a 20 um grid so that every patch has nuclei and is yielded
        grid = np.arange(0.0, 501.0, 20.0)
        nuc_xy = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
        return TxPatchDataset(
            boxes=self.BOXES,
            tx_dataset=tx_dataset,
            nuc_xy=nuc_xy,
            genes=self.GENES,
            patch_size=100.0,
            patch_padding=100.0,
            overlap=20.0,
            max_tx=1_000_000,
            x_col="x_location",
            y_col="y_location",
        )

    @pytest.mark.parametrize("boxes_per_scan", [1, 32])
    def test_patches_get_the_rows_of_a_full_file_filter(
        self, tx_dataset, monkeypatch, boxes_per_scan
    ):
        """Each patch gets the rows, in file order, that filtering the whole file selects"""
        monkeypatch.setattr(datasets, "BOXES_PER_SCAN", boxes_per_scan)
        dataset = self._make(tx_dataset)
        elements = {tuple(e["patch_indices"][0].tolist()): e for e in dataset}
        assert len(elements) == len(dataset)

        for box_idx, patch_i, patch_j in dataset.patch_coords:
            ## Patch bounds as the dataset defines them; the last patches of a
            ## box extend past the box edge
            xmin, xmax, ymin, ymax = dataset._get_patch_bounds(box_idx, patch_i, patch_j)
            expected = tx_dataset.filter(
                ds.field("is_gene")
                & (ds.field("x_location") >= xmin)
                & (ds.field("y_location") >= ymin)
                & (ds.field("x_location") <= xmax)
                & (ds.field("y_location") <= ymax)
            ).to_table()
            element = elements[(box_idx, patch_i, patch_j)]
            np.testing.assert_array_equal(
                element["tx_idx"].numpy(), expected["transcript_id"].to_numpy().astype(np.int64)
            )
            gene_idx = {g: i for i, g in enumerate(self.GENES)}
            np.testing.assert_array_equal(
                element["tx_gene_idx"].numpy(),
                [gene_idx[g] for g in expected["feature_name"].to_pylist()],
            )

    @pytest.mark.parametrize(("boxes_per_scan", "n_scans"), [(1, 2), (32, 1)])
    def test_scans_the_transcript_file_once_per_group_of_boxes(
        self, tx_dataset, monkeypatch, boxes_per_scan, n_scans
    ):
        monkeypatch.setattr(datasets, "BOXES_PER_SCAN", boxes_per_scan)
        counting = _CountingDataset(tx_dataset)
        dataset = self._make(counting)
        n_elements = sum(1 for _ in dataset)
        assert n_elements == len(dataset) > len(self.BOXES)
        assert counting.n_filters == n_scans

    def test_pickles_after_iteration(self, tx_dataset):
        """A DataLoader that spawns workers pickles the dataset, possibly after a pass"""
        dataset = self._make(tx_dataset)
        first = [e["tx_idx"] for e in dataset]
        again = [e["tx_idx"] for e in pickle.loads(pickle.dumps(dataset))]
        assert len(first) == len(again)
        assert all(torch.equal(a, b) for a, b in zip(first, again))
