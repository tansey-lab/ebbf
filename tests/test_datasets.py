"""Tests for the datasets module, particularly patch sampling behavior."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_array

from ebbf.data.datasets import NuclearPatchDataset
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

    def test_rtree_get_bbox(self):
        """Test that get_bbox returns correct bounds."""
        xy = np.array(
            [
                [100.0, 200.0],
                [300.0, 400.0],
            ]
        )
        rtree = RTree(xy)
        bbox = rtree.get_bbox()
        # bbox is (xmin, xmax, ymin, ymax)
        assert bbox[0] == 100.0  # xmin
        assert bbox[1] == 300.0  # xmax
        assert bbox[2] == 200.0  # ymin
        assert bbox[3] == 400.0  # ymax


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

        # Single punch covering the nuclei
        punch_bounds = np.array([[0.0, 0.0, 200.0, 200.0]])

        return NuclearPatchDataset(
            nuc2xy=nuc_xy,
            nuc2total_gene_count=nuc2total_gene_count,
            nuc2gene_probs=nuc2gene_probs,
            nuc2ctype=nuc2ctype,
            nuc2cell_count=nuc2cell_count,
            punch_bounds=punch_bounds,
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

    def test_punch_scale_and_translate(self, simple_dataset):
        """
        Test that punch_scale_factor and
        punch_translate_factor are computed correctly.
        """
        # punch_bounds = [[0, 0, 200, 200]]
        # scale_factor should be [200, 200] - [0, 0] = [200, 200]
        # translate_factor should be [0, 0]
        assert torch.allclose(
            simple_dataset.punch_scale_factor, torch.tensor([[200.0, 200.0]])
        )
        assert torch.allclose(
            simple_dataset.punch_translate_factor, torch.tensor([[0.0, 0.0]])
        )

    def test_patch_center_sampling_in_bounds(self, simple_dataset):
        """Test that sampled patch centers fall within punch bounds."""
        for _ in range(100):
            punch_idx = torch.randint(simple_dataset.n_punches, size=()).item()
            patch_center = torch.rand(1, 2)
            patch_center = (
                simple_dataset.punch_scale_factor[punch_idx] * patch_center
                + simple_dataset.punch_translate_factor[punch_idx]
            )

            # patch_center should be within [0, 200] for both x and y
            assert 0 <= patch_center[0, 0].item() <= 200
            assert 0 <= patch_center[0, 1].item() <= 200

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
    """Tests for sparse datasets like the integration test data."""

    @pytest.fixture
    def sparse_dataset(self):
        """Create a sparse dataset similar to integration test data.

        This mimics the issue where nuclei are clustered in a small region
        while punch bounds are much larger.
        """
        # Nuclei clustered in a small Y range (290-390)
        # while punch Y range is 290-690 (400 units)
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

        # Large punch bounds - nuclei only in bottom portion
        punch_bounds = np.array([[350.0, 290.0, 750.0, 690.0]])

        return NuclearPatchDataset(
            nuc2xy=nuc_xy,
            nuc2total_gene_count=nuc2total_gene_count,
            nuc2gene_probs=nuc2gene_probs,
            nuc2ctype=nuc2ctype,
            nuc2cell_count=nuc2cell_count,
            punch_bounds=punch_bounds,
            patch_size=100,
            patch_padding=10,
            dataset_size=10,
            min_nucs=2,
        )

    def test_sparse_dataset_nuclei_coverage(self, sparse_dataset):
        """Test what fraction of random patches contain nuclei."""
        n_trials = 1000
        n_with_nuclei = 0
        n_with_min_nuclei = 0

        for _ in range(n_trials):
            punch_idx = torch.randint(sparse_dataset.n_punches, size=()).item()
            patch_center = torch.rand(1, 2)
            patch_center = (
                sparse_dataset.punch_scale_factor[punch_idx] * patch_center
                + sparse_dataset.punch_translate_factor[punch_idx]
            )

            # Check how many nuclei are in range
            idx = sparse_dataset.rtree.lookup_center_radius(
                x_center=patch_center[0, 0].item(),
                y_center=patch_center[0, 1].item(),
                radius=(0.5 * sparse_dataset.patch_size + sparse_dataset.patch_padding),
            )

            if len(idx) > 0:
                n_with_nuclei += 1
            if len(idx) >= sparse_dataset.min_nucs:
                n_with_min_nuclei += 1

        # With sparse data, we expect some patches to have nuclei
        # but the success rate should be relatively low
        success_rate = n_with_min_nuclei / n_trials
        print(
            f"Success rate finding patches with "
            f">={sparse_dataset.min_nucs} nuclei: {success_rate:.1%}"
        )

        # The success rate should be > 0 (we can find some valid patches)
        assert n_with_min_nuclei > 0, (
            "Should be able to find at least some valid patches"
        )

        # But not too high (data is sparse)
        assert success_rate < 0.5, "Success rate should be < 50% for sparse data"

    def test_rtree_contains_all_nuclei(self, sparse_dataset):
        """Verify RTree contains all nuclei from the dataset."""
        bbox = sparse_dataset.rtree.get_bbox()
        xmin, xmax, ymin, ymax = bbox

        # Query for all nuclei using the bounding box
        all_indices = sparse_dataset.rtree.lookup(
            xmin=xmin - 1, xmax=xmax + 1, ymin=ymin - 1, ymax=ymax + 1
        )

        assert len(all_indices) == sparse_dataset.nuc2xy.shape[0], (
            "RTree should contain all nuclei"
        )

    def test_sample_patch_eventually_succeeds(self, sparse_dataset):
        """Test that _sample_patch eventually finds a valid patch."""
        # Even with sparse data, _sample_patch should succeed
        # (it loops until it finds a valid patch)
        patch = sparse_dataset._sample_patch()

        assert patch is not None
        assert patch["nuc2xy"].shape[0] >= sparse_dataset.min_nucs
