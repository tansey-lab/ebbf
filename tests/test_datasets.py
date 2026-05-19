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


class TestTissueBoundarySampling:
    """Sampling must be restricted to the supplied tissue polygon."""

    def _build_dataset(self, tissue_polygon):
        rng = np.random.default_rng(0)
        # Dense grid of nuclei spanning [0, 100] x [0, 100] so min_nucs is never
        # the limiting factor — the tissue polygon must be what constrains sampling.
        xs, ys = np.meshgrid(np.linspace(1, 99, 50), np.linspace(1, 99, 50))
        nuc_xy = np.stack([xs.ravel(), ys.ravel()], axis=1) + rng.normal(scale=0.1, size=(2500, 2))
        n = nuc_xy.shape[0]
        return NuclearPatchDataset(
            nuc2xy=nuc_xy,
            nuc2total_gene_count=np.ones(n, dtype=np.int64),
            nuc2gene_probs=csr_array(np.ones((n, 4)) / 4),
            nuc2ctype=np.zeros(n, dtype=np.int64),
            nuc2cell_count=np.ones(n, dtype=np.int64),
            patch_size=4,
            patch_padding=1,
            dataset_size=200,
            min_nucs=1,
            tissue_polygon=tissue_polygon,
        )

    def test_patch_centers_inside_polygon(self):
        """All sampled patch centers should lie inside the tissue polygon."""
        from shapely.geometry import Polygon

        # Tissue restricted to a 50x50 square in the middle of the data extent.
        tissue = Polygon([(25, 25), (75, 25), (75, 75), (25, 75)])
        dataset = self._build_dataset(tissue)

        for _ in range(200):
            patch = dataset._sample_patch()
            cx, cy = patch["patch_center"][0].tolist()
            assert 25 <= cx <= 75, f"x={cx} outside tissue"
            assert 25 <= cy <= 75, f"y={cy} outside tissue"

    def test_bounds_tightened_to_polygon_bbox(self):
        """data_min/data_max should be tightened to the polygon bounding box."""
        from shapely.geometry import Polygon

        tissue = Polygon([(25, 25), (75, 25), (75, 75), (25, 75)])
        dataset = self._build_dataset(tissue)

        assert dataset.data_min[0].item() == pytest.approx(25.0)
        assert dataset.data_min[1].item() == pytest.approx(25.0)
        assert dataset.data_max[0].item() == pytest.approx(75.0)
        assert dataset.data_max[1].item() == pytest.approx(75.0)

    def test_non_convex_polygon_excludes_holes(self):
        """L-shaped tissue: patches must never land in the excluded quadrant."""
        from shapely.geometry import Polygon

        # L-shape covering [0,100]x[0,100] minus the top-right quadrant [50,100]x[50,100]
        tissue = Polygon([(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)])
        dataset = self._build_dataset(tissue)

        for _ in range(300):
            patch = dataset._sample_patch()
            cx, cy = patch["patch_center"][0].tolist()
            in_excluded = cx > 50 and cy > 50
            assert not in_excluded, f"patch center ({cx}, {cy}) landed in excluded quadrant"


class TestBoxFilteringByTissue:
    """_filter_boxes_by_tissue should keep only boxes intersecting the polygon."""

    def test_filter_boxes_by_tissue(self):
        from shapely.geometry import Polygon

        from ebbf.data.st_dataset import _filter_boxes_by_tissue

        boxes = np.array(
            [
                [0, 0, 10, 10],  # fully inside
                [40, 40, 60, 60],  # straddles boundary -> intersects
                [80, 80, 90, 90],  # fully outside
                [49, 49, 51, 51],  # nearly all inside, touches boundary
            ],
            dtype=float,
        )
        tissue = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])

        kept = _filter_boxes_by_tissue(boxes, tissue)
        assert kept.shape[0] == 3
        # Excluded box (the [80,80,90,90] one) is gone
        assert not np.any(np.all(kept == np.array([80, 80, 90, 90]), axis=1))

    def test_filter_boxes_empty_input(self):
        from shapely.geometry import Polygon

        from ebbf.data.st_dataset import _filter_boxes_by_tissue

        tissue = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        kept = _filter_boxes_by_tissue(np.empty((0, 4)), tissue)
        assert kept.shape == (0, 4)

    def test_no_polygon_does_not_serialize_none(self, tmp_path):
        """An STDataset built without a tissue polygon must not write a None
        scalar that would later be misinterpreted as WKB bytes."""
        from ebbf.data.st_dataset import STDataset

        stdata = STDataset(
            boxes=np.array([[0, 0, 1, 1]], dtype=float),
            nuc_cell_id=np.array(["a"]),
            nuc_total_tx_counts=np.array([1]),
            nuc_ctype_idx=np.array([0]),
            nuc_cell_counts=np.array([1]),
            nuc_xy=np.array([[0.0, 0.0]]),
            nuc_ellipse_mat=np.eye(2)[None],
            nuc_countmat_nuc_idx=np.array([0]),
            nuc_countmat_gene_idx=np.array([0]),
            nuc_countmat_counts=np.array([1]),
            genes=np.array(["g"]),
            tissue_polygon_wkb=None,
        )
        path = tmp_path / "stdata.npz"
        stdata.save(str(path))

        with np.load(str(path), allow_pickle=False) as loaded:
            assert "tissue_polygon_wkb" not in loaded.files

        # And reloading shouldn't crash when the sampler asks for the polygon
        restored = STDataset.load(str(path))
        assert getattr(restored, "tissue_polygon_wkb", None) is None

    def test_tissue_polygon_roundtrip_through_npz(self, tmp_path):
        """Tissue polygon must round-trip through stdata.npz without pickle."""
        import shapely
        from shapely.geometry import Polygon

        from ebbf.data.st_dataset import STDataset

        tissue = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        wkb = np.frombuffer(shapely.to_wkb(tissue), dtype=np.uint8)

        stdata = STDataset(
            boxes=np.array([[0, 0, 1, 1]], dtype=float),
            nuc_cell_id=np.array(["a"]),
            nuc_total_tx_counts=np.array([1]),
            nuc_ctype_idx=np.array([0]),
            nuc_cell_counts=np.array([1]),
            nuc_xy=np.array([[0.0, 0.0]]),
            nuc_ellipse_mat=np.eye(2)[None],
            nuc_countmat_nuc_idx=np.array([0]),
            nuc_countmat_gene_idx=np.array([0]),
            nuc_countmat_counts=np.array([1]),
            genes=np.array(["g"]),
            tissue_polygon_wkb=wkb,
        )
        path = tmp_path / "stdata.npz"
        stdata.save(str(path))

        # Load without pickle support to prove no object arrays are written
        with np.load(str(path), allow_pickle=False) as loaded:
            assert "tissue_polygon_wkb" in loaded.files
            restored = shapely.from_wkb(loaded["tissue_polygon_wkb"].tobytes())
        assert restored.equals(tissue)
