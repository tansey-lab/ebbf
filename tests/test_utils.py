import numpy as np

from ebbf.utils import vectorized_means_numpy


class TestVectorizedMeansNumpy:
    def test_simple_means(self):
        """Test basic functionality with simple data."""
        X = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        result = vectorized_means_numpy(X, labels)

        expected = np.array(
            [
                [2.0, 3.0],  # mean of rows 0 and 1
                [6.0, 7.0],  # mean of rows 2 and 3
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_label(self):
        """Test when all points have the same label."""
        X = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        labels = np.array([0, 0, 0])

        result = vectorized_means_numpy(X, labels)

        expected = np.array([[3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_non_contiguous_labels(self):
        """Test with non-contiguous label values (gap in labels)."""
        X = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        labels = np.array([0, 2])

        result = vectorized_means_numpy(X, labels)

        # Should have 3 rows (0, 1, 2) with row 1 being zeros
        assert result.shape == (3, 2)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result[1], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[2], [3.0, 4.0])

    def test_single_element_per_label(self):
        """Test when each label has exactly one element."""
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        labels = np.array([0, 1, 2])

        result = vectorized_means_numpy(X, labels)

        np.testing.assert_array_almost_equal(result, X)
