import numpy as np
import pytest
from abm.utils import x_y_to_i_j, discount_by_distance, find_peak


def test_x_y_to_i_j():
    assert x_y_to_i_j(1, 2) == (2, 1)
    assert x_y_to_i_j(0, 0) == (0, 0)
    assert x_y_to_i_j(5, 10) == (10, 5)
    assert x_y_to_i_j(-1, -2) == (-2, -1)


def test_discount_by_distance():
    array = np.array([[1, 2], [3, 4]])
    agent_location = (0, 0)
    discounted_array = discount_by_distance(array, agent_location, discount_factor=0.5)

    # Manually compute the expected discounted values
    expected_array = np.array([
        [1 * 0.5 ** 0, 2 * 0.5 ** 1.0],
        [3 * 0.5 ** 1.0, 4 * 0.5 ** 1.4142]
    ])

    np.testing.assert_allclose(discounted_array, expected_array, rtol=1e-5)

    # Test with different discount factors
    discounted_array = discount_by_distance(array, agent_location, discount_factor=0.8)
    expected_array = np.array([
        [1 * 0.8 ** 0, 2 * 0.8 ** 1.0],
        [3 * 0.8 ** 1.0, 4 * 0.8 ** 1.4142]
    ])
    np.testing.assert_allclose(discounted_array, expected_array, rtol=1e-5)

    # Test with a different agent location
    agent_location = (1, 1)
    discounted_array = discount_by_distance(array, agent_location, discount_factor=0.5)
    expected_array = np.array([
        [1 * 0.5 ** 1.4142, 2 * 0.5 ** 1.0],
        [3 * 0.5 ** 1.0, 4 * 0.5 ** 0]
    ])
    np.testing.assert_allclose(discounted_array, expected_array, rtol=1e-5)


def test_find_peak():
    array = np.array([[1, 2], [3, 4]])
    assert find_peak(array) == (1, 1)  # 4 is the peak value

    array = np.array([[1, 1], [1, 1]])
    assert find_peak(array) == (0, 0)  # Multiple peaks, return the first

    array = np.array([[10, 20, 30], [5, 15, 25]])
    assert find_peak(array) == (0, 2)  # 30 is the peak value

    array = np.array([[0]])
    assert find_peak(array) == (0, 0)  # Single element

    # Test with a 3D array
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert find_peak(array) == (1, 1, 1)  # 8 is the peak value

    # Test with an empty array
    array = np.array([])
    with pytest.raises(ValueError):
        find_peak(array)


if __name__ == "__main__":
    pytest.main()
