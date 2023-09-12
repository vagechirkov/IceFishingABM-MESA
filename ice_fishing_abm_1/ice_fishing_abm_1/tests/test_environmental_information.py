import numpy as np

from ice_fishing_abm_1.ice_fishing_abm_1.environmental_information import smooth_observations_with_gaussian_filter


def test_smooth_observations_with_gaussian_filter():
    observations = np.zeros((100, 100))
    observations[50, 50] = 0.5

    result = smooth_observations_with_gaussian_filter(observations, sigma=10)

    assert np.sum(result > 0) > 1, "there should be more than one cell with a value greater than 0"
