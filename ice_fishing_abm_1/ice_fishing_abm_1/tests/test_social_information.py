import numpy as np

from ice_fishing_abm_1.ice_fishing_abm_1.social_information import estimate_social_vector


def test_estimate_social_vector():
    v = estimate_social_vector((0, 0), [(1, 1)])
    assert np.allclose(v, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))

    v = estimate_social_vector((0, 0), [(1, 1), (2, 2)])
    assert np.allclose(v, np.array([2 / np.sqrt(2), 2 / np.sqrt(2)]))
