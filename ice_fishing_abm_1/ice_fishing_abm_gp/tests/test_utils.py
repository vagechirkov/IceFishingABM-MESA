from ice_fishing_abm_1.ice_fishing_abm_gp.utils import x_y_to_i_j


def test_x_y_to_i_j():
    assert x_y_to_i_j(1, 2) == (2, 1)
