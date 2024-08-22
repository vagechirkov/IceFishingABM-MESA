import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import fixture
from ice_fishing_abm_1.ice_fishing_abm_gp.movement_destination_subroutine import ExplorationStrategy, \
    GPExplorationStrategy


@fixture
def exploration_strategy():
    return ExplorationStrategy()


@fixture
def gp_exploration_strategy():
    return GPExplorationStrategy()


def test_initialization_exploration_strategy(exploration_strategy):
    assert exploration_strategy.grid_size == 100
    assert exploration_strategy.ucb_beta == 0.2
    assert exploration_strategy.softmax_tau == 0.01
    assert exploration_strategy.mesh.shape == (10000, 2)
    assert exploration_strategy.belief_softmax.shape == (100, 100)


def test_initialization_gp_exploration_strategy(gp_exploration_strategy):
    assert gp_exploration_strategy.grid_size == 100
    assert gp_exploration_strategy.w_social == 0.4
    assert gp_exploration_strategy.w_success == 0.3
    assert gp_exploration_strategy.w_failure == 0.3
    assert gp_exploration_strategy.social_feature_m.shape == (100, 100)
    assert gp_exploration_strategy.success_feature_std.shape == (100, 100)


@pytest.mark.parametrize("social_length_scale", [1, 5, 12, 20])
def test_calculate_gp_feature(social_length_scale):
    gp_model = GPExplorationStrategy(social_length_scale=social_length_scale)

    # empty data case
    data = np.empty((0, 2))
    feature_m, feature_std = gp_model._calculate_gp_feature(data, gp_model.social_gpr, gp_model.grid_size)

    assert feature_m.shape == (100, 100)
    assert feature_std.shape == (100, 100)
    assert np.all(feature_m == 0)
    assert np.all(feature_std == 0)

    # not empty data case
    data = np.array([[10, 10], [20, 20]])
    feature_m, feature_std = gp_model._calculate_gp_feature(data, gp_model.social_gpr, gp_model.grid_size)

    assert feature_m.shape == (100, 100)
    assert feature_std.shape == (100, 100)
    assert np.allclose(feature_m[10, 10], 1, atol=0.1)

    # when debugging, it is nice to plot the feature_m to visually inspect
    if False:
        plt.imshow(feature_m)
        plt.colorbar()
        plt.show()


@pytest.mark.parametrize("w_social,w_success,w_failure,grid_size",
                         [(0.4, 0.3, 0.3, 50), (0.1, 0.5, 0.4, 50), (0.2, 0.2, 0.6, 100)])
def test_compute_beliefs(w_social, w_success, w_failure, grid_size):
    gp_model = GPExplorationStrategy(w_social=w_social, w_success=w_success, w_failure=w_failure, grid_size=grid_size)
    gp_model.social_feature_m = np.ones((grid_size, grid_size))
    gp_model.success_feature_m = np.ones((grid_size, grid_size))
    gp_model.failure_feature_m = np.ones((grid_size, grid_size))

    gp_model.social_feature_std = np.ones((grid_size, grid_size))
    gp_model.success_feature_std = np.ones((grid_size, grid_size))
    gp_model.failure_feature_std = np.ones((grid_size, grid_size))

    gp_model._compute_beliefs()

    assert gp_model.belief_m.shape ==(grid_size, grid_size)
    assert gp_model.belief_std.shape ==(grid_size, grid_size)
    assert gp_model.belief_ucb.shape ==(grid_size, grid_size)
    assert gp_model.belief_softmax.shape ==(grid_size, grid_size)
    assert np.isclose(np.sum(gp_model.belief_softmax), 1.0)


def test_choose_destination(exploration_strategy):
    exploration_strategy.belief_softmax = np.ones((100, 100)) / 10000
    exploration_strategy.choose_destination()
    assert exploration_strategy.destination.shape == (2,)
    assert exploration_strategy.destination[0] < 100
    assert exploration_strategy.destination[1] < 100


def test_movement_destination(gp_exploration_strategy):
    # Mocking the values directly
    gp_exploration_strategy.social_feature_m = np.ones((100, 100))
    gp_exploration_strategy.social_feature_std = np.ones((100, 100))
    gp_exploration_strategy.success_feature_m = np.ones((100, 100))
    gp_exploration_strategy.success_feature_std = np.ones((100, 100))
    gp_exploration_strategy.failure_feature_m = np.ones((100, 100))
    gp_exploration_strategy.failure_feature_std = np.ones((100, 100))

    # Mocking the agent property
    gp_exploration_strategy.agent = type('Agent', (object,), {'_is_moving': False})()

    gp_exploration_strategy.movement_destination()

    assert gp_exploration_strategy.belief_m.shape == (100, 100)
    assert gp_exploration_strategy.belief_ucb.shape == (100, 100)
    assert gp_exploration_strategy.destination.shape == (2,)
    assert gp_exploration_strategy.destination[0] < 100
    assert gp_exploration_strategy.destination[1] < 100
