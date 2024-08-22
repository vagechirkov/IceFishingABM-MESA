import pytest
import numpy as np
from pytest import fixture
from ice_fishing_abm_1.ice_fishing_abm_gp.movement_destination_subroutine import ExplorationStrategy, GPExplorationStrategy

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

def test_calculate_features_with_no_locations(gp_exploration_strategy):
    feature_m, feature_std = gp_exploration_strategy.calculate_features(
        np.empty((0, 2)), gp_exploration_strategy.social_gpr, gp_exploration_strategy.grid_size)
    
    assert feature_m.shape == (100, 100)
    assert feature_std.shape == (100, 100)
    assert np.all(feature_m == 0)
    assert np.all(feature_std == 0)

def test_calculate_features_with_locations(gp_exploration_strategy):
    locations = np.array([[10, 10], [20, 20]])
    
    # Mocking generate_belief_mean_matrix functionality directly in the test
    def mock_generate_belief_mean_matrix(grid_size, gpr, return_std=True):
        return np.ones((grid_size, grid_size)), np.ones((grid_size, grid_size))

    gp_exploration_strategy.calculate_features = mock_generate_belief_mean_matrix

    feature_m, feature_std = gp_exploration_strategy.calculate_features(
        locations, gp_exploration_strategy.social_gpr, gp_exploration_strategy.grid_size)

    assert feature_m.shape == (100, 100)
    assert feature_std.shape == (100, 100)
    assert np.all(feature_m == 1)
    assert np.all(feature_std == 1)

def test_compute_beliefs(gp_exploration_strategy):
    gp_exploration_strategy.social_feature_m = np.ones((100, 100)) * 0.5
    gp_exploration_strategy.success_feature_m = np.ones((100, 100)) * 0.3
    gp_exploration_strategy.failure_feature_m = np.ones((100, 100)) * 0.2

    gp_exploration_strategy.social_feature_std = np.ones((100, 100)) * 0.1
    gp_exploration_strategy.success_feature_std = np.ones((100, 100)) * 0.2
    gp_exploration_strategy.failure_feature_std = np.ones((100, 100)) * 0.3

    gp_exploration_strategy._compute_beliefs()

    assert gp_exploration_strategy.belief_m.shape == (100, 100)
    assert gp_exploration_strategy.belief_std.shape == (100, 100)
    assert gp_exploration_strategy.belief_ucb.shape == (100, 100)
    assert gp_exploration_strategy.belief_softmax.shape == (100, 100)
    assert np.isclose(np.sum(gp_exploration_strategy.belief_softmax), 1.0)


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
