import pytest
import numpy as np
from unittest.mock import Mock
from ice_fishing_abm_1.ice_fishing_abm_gp.movement_destination_subroutine import GPMovementDestinationSubroutine

@pytest.fixture
def setup_mock_agent():
    # Create a mock for the GaussianProcessRegressor
    mock_gpr = Mock()
    mock_gpr.fit = Mock()
    return mock_gpr

def test_calculate_features_basic(setup_mock_agent):
    mock_gpr = setup_mock_agent
    grid_size = 10
    locations = np.array([[1, 1], [2, 2]])

    # Mock the return value of predict
    mock_gpr.predict = Mock(return_value=(np.zeros((10, 10)), np.ones((10, 10))))

    mock_agent = Mock()
    mock_agent.model.grid_size = grid_size
    mock_agent.social_gpr = mock_gpr
    mock_agent.success_gpr = mock_gpr
    mock_agent.failure_gpr = mock_gpr
    mock_agent.other_agent_locs = np.empty((0, 2))
    mock_agent.success_locs = np.empty((0, 2))
    mock_agent.failure_locs = np.empty((0, 2))
    mock_agent.mesh = np.array(np.meshgrid(range(grid_size), range(grid_size))).reshape(2, -1).T
    mock_agent.mesh_indices = np.arange(0, mock_agent.mesh.shape[0])
    mock_agent.ucb_beta = 0.2
    mock_agent.softmax_tau = 0.01

    subroutine = GPMovementDestinationSubroutine(agent=mock_agent)
    feature_m, feature_std = subroutine.calculate_features(locations, mock_gpr, grid_size)

    assert feature_m.shape == (grid_size, grid_size)
    assert feature_std.shape == (grid_size, grid_size)

def test_calculate_features_different_grid_sizes(setup_mock_agent):
    mock_gpr = setup_mock_agent
    for grid_size in [5, 10, 15]:
        locations = np.array([[1, 1], [2, 2]])
        mock_gpr.predict = Mock(return_value=(np.zeros((grid_size, grid_size)), np.ones((grid_size, grid_size))))

        mock_agent = Mock()
        mock_agent.model.grid_size = grid_size
        mock_agent.social_gpr = mock_gpr
        mock_agent.success_gpr = mock_gpr
        mock_agent.failure_gpr = mock_gpr
        mock_agent.other_agent_locs = np.empty((0, 2))
        mock_agent.success_locs = np.empty((0, 2))
        mock_agent.failure_locs = np.empty((0, 2))
        mock_agent.mesh = np.array(np.meshgrid(range(grid_size), range(grid_size))).reshape(2, -1).T
        mock_agent.mesh_indices = np.arange(0, mock_agent.mesh.shape[0])
        mock_agent.ucb_beta = 0.2
        mock_agent.softmax_tau = 0.01

        subroutine = GPMovementDestinationSubroutine(agent=mock_agent)
        feature_m, feature_std = subroutine.calculate_features(locations, mock_gpr, grid_size)

        assert feature_m.shape == (grid_size, grid_size)
        assert feature_std.shape == (grid_size, grid_size)

def test_calculate_features_empty_locations(setup_mock_agent):
    mock_gpr = setup_mock_agent
    grid_size = 10
    locations = np.array([])  # Empty locations

    # Mock the return value of predict
    mock_gpr.predict = Mock(return_value=(np.zeros((10, 10)), np.ones((10, 10))))

    mock_agent = Mock()
    mock_agent.model.grid_size = grid_size
    mock_agent.social_gpr = mock_gpr
    mock_agent.success_gpr = mock_gpr
    mock_agent.failure_gpr = mock_gpr
    mock_agent.other_agent_locs = np.empty((0, 2))
    mock_agent.success_locs = np.empty((0, 2))
    mock_agent.failure_locs = np.empty((0, 2))
    mock_agent.mesh = np.array(np.meshgrid(range(grid_size), range(grid_size))).reshape(2, -1).T
    mock_agent.mesh_indices = np.arange(0, mock_agent.mesh.shape[0])
    mock_agent.ucb_beta = 0.2
    mock_agent.softmax_tau = 0.01

    subroutine = GPMovementDestinationSubroutine(agent=mock_agent)
    feature_m, feature_std = subroutine.calculate_features(locations, mock_gpr, grid_size)

    assert feature_m.shape == (grid_size, grid_size)
    assert feature_std.shape == (grid_size, grid_size)

def test_calculate_features_with_edge_values(setup_mock_agent):
    mock_gpr = setup_mock_agent
    grid_size = 10
    locations = np.array([[0, 0], [9, 9]])

    # Mock the return value of predict with edge values
    mock_gpr.predict = Mock(return_value=(np.full((10, 10), 100), np.full((10, 10), 1)))

    mock_agent = Mock()
    mock_agent.model.grid_size = grid_size
    mock_agent.social_gpr = mock_gpr
    mock_agent.success_gpr = mock_gpr
    mock_agent.failure_gpr = mock_gpr
    mock_agent.other_agent_locs = np.empty((0, 2))
    mock_agent.success_locs = np.empty((0, 2))
    mock_agent.failure_locs = np.empty((0, 2))
    mock_agent.mesh = np.array(np.meshgrid(range(grid_size), range(grid_size))).reshape(2, -1).T
    mock_agent.mesh_indices = np.arange(0, mock_agent.mesh.shape[0])
    mock_agent.ucb_beta = 0.2
    mock_agent.softmax_tau = 0.01

    subroutine = GPMovementDestinationSubroutine(agent=mock_agent)
    feature_m, feature_std = subroutine.calculate_features(locations, mock_gpr, grid_size)

    assert feature_m.shape == (grid_size, grid_size)
    assert feature_std.shape == (grid_size, grid_size)
    assert np.all(feature_m == 100)
    assert np.all(feature_std == 1)

if __name__ == "__main__":
    pytest.main()
