import pytest
import numpy as np
from unittest.mock import Mock
from ice_fishing_abm_1.ice_fishing_abm_gp.movement_destination_subroutine import GPMovementDestinationSubroutine

def test_calculate_features():
    # Create a mock for the GaussianProcessRegressor
    mock_gpr = Mock()
    mock_gpr.fit = Mock()
    mock_gpr.predict = Mock(return_value=(np.zeros((10, 10)), np.ones((10, 10))))  # Mock the return value to be a tuple of arrays

    grid_size = 10
    locations = np.array([[1, 1], [2, 2]])

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

    # Initialize with the mock_agent
    subroutine = GPMovementDestinationSubroutine(agent=mock_agent)

    # Test the calculate_features method
    feature_m, feature_std = subroutine.calculate_features(locations, mock_gpr, grid_size)

    assert feature_m.shape == (grid_size, grid_size)
    assert feature_std.shape == (grid_size, grid_size)

if __name__ == "__main__":
    pytest.main()
