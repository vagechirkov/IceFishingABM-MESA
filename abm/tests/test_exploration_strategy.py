import pytest
import numpy as np
from abm.exploration_strategy import (
    ExplorationStrategy,
    RandomWalkerExplorationStrategy,
)


@pytest.fixture
def exploration_strategy():
    np.random.seed(0)
    return ExplorationStrategy(grid_size=10)


@pytest.fixture
def random_walker():
    np.random.seed(0)
    return RandomWalkerExplorationStrategy(
        grid_size=10, mu=1.5, dmin=1.0, L=5.0, alpha=0.1
    )


# Tests for ExplorationStrategy
def test_choose_destination_trivial(exploration_strategy):
    # Basic test to ensure destination is within grid bounds
    destination = exploration_strategy.choose_destination(
        np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
    )
    x, y = destination
    assert 0 <= x < exploration_strategy.grid_size
    assert 0 <= y < exploration_strategy.grid_size


def test_choose_destination_probability_distribution(exploration_strategy):
    # Check if belief_softmax is normalized and has the expected shape
    assert np.isclose(np.sum(exploration_strategy.belief_softmax), 1.0, atol=1e-5)
    assert exploration_strategy.belief_softmax.shape == (
        exploration_strategy.grid_size,
        exploration_strategy.grid_size,
    )


def test_invalid_inputs(exploration_strategy):
    # Verify that invalid input raises an assertion error
    with pytest.raises(AssertionError):
        exploration_strategy.choose_destination(
            np.array([1, 2]), np.empty((0, 2)), np.empty((0, 2))
        )

    with pytest.raises(AssertionError):
        exploration_strategy.choose_destination(
            np.empty((0, 2)), np.array([[1, 2, 3]]), np.empty((0, 2))
        )


# Tests for RandomWalkerExplorationStrategy
def test_levy_flight_trivial(random_walker):
    # Ensure destination is within grid bounds
    destination = random_walker._levy_flight()
    x, y = destination
    assert 0 <= x < random_walker.grid_size
    assert 0 <= y < random_walker.grid_size


def test_levy_flight_specific_values():
    # Use a specific random seed to check for expected output
    np.random.seed(1)
    random_walker = RandomWalkerExplorationStrategy(
        grid_size=10, mu=1.5, dmin=1.0, L=5.0, alpha=0.1
    )
    destination = random_walker._levy_flight()
    expected_destination = np.array([3, 4])
    np.testing.assert_array_equal(destination, expected_destination)


def test_adjust_for_social_cue():
    np.random.seed(0)
    rw = RandomWalkerExplorationStrategy(
        grid_size=10, mu=1.5, dmin=1.0, L=5.0, alpha=0.1
    )
    levi_flight_destination = np.array([0, 0])
    current_positions = np.array([4, 4])

    # No social information
    rw.alpha = 1
    rw.destination = levi_flight_destination
    social_locs = np.array([])
    rw._adjust_for_social_cue(current_positions, social_locs)
    assert np.array_equal(rw.destination, levi_flight_destination)

    # Always move towards social cue
    rw.alpha = 0
    rw.destination = levi_flight_destination
    social_locs = np.array([[5, 5], [1, 1]])
    rw._adjust_for_social_cue(current_positions, social_locs)
    assert np.allclose(rw._prob_social, 1, atol=1e-5)
    assert np.array_equal(rw.destination, [5, 5])

    # Never move towards social cue
    rw.alpha = 100
    rw.destination = levi_flight_destination
    social_locs = np.array([[5, 5], [1, 1]])
    rw._adjust_for_social_cue(current_positions, social_locs)
    assert np.allclose(rw._prob_social, 0, atol=1e-5)
    assert np.array_equal(rw.destination, levi_flight_destination)


def test_destination_with_social_cue(random_walker):
    # Test if the walker selects a destination considering social cues
    social_locs = np.array([[3, 3], [7, 7]])
    destination = random_walker.choose_destination(
        social_locs, np.empty((0, 2)), np.empty((0, 2)), social_cue=True
    )
    x, y = destination
    assert 0 <= x < random_walker.grid_size
    assert 0 <= y < random_walker.grid_size


def test_get_new_position(random_walker):
    # Check new position calculation with wrapping around the grid
    dx, dy = 3, 3
    new_position = random_walker._get_new_position(dx, dy)
    expected_position = np.array([8, 8])  # Assuming center start in a 10x10 grid
    np.testing.assert_array_equal(new_position, expected_position)
