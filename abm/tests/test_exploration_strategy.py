import pytest
import numpy as np
from abm.exploration_strategy import (
    ExplorationStrategy,
    RandomWalkerExplorationStrategy,
    SocialInfotaxisExplorationStrategy,
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
        current_position=np.zeros(2),
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=np.empty((0, 2))
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
    # Test invalid position array
    with pytest.raises(AssertionError):
        exploration_strategy.choose_destination(
            current_position=np.array([1, 2, 3]),  # 3D position
            success_locs=np.ones((1, 2)),
            failure_locs=np.ones((1, 2)),
            other_agent_locs=np.ones((1, 2)),
        )

    # Test invalid locations array shape
    with pytest.raises(AssertionError):
        exploration_strategy.choose_destination(
            current_position=np.array([1, 2]),
            success_locs=np.ones((2, 3)),  # Wrong shape - 3 columns
            failure_locs=np.ones((1, 2)),
            other_agent_locs=np.ones((1, 2)),
        )

    # Test invalid catch_locs shape
    with pytest.raises(AssertionError):
        exploration_strategy.choose_destination(
            current_position=np.array([1, 2]),
            success_locs=np.empty((0, 2)),
            failure_locs=np.array([[1, 2, 3]]),
            other_agent_locs=np.empty((0, 2))
        )

    # Test invalid loss_locs shape
    with pytest.raises(AssertionError):
        exploration_strategy.choose_destination(
            current_position=np.array([1, 2]),
            success_locs=np.empty((0, 2)),
            failure_locs=np.empty((0, 2)),
            other_agent_locs=np.array([[1, 2, 3]])
        )


# Tests for RandomWalkerExplorationStrategy
def test_levy_flight_trivial(random_walker):
    # Ensure destination is within grid bounds
    current_position = np.array([5, 5])
    destination = random_walker._levy_flight(current_position)
    x, y = destination
    assert 0 <= x < random_walker.grid_size
    assert 0 <= y < random_walker.grid_size


def test_levy_flight_specific_values():
    np.random.seed(1)
    random_walker = RandomWalkerExplorationStrategy(
        grid_size=10, mu=1.5, dmin=1.0, L=5.0, alpha=0.1
    )
    current_position = np.array([5, 5])
    destination = random_walker._levy_flight(current_position)
    expected_destination = np.array([4, 5])
    np.testing.assert_array_equal(destination, expected_destination)


@pytest.mark.xfail
# TODO: fix this
def test_adjust_for_social_cue():
    np.random.seed(0)
    rw = RandomWalkerExplorationStrategy(
        grid_size=10, mu=1.5, dmin=1.0, L=5.0, alpha=0.1
    )
    levi_flight_destination = np.array([0, 0])
    current_positions = np.array([4, 4])

    # No social information case - empty array
    rw.alpha = 1
    rw.destination = levi_flight_destination
    social_locs = np.array([])
    rw._adjust_for_social_cue(current_positions, social_locs)
    assert np.array_equal(rw.destination, levi_flight_destination)

    # No social influence case (alpha = 0)
    rw.alpha = 0
    rw.destination = levi_flight_destination
    social_locs = np.array([[5, 5], [1, 1]])
    rw._adjust_for_social_cue(current_positions, social_locs)
    assert np.allclose(rw._prob_social, 1, atol=1e-5)
    assert np.array_equal(
        rw.destination, levi_flight_destination
    )  # Keeps original destination

    # Strong social influence case (high alpha)
    rw.alpha = 100
    rw.destination = levi_flight_destination
    social_locs = np.array([[5, 5], [1, 1]])
    rw._adjust_for_social_cue(current_positions, social_locs)
    assert np.allclose(rw._prob_social, 0, atol=1e-5)
    assert np.array_equal(rw.destination, [5, 5])  # Moves to nearest social location


def test_destination_with_social_cue(random_walker):
    # Test if the walker selects a destination considering social cues
    social_locs = np.array([[3, 3], [7, 7]])
    current_position = np.array([5, 5])
    destination = random_walker.choose_destination(
        current_position=current_position,
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=social_locs
    )
    x, y = destination
    assert 0 <= x < random_walker.grid_size
    assert 0 <= y < random_walker.grid_size


## Tests for Social Infotaxis Exploration Strategy
@pytest.fixture
def social_infotaxis():
    np.random.seed(0)
    return SocialInfotaxisExplorationStrategy(grid_size=10, tau=0.1, epsilon=0.1)


@pytest.mark.xfail
# Tests for SocialInfotaxisExplorationStrategy
def test_choose_destination_trivial_social_infotaxis(social_infotaxis):
    # Basic test to ensure destination is within grid bounds
    belief = np.ones((10, 10)) / 100  # Uniform belief distribution
    action_set = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
    destination = social_infotaxis.choose_destination(
        current_position=np.array([5, 5]),
        belief=belief,
        action_set=action_set,
    )
    x, y = destination
    assert 0 <= x < social_infotaxis.grid_size
    assert 0 <= y < social_infotaxis.grid_size


def test_entropy_computation_social_infotaxis(social_infotaxis):
    # Explicit entropy computation check
    belief = np.array([[0.25, 0.25], [0.25, 0.25]])  # Uniform belief
    entropy = social_infotaxis._compute_entropy(belief)
    expected_entropy = -np.sum(belief * np.log(belief + 1e-9))
    assert np.isclose(entropy, expected_entropy, atol=1e-4)


@pytest.mark.xfail
def test_expected_entropy_computation_social_infotaxis(social_infotaxis):
    belief = np.ones((10, 10)) / 100  # Uniform belief distribution
    action_set = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
    current_position = np.array([5, 5])
    action = action_set[0]  # Move up
    expected_entropy = social_infotaxis._compute_expected_entropy(
        current_position, action, belief
    )

    # Manually simulate belief update and entropy calculation
    new_position = np.clip(current_position + action, 0, 9)
    new_belief = belief.copy()
    social_infotaxis._update_belief(new_belief, new_position)
    manual_entropy = social_infotaxis._compute_entropy(new_belief)

    print(f"Expected Entropy: {expected_entropy}, Manual Entropy: {manual_entropy}")
    assert np.isclose(expected_entropy, manual_entropy, atol=1e-4)


def test_belief_update_social_infotaxis(social_infotaxis):
    # Verify that the belief state is updated correctly
    belief = np.ones((10, 10)) / 100  # Uniform belief distribution
    new_position = np.array([5, 5])
    social_infotaxis._update_belief(belief, new_position)
    assert np.isclose(np.sum(belief), 1.0, atol=1e-5)  # Belief should remain normalized


@pytest.mark.xfail
def test_action_selection_social_infotaxis(social_infotaxis):
    # Verify that actions are selected based on softmax probabilities
    belief = np.ones((10, 10)) / 100  # Uniform belief distribution
    action_set = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
    current_position = np.array([5, 5])

    # Run action selection multiple times and check bounds
    for _ in range(10):
        destination = social_infotaxis.choose_destination(
            current_position=current_position,
            belief=belief,
            action_set=action_set,
        )
        x, y = destination
        assert 0 <= x < social_infotaxis.grid_size
        assert 0 <= y < social_infotaxis.grid_size
