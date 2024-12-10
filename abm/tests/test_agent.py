import pytest
from unittest.mock import Mock
import mesa

from abm.agent import Agent
from abm.exploration_strategy import (
    ExplorationStrategy,
    RandomWalkerExplorationStrategy,
)
from abm.exploitation_strategy import ExploitationStrategy
from abm.utils import ij2xy


@pytest.fixture
def setup_agent():
    # Create a mock model with a grid
    model = Mock()
    model.grid = mesa.space.MultiGrid(10, 10, torus=False)

    # Create mock strategies
    exploration_strategy = Mock(spec=ExplorationStrategy)
    exploitation_strategy = Mock(spec=ExploitationStrategy)

    # Create an agent
    agent = Agent(
        unique_id=1,
        model=model,
        resource_cluster_radius=1,
        social_info_quality=None,
        exploration_strategy=exploration_strategy,
        exploitation_strategy=exploitation_strategy,
    )

    # Place the agent at a starting position
    model.grid.place_agent(agent, (0, 0))

    return agent, exploration_strategy


@pytest.fixture
def setup_random_walk_agent():
    # Create a mock model with a grid
    model = Mock()
    model.grid = mesa.space.MultiGrid(10, 10, torus=False)

    # Create mock strategies
    exploration_strategy = RandomWalkerExplorationStrategy(alpha=1, grid_size=10)
    exploitation_strategy = Mock(spec=ExploitationStrategy)

    # Create an agent
    agent = Agent(
        unique_id=1,
        model=model,
        resource_cluster_radius=1,
        social_info_quality="sampling",
        exploration_strategy=exploration_strategy,
        exploitation_strategy=exploitation_strategy,
    )

    # Place the agent at a starting position
    model.grid.place_agent(agent, (0, 0))

    return agent, exploration_strategy


@pytest.mark.parametrize("destination", [(5, 5), (1, 3), (5, 9)])
def test_agent_moves_to_correct_destination(setup_agent, destination: tuple[int, int]):
    agent, exploration_strategy = setup_agent

    # Mock the choose_destination method to return a specific destination
    exploration_strategy.choose_destination.return_value = destination

    # Run the agent's step method
    agent.step()

    # Check if the agent's destination is set correctly
    assert agent.destination == ij2xy(*destination)

    # Move the agent towards the destination
    while agent.is_moving:
        agent.move()

    # Check if the agent has arrived at the correct destination
    assert agent.pos == ij2xy(*destination)


@pytest.mark.parametrize("other_agent_pos", [(5, 5), (1, 3), (5, 9)])
def test_random_walk_agent_moves_to_other_agent(
    setup_random_walk_agent, other_agent_pos: tuple[int, int]
):
    agent, exploration_strategy = setup_random_walk_agent

    # Place another agent at a specific position
    other_agent = Mock(spec=Agent)
    other_agent.pos = other_agent_pos
    other_agent.is_sampling = True
    other_agent.is_consuming = False
    agent.model.grid.place_agent(other_agent, other_agent.pos)

    # Add the other agent's position to the social locations
    agent.add_other_agent_locs()

    agent._is_moving = False
    agent._is_sampling = False

    # Run the agent's step method
    agent.step()

    # Check if the agent's destination is set to the other agent's position
    assert agent.destination == other_agent_pos

    # Move the agent towards the destination
    while agent.is_moving:
        agent.move()

    # Check if the agent has arrived at the other agent's position
    assert agent.pos == other_agent_pos
