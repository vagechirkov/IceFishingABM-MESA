import pytest
from unittest.mock import Mock
import mesa

from abm.agent import Agent
from abm.exploration_strategy import ExplorationStrategy
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
