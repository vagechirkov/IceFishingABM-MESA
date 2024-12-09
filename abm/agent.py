from typing import Union
import mesa
import numpy as np

from .exploration_strategy import ExplorationStrategy
from .exploitation_strategy import ExploitationStrategy
from .resource import Resource
from .utils import x_y_to_i_j


class Agent(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        resource_cluster_radius,
        exploration_strategy: ExplorationStrategy,
        exploitation_strategy: ExploitationStrategy,
    ):
        super().__init__(unique_id, model)
        # Parameters
        self.exploitation_strategy = exploitation_strategy
        self.exploration_strategy = exploration_strategy

        # State variables
        self._is_moving: bool = False
        self._is_sampling: bool = False
        self._is_consuming: bool = False
        self._destination: Union[None, tuple[int, int]] = None
        self._time_on_patch: int = 0
        self._time_since_last_catch: int = 0
        self._collected_resource_last_spot: int = 0
        self._collected_resource: int = 0
        self.resource_cluster_radius = resource_cluster_radius
        self.success_locs = np.empty((0, 2))
        self.failure_locs = np.empty((0, 2))
        self.other_agent_locs = np.empty((0, 2))
        self.is_agent: bool = True
        self.model.social_information = None  # If True, agent has full access to social catch information, by default this will be False

    @property
    def is_moving(self):
        return self._is_moving

    @property
    def is_sampling(self):
        return self._is_sampling

    @property
    def is_consuming(self):
        return self._is_consuming

    @property
    def destination(self):
        return self._destination

    @property
    def collected_resource(self):
        return self._collected_resource

    def move(self):
        """
        Move agent one cell closer to the destination.
        Add variable of distance moved by agent (TODO)
        """
        x, y = self.pos
        dx, dy = self._destination
        if x < dx:
            x += 1
        elif x > dx:
            x -= 1
        if y < dy:
            y += 1
        elif y > dy:
            y -= 1
        self.model.grid.move_agent(self, (x, y))

        if np.array_equal(self.pos, self._destination):
            self._is_moving = False
            self._is_sampling = True

    def sample(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,
            include_center=True,
            radius=self.resource_cluster_radius,
        )
        resource_collected = False
        self._time_on_patch += 1
        self._is_consuming = False

        for neighbor in neighbors:
            if isinstance(neighbor, Resource) and neighbor.catch():
                self._collected_resource += 1
                self._collected_resource_last_spot += 1
                resource_collected = True
                self._is_consuming = True

        if not resource_collected:
            self._time_since_last_catch += 1

        if not self.exploitation_strategy.stay_on_patch(
            self._time_on_patch, self._time_since_last_catch
        ):
            self._is_sampling = False
            self._time_since_last_catch = 0

            if self._collected_resource_last_spot == 0:
                self.add_failure_loc(self.pos)

            self._collected_resource_last_spot = 0
            self._time_on_patch = 0

    def step(self):
        if self._is_moving and not self._is_sampling:
            self.move()
        elif self._is_sampling and not self._is_moving:
            self.sample()
        else:
            # Select a new destination
            self._destination = self.exploration_strategy.choose_destination(
                x_y_to_i_j(*self.pos),
                self.success_locs,
                self.failure_locs,
                self.other_agent_locs,
            )
            self._is_moving = True

        # Update social information at the end of each step
        self.add_other_agent_locs()

    def add_success_loc(self, loc: tuple):
        self.success_locs = np.vstack(
            [self.success_locs, np.array(x_y_to_i_j(*loc))[np.newaxis, :]]
        )

    def add_failure_loc(self, loc: tuple):
        self.failure_locs = np.vstack(
            [self.failure_locs, np.array(x_y_to_i_j(*loc))[np.newaxis, :]]
        )

    def add_other_agent_locs(self):
        other_agents = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.model.grid.width
        )
        # Initialize agents as empty list
        agents = []
        if not self.model.social_information:
            # No social info case - empty array
            self.other_agent_locs = np.empty((0, 2))
        elif self.model.social_information == "consuming":
            # Only get positions of agents that are both sampling AND consuming
            agents = [
                np.array(x_y_to_i_j(*agent.pos))[np.newaxis, :]
                for agent in other_agents
                if isinstance(agent, Agent) and agent.is_sampling and agent.is_consuming
            ]
        elif self.model.social_information == "sampling":
            # Get positions of all sampling agents
            agents = [
                np.array(x_y_to_i_j(*agent.pos))[np.newaxis, :]
                for agent in other_agents
                if isinstance(agent, Agent) and agent.is_sampling
            ]

        # Stack positions if any agents were found
        if len(agents) > 0:
            self.other_agent_locs = np.vstack(agents)
        else:
            self.other_agent_locs = np.empty((0, 2))
