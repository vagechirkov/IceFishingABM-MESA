from typing import Union
import mesa
import numpy as np

from .exploration_strategy import ExplorationStrategy
from .exploitation_strategy import ExploitationStrategy
from .resource import Resource
from .utils import ij2xy, xy2ij


class Agent(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        resource_cluster_radius,
        social_info_quality,
        exploration_strategy: ExplorationStrategy,
        exploitation_strategy: ExploitationStrategy,
    ):
        super().__init__(unique_id, model)
        # Parameters
        self.exploitation_strategy = exploitation_strategy
        self.exploration_strategy = exploration_strategy
        self.social_info_quality = social_info_quality

        # State variables
        self._is_moving: bool = False
        self._is_sampling: bool = False
        self._is_consuming: bool = False
        self._destination: Union[None, tuple[int, int]] = None
        self._time_on_patch: int = 0
        self._time_since_last_catch: int = 0
        self._collected_resource_last_spot: int = 0

        self.resource_cluster_radius = resource_cluster_radius
        self.success_locs = np.empty((0, 2))
        self.failure_locs = np.empty((0, 2))
        self.other_agent_locs = np.empty((0, 2))
        self.is_agent: bool = True

        # Output / Tracked variables
        self._collected_resource: int = 0
        self._traveled_distance: int = 0
        self._step_sizes = []  # Track movement distances
        self._time_to_first_catch = None  # Track time to first catch

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

    @property
    def traveled_distance(self):
        return self._traveled_distance

    def move(self):
        """
        Move agent one cell closer to the destination.
        """
        x, y = self.pos
        dx, dy = self._destination
        x_, y_ = x, y  # Store old position for step size calculation
        if x < dx:
            x += 1
        elif x > dx:
            x -= 1
        if y < dy:
            y += 1
        elif y > dy:
            y -= 1
        self.model.grid.move_agent(self, (x, y))

        # Calculate step size
        step_size = np.sqrt((x - x_) ** 2 + (y - y_) ** 2)
        self._step_sizes.append(step_size)
        self._traveled_distance += step_size  # I update the traveled distance here instead of incrementing by 1

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
        _is_resource_collected = False
        self._time_on_patch += 1
        self._is_consuming = False

        for neighbor in neighbors:
            if isinstance(neighbor, Resource) and neighbor.catch():
                self._collected_resource += 1
                self._collected_resource_last_spot += 1
                _is_resource_collected = True
                self._is_consuming = True

        # Save time to first catch
        if _is_resource_collected and self._time_to_first_catch is None:
            self._time_to_first_catch = self.model.schedule.steps

        if not _is_resource_collected:
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
                current_position=xy2ij(*self.pos),
                success_locs=self.success_locs,
                failure_locs=self.failure_locs,
                other_agent_locs=self.other_agent_locs,
            )
            self._is_moving = True

            # convert destination back to x,y
            self._destination = ij2xy(*self._destination)

        # Update social information at the end of each step
        self.add_other_agent_locs()

    def add_success_loc(self, loc: tuple):
        self.success_locs = np.vstack(
            [self.success_locs, np.array(xy2ij(*loc))[np.newaxis, :]]
        )

    def add_failure_loc(self, loc: tuple):
        self.failure_locs = np.vstack(
            [self.failure_locs, np.array(xy2ij(*loc))[np.newaxis, :]]
        )

    def add_other_agent_locs(self):
        self.other_agent_locs = np.empty((0, 2))

        # avoid searching for neighbors if social info is ignored
        if self.social_info_quality is None:
            return

        other_agents = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.model.grid.width
        )

        agents = []
        if self.social_info_quality == "consuming":
            # Only get positions of agents that are both sampling AND consuming
            agents = [
                np.array(xy2ij(*agent.pos))[np.newaxis, :]
                for agent in other_agents
                if isinstance(agent, Agent) and agent.is_sampling and agent.is_consuming
            ]
        elif self.social_info_quality == "sampling":
            # Get positions of all sampling agents
            agents = [
                np.array(xy2ij(*agent.pos))[np.newaxis, :]
                for agent in other_agents
                if isinstance(agent, Agent) and agent.is_sampling
            ]
        else:
            raise ValueError(
                f"Unknown social info quality parameter value: {self.social_info_quality}"
            )

        # Stack positions if any agents were found
        if len(agents) > 0:
            self.other_agent_locs = np.vstack(agents)
