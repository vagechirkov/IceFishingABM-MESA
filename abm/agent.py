from typing import Union
import mesa
import numpy as np
from scipy.spatial.distance import pdist

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
        self._traveled_distance_euclidean: int = 0
        self._traveled_distance_manhattan: int = 0
        self._step_sizes = []  # Track movement distances
        self._time_to_first_catch = None  # Track time to first catch
        self._total_sampling_time: int = 0    # Total time spent sampling
        self._total_consuming_time: int = 0    # Total time spent consuming
        self._cluster_catches: int = 0  # Number of successful catches in clusters
        self._last_catch_pos = None     # Track last catch position

        

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
    def traveled_distance_euclidean(self):
        return self._traveled_distance_euclidean

    @property
    def traveled_distance_manhattan(self):
        return self._traveled_distance_manhattan

    @property
    def total_sampling_time(self):
        return self._total_sampling_time

    @property
    def total_consuming_time(self):
        return self._total_consuming_time


    @property
    def cluster_catches(self):
        return self._cluster_catches

    def move(self):
        """
        Move agent one cell closer to the destination.
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
        _is_resource_collected = False
        self._time_on_patch += 1
        self._is_consuming = False

        # if not consuming then sampling:
        self._total_sampling_time += 1

        for neighbor in neighbors:
            if isinstance(neighbor, Resource) and neighbor.catch():
                self._collected_resource += 1
                self._collected_resource_last_spot += 1
                _is_resource_collected = True
                self._is_consuming = True
                self._total_consuming_time += 1

                # Track cluster catches
                if self._last_catch_pos is None:
                    self._cluster_catches += 1
                    self._last_catch_pos = self.pos
                else:
                    # If catch is in same cluster (within radius)
                    distance = np.linalg.norm(np.array(self.pos) - np.array(self._last_catch_pos))
                    if distance > self.resource_cluster_radius * 2:  # New cluster
                        self._cluster_catches += 1
                    self._last_catch_pos = self.pos

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

            # Calculate euclidean distance to destination using scipy pdist
            step_size = pdist(np.array([self.pos, self._destination]), "euclidean")[0]
            self._step_sizes.append(step_size)
            self._traveled_distance_euclidean += step_size
            self._traveled_distance_manhattan += pdist(np.array([self.pos, self._destination]), "cityblock")[0]

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
        
        other_agents = self.model.schedule.agents
        other_agents = [agent for agent in other_agents if
                        isinstance(agent, Agent) and (agent.unique_id != self.unique_id)]

        if self.social_info_quality == "consuming":
            # Only get positions of agents that are both sampling AND consuming
            agents = [
                np.array(xy2ij(*agent.pos))[np.newaxis, :]
                for agent in other_agents
                if agent.is_sampling and agent.is_consuming
            ]
        elif self.social_info_quality == "sampling":
            # Get positions of all sampling agents
            agents = [
                np.array(xy2ij(*agent.pos))[np.newaxis, :]
                for agent in other_agents
                if agent.is_sampling
            ]
        else:
            raise ValueError(
                f"Unknown social info quality parameter value: {self.social_info_quality}"
            )

        # Stack positions if any agents were found
        if len(agents) > 0:
            self.other_agent_locs = np.vstack(agents)
