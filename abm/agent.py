from typing import Union
import numpy as np
import mesa
from scipy.spatial.distance import pdist

from abm.resource import Resource
from abm.utils import ij2xy, xy2ij
from abm.exploitation_strategy import IceFishingExploitationStrategy
from abm.exploration_strategy import KernelBeliefExploration


class Agent(mesa.Agent):
    def __init__(
        self,
        model,
        initial_position: tuple,
        exploration_strategy: KernelBeliefExploration,
        exploitation_strategy: IceFishingExploitationStrategy,
        speed_m_per_step: float = 1.0,  # 15.0,
        margin_from_others: float = 0.0,  # 5.0
        social_info_quality = "sampling",
        resource_cluster_radius = None
    ):
        super().__init__(model)
        self.model.grid.place_agent(self, initial_position)

        # Parameters
        self.exploitation_strategy = exploitation_strategy
        self.exploration_strategy = exploration_strategy
        self.social_info_quality = social_info_quality
        self._move_budget = speed_m_per_step
        self._margin_from_others = margin_from_others

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
        self._sampling_time_current_spot: int = 0
        self._time_sampling_successful_spots: int = 0
        self._time_sampling_failure_spots: int = 0

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

    @property
    def n_successful_locations(self):
        return self.success_locs.shape[0]

    @property
    def n_failure_locations(self):
        return self.failure_locs.shape[0]

    @property
    def time_sampling_successful_spots(self):
        return self._time_sampling_successful_spots

    @property
    def time_sampling_failure_spots(self):
        return self._time_sampling_failure_spots

    def move(self):
        """
        Move toward destination using a per-step movement budget (meters).
        If arriving exactly at the destination but others are within the
        required margin, pick the nearest valid cell to the destination
        that is at least `self._margin_from_others` away from all others.
        """
        x, y = self.pos
        dx, dy = self._destination
        vx, vy = dx - x, dy - y
        dist = float(np.hypot(vx, vy))

        # unit direction (handle zero-length safely)
        if dist > 0:
            ux, uy = vx / dist, vy / dist
        else:
            ux, uy = 0.0, 0.0

        s_max = min(self._move_budget, dist)

        # propose the usual step (rounded to nearest grid cell)
        sx = int(np.rint(ux * s_max))
        sy = int(np.rint(uy * s_max))
        new_pos = (x + sx, y + sy)

        # gather other agents' positions (No, 2)
        others = self.other_agent_locs.copy()
        margin = float(self._margin_from_others)

        def in_bounds(p):
            return (0 <= p[0] < self.model.grid.width) and (0 <= p[1] < self.model.grid.height)

        def ok_spacing(p):
            if others.size == 0:
                return True
            d = np.linalg.norm(others - np.array(p), axis=1)
            return np.all(d >= margin)

        if (dist < self._move_budget) and not ok_spacing(new_pos):
            chosen = None
            max_radius = int(np.ceil(self._move_budget)) + 5  # modest expansion beyond margin
            found = False
            for r in range(0, max_radius + 1):
                for i in range(-r, r + 1):
                    for j in range(-r, r + 1):
                        cx = int(dx + i)
                        cy = int(dy + j)
                        cand = (cx, cy)
                        if not in_bounds(cand):
                            continue
                        # must be within this tick's movement budget from current pos
                        # if np.hypot(cx - x, cy - y) > s_max + 1e-9:
                        #     continue
                        # only accept if spacing is OK (since we're in the bubble)
                        if ok_spacing(cand):
                            chosen = cand
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

            new_pos = chosen if chosen is not None else (x, y)
            # record actual per-tick destination you targeted
            self._destination = new_pos

        # Move on the grid
        self.model.grid.move_agent(self, new_pos)

        # Only mark arrived if we exactly hit the destination
        if new_pos == (dx, dy):
            self._is_moving = False
            self._is_sampling = True

    def sample_fish_density(self):
        self._total_sampling_time += 1
        self._sampling_time_current_spot += 1
        j, i = self.pos[0], self.pos[1]

        # if catch
        if self.model.sample_fish_density(i, j):
            self._collected_resource += 1
            self._collected_resource_last_spot += 1
            self._is_consuming = True
            self._time_since_last_catch = 0
        else:
            self._time_since_last_catch += 1

        # if decided to leave (not to stay)
        # NOTE: IceFishingExploitationStrategy
        if not self.exploitation_strategy.stay_on_patch(
                time_since_last_catch=self._time_since_last_catch,
                is_spot_successful=float(self._collected_resource_last_spot > 0),
                z_social_feature=self.exploration_strategy.social_feature_kde[xy2ij(*self.pos)]
        ):
            self._is_sampling = False
            self._is_consuming = False
            self._time_since_last_catch = 0
            self.exploitation_strategy.reset_p_leave()

            if self._collected_resource_last_spot == 0:
                self.add_failure_loc(self.pos)
                self._time_sampling_failure_spots += self._sampling_time_current_spot
            else:
                self.add_success_loc(self.pos)
                self._time_sampling_successful_spots += self._sampling_time_current_spot

            self._collected_resource_last_spot = 0
            self._sampling_time_current_spot = 0

    def step(self):
        # Update social information at the beginning of each step except the very first
        if self.model.steps > 1:
            self.add_other_agent_locs()

            # update features
            self.exploration_strategy.update_features(
                current_position=xy2ij(*self.pos),
                success_locs=self.success_locs,
                failure_locs=self.failure_locs,
                other_agent_locs=self.other_agent_locs
            )

        if self._is_moving and not self._is_sampling:
            self.move()
        elif self._is_sampling and not self._is_moving:
            self.sample_fish_density()
        else:
            # Select a new destination
            self._destination = self.exploration_strategy.choose_destination(
                current_position=xy2ij(*self.pos),
                success_locs=self.success_locs,
                failure_locs=self.failure_locs,
                other_agent_locs=self.other_agent_locs,
            )
            self._is_moving = True
            # self._destination = ij2xy(*self._destination)
            self.calculate_step_size()

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
        
        other_agents = self.model.agents
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

    def calculate_step_size(self):
        # Calculate euclidean distance to destination using scipy pdist
        step_size = pdist(np.array([self.pos, self._destination]), "euclidean")[0]
        self._step_sizes.append(step_size)
        self._traveled_distance_euclidean += step_size
        self._traveled_distance_manhattan += pdist(np.array([self.pos, self._destination]), "cityblock")[0]
