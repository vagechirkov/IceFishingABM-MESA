from typing import Union

import mesa
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from .resource import Resource
from .belief import construct_dataset_info, generate_belief_matrix
from .utils import x_y_to_i_j, find_peak


class Agent(mesa.Agent):
    def __init__(self, unique_id, model, resource_cluster_radius, agent_type='greedy'):
        super().__init__(unique_id, model)

        # states
        # ---- movement-related states ----
        self._is_moving: bool = False
        self._destination: Union[None, tuple] = None

        # ---- sampling-related states ----
        self._is_sampling: bool = False
        self._time_since_last_catch: int = 0
        self._collected_resource_last_spot: int = 0
        self._collected_resource: int = 0
        self.resource_cluster_radius = resource_cluster_radius
        self.catch_wait_time = 5

        # ---- belief values ----
        self.agent_type = agent_type
        self.other_agent_locs = np.empty((0, 2))
        self.social_gpc = GaussianProcessClassifier(kernel=RBF(12), random_state=0, optimizer=None)
        self.social_feature = np.zeros((self.model.grid_size, self.model.grid_size))
        self.success_locs = np.empty((0, 2))
        self.success_gpc = GaussianProcessClassifier(kernel=RBF(12), random_state=0, optimizer=None)
        self.success_feature = np.zeros((self.model.grid_size, self.model.grid_size))
        self.failure_locs = np.empty((0, 2))
        self.failure_gpc = GaussianProcessClassifier(kernel=RBF(12), random_state=0, optimizer=None)
        self.failure_feature = np.zeros((self.model.grid_size, self.model.grid_size))
        self.belief = np.zeros((self.model.grid_size, self.model.grid_size))

    @property
    def is_moving(self):
        return self._is_moving

    @property
    def is_sampling(self):
        return self._is_sampling

    def step(self):
        if self._is_moving and not self._is_sampling:
            self._adjust_destination_if_cell_occupied()
            self.move()
            # update movement destination because social feature might have changed
            # self.movement_destination()

            return

        if self._is_sampling and not self._is_moving:
            self.sample()

        if not self._is_sampling and not self._is_moving:  # this is also the case when the agent is initialized
            # if the first step then just sample in the current position
            if self.model.schedule.steps == 0:
                self._is_sampling = True
                return

            # select a new destination of movement
            self.movement_destination()
            self._is_moving = True

        if self._is_moving and self._is_sampling:
            raise ValueError("Agent is both sampling and moving.")

        # update social information at the end of each step
        self.add_other_agent_locs()

    def move(self):
        """
        Move agent one cell closer to the destination
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

        # check if destination has been reached
        if self.pos == self._destination:
            self._is_moving = False
            # start sampling
            self._is_sampling = True

    def sample(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True,
                                                  radius=self.resource_cluster_radius)
        resource_collected = False

        # check if there are any resources in the neighborhood
        if len(neighbors) > 0:
            for neighbour in neighbors:
                if isinstance(neighbour, Resource):
                    if neighbour.catch():
                        # NB: sample multiple times if resources overlap
                        resource_collected = True

        if resource_collected:
            self._time_since_last_catch = 0
            self._collected_resource += 1
            self._collected_resource_last_spot += 1
            self.add_success_loc(self.pos)
        else:
            self._time_since_last_catch += 1

        if self._time_since_last_catch >= self.catch_wait_time:
            self._is_sampling = False
            self._time_since_last_catch = 0

            if self._collected_resource_last_spot == 0:
                self.add_failure_loc(self.pos)

            self._collected_resource_last_spot = 0

    def movement_destination(self):
        margin = 40
        step_size = 20
        if self.other_agent_locs.size == 0:
            self.social_feature = np.zeros((self.model.grid_size, self.model.grid_size))
        else:
            Xs, ys = construct_dataset_info(self.model.grid_size, margin, self.other_agent_locs, step_size)
            self.social_feature = generate_belief_matrix(self.model.grid_size, margin, Xs, ys, self.social_gpc, 1).T

        # recompute this features only if the agent is not moving
        if not self._is_moving:
            # success feature
            if self.success_locs.size == 0:
                self.success_feature = np.zeros((self.model.grid_size, self.model.grid_size))
            else:
                Xc, yc = construct_dataset_info(self.model.grid_size, margin, self.success_locs, step_size)
                self.success_feature = generate_belief_matrix(
                    self.model.grid_size, margin, Xc, yc, self.success_gpc, 1).T

            # failure feature
            if self.failure_locs.size == 0:
                self.failure_feature = np.zeros((self.model.grid_size, self.model.grid_size))
            else:
                Xf, yf = construct_dataset_info(self.model.grid_size, margin, self.failure_locs, step_size)
                self.failure_feature = generate_belief_matrix(
                    self.model.grid_size, margin, Xf, yf, self.failure_gpc, 0).T

        # combine features
        self.belief = (self.model.w_social * self.social_feature +
                       self.model.w_success * self.success_feature +
                       self.model.w_failure * self.failure_feature)

        # Add spacial discounting
        # TODO: add spacial discounting

        # find the next destination as a random sample from the belief using belief as a probability distribution
        self._destination = x_y_to_i_j(*find_peak(self.belief))

    def _adjust_destination_if_cell_occupied(self):
        if self._destination is None:
            return

        if self.model.grid.is_cell_empty(self._destination):
            return

        # get cell inhabitants
        inhabitants = self.model.grid.get_cell_list_contents(self._destination)

        agents = [agent for agent in inhabitants if isinstance(agent, Agent)]
        if any([agent.is_sampling for agent in agents]):
            self._destination = self._closest_empty_cell(self._destination)

    def _closest_empty_cell(self, destination: tuple[int, int]):
        # check if the destination is empty
        if self.model.grid.is_cell_empty(destination):
            return destination

        radius = 2
        empty_cells = self._get_empty_cells(destination, radius)

        # increase the radius until an empty cell is found
        while len(empty_cells) == 0:
            radius += 1
            empty_cells = self._get_empty_cells(destination, radius=radius)
        return self.random.choice(empty_cells)

    def _get_empty_cells(self, destination, radius=1):
        neighbors = self.model.grid.get_neighborhood(destination, moore=True, include_center=True, radius=radius)
        return [cell for cell in neighbors if self.model.grid.is_cell_empty(cell)]

    def add_success_loc(self, loc: tuple):
        self.success_locs = np.vstack([self.success_locs, np.array(x_y_to_i_j(*loc))[np.newaxis, :]])

    def add_failure_loc(self, loc: tuple):
        self.failure_locs = np.vstack([self.failure_locs, np.array(x_y_to_i_j(*loc))[np.newaxis, :]])

    def add_other_agent_locs(self):
        other_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False,
                                                     radius=self.model.grid.width)
        agents = np.array([agent for agent in other_agents if isinstance(agent, Agent)])

        # sampling agents
        agents = [np.array(x_y_to_i_j(*agent.pos))[np.newaxis, :] for agent in agents if agent.is_sampling]

        if len(agents) > 0:
            self.other_agent_locs = np.vstack(agents)
        else:
            self.other_agent_locs = np.empty((0, 2))
