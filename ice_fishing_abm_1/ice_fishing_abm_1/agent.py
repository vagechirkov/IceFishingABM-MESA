from typing import Union

import mesa
import numpy as np

from .utils import discount_by_distance, find_peak, smooth_with_gaussian_filter


class Agent(mesa.Agent):
    def __init__(
            self,
            unique_id,
            model,
            sampling_length: int = 10,
            relocation_threshold: float = 0.7,
            local_search_counter: int = 4,
            meso_grid_step: int = 10,
            prior_knowledge: float = 0.05,
            alpha_social: float = 0.4,
            alpha_env: float = 0.4,
            visualization: bool = False,
    ):
        super().__init__(unique_id, model)
        # parameters
        # ---- sampling parameters ----
        self.sampling_length: int = sampling_length

        # ---- local search parameters ----
        self.relocation_threshold: float = relocation_threshold
        self.local_search_counter: int = local_search_counter

        # ---- belief parameters ----
        self.meso_grid_step: int = meso_grid_step  # size of the meso-scale grid cells
        assert self.model.grid.width % self.meso_grid_step == 0, \
            'grid width must be divisible by meso scale grid step'
        assert self.model.grid.height % self.meso_grid_step == 0, \
            'grid height must be divisible by meso scale grid step'
        self.meso_grid_width: int = self.model.grid.width // self.meso_grid_step
        self.meso_grid_height: int = self.model.grid.height // self.meso_grid_step
        self.prior_knowledge: float = prior_knowledge

        # ---- global displacement parameters ----
        self.alpha_social: float = alpha_social  # how much does social information influence the agent relocation?
        self.alpha_env: float = alpha_env  # how much does environmental information influence the agent relocation?
        # how much does random exploration influence the agent relocation?
        self.alpha_random = 1 - self.alpha_social - self.alpha_env if self.alpha_social + self.alpha_env < 1 else 0
        # normalize the sum of the parameters to 1
        self.alpha_social /= self.alpha_social + self.alpha_env + self.alpha_random
        self.alpha_env /= self.alpha_social + self.alpha_env + self.alpha_random
        self.alpha_random /= self.alpha_social + self.alpha_env + self.alpha_random

        # ---- debug parameters ----
        self.visualization: bool = visualization

        # states
        # ---- movement-related states ----
        self.is_moving: bool = False
        self.destination: Union[None, tuple] = None

        # ---- sampling-related states ----
        self.is_sampling: bool = False
        self.sampling_sequence: list[int, ...] = []
        self.observations: np.ndarray = np.ndarray(shape=(model.grid.width, model.grid.height), dtype=float)
        self.observations.fill(1e-6)
        self.local_search_count: int = 0
        self.collected_resource: int = 0

        # ---- meso-scale beliefs ----
        self.meso_belief: np.ndarray = np.zeros(shape=(self.meso_grid_width, self.meso_grid_height), dtype=float)
        self.meso_env_belief: np.ndarray = np.zeros(shape=(self.meso_grid_width, self.meso_grid_height), dtype=float)
        self.meso_soc_density: np.ndarray = np.zeros(shape=(self.meso_grid_width, self.meso_grid_height), dtype=float)
        self.meso_rand_array: np.ndarray = np.zeros(shape=(self.meso_grid_width, self.meso_grid_height), dtype=float)

        self.update_observations_with_prior_knowledge()

    def update_observations_with_prior_knowledge(self):
        self.meso_env_belief.fill(self.prior_knowledge)

    def move(self):
        """
        Move agent one cell closer to the destination
        """
        x, y = self.pos
        dx, dy = self.destination
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
        if self.pos == self.destination:
            self.is_moving = False
            # start sampling
            self.is_sampling = True

    def sample(self):
        """
        Sample the resource at the current location
        """
        x, y = self.pos

        if self.model.random.random() < self.model.resource_distribution[x, y]:
            self.collected_resource += 1
            self.sampling_sequence.append(1)
        else:
            self.sampling_sequence.append(0)

        # finish sampling and update observations
        if len(self.sampling_sequence) == self.sampling_length:
            self.update_local_observations()

            # finish sampling
            self.is_sampling = False

            # reset sampling sequence
            self.sampling_sequence = []

            # update local search counter
            self.local_search_count += 1

    def update_local_observations(self):
        """
        Update the agent's observations with the current resource distribution.
        """
        x, y = self.pos

        # replace previous observation with the new observation
        # TODO: implement a proper running average
        self.observations[x, y] = np.mean([np.mean(self.sampling_sequence), self.observations[x, y]])

    def update_meso_social_density(self):
        # get neighboring agents
        other_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False,
                                                     radius=self.model.grid.width)
        # get discounted smoothed locations map of other agents
        # TODO: use the observation history with the discount factor to incorporate the temporal aspect
        self.meso_soc_density.fill(0)
        for agent in other_agents:
            if agent.is_sampling:
                x, y = agent.pos
                self.meso_soc_density[x // self.meso_grid_step, y // self.meso_grid_step] += 1

        # normalize to make the sum equal to 1
        # TODO: add information about social density on the previous step
        self.meso_soc_density /= np.sum(self.meso_soc_density) + 1e-6

    def update_meso_environmental_belief(self):
        x_slice, y_slice, meso_x, meso_y = self._micro_slice_from_meso()
        max_obs = np.max(self.observations[x_slice, y_slice])

        # skip if the maximum observation < 1e-6
        if max_obs < 0.1:
            return

        # TODO: add information about the previous observation with the discount factor
        # np.mean([max_obs, self.meso_env_belief[meso_x, meso_y]])
        self.meso_env_belief[meso_x, meso_y] += max_obs / (np.sum(self.meso_env_belief) + 1e-6)

        # normalize
        self.meso_env_belief /= np.sum(self.meso_env_belief) + 1e-6

    def update_meso_random_preference(self):
        # generate random values for the preferences
        self.meso_rand_array = np.random.random(size=(self.meso_grid_width, self.meso_grid_height))

        # normalize
        self.meso_rand_array /= np.sum(self.meso_rand_array) + 1e-6

    def update_meso_beliefs(self):
        self.update_meso_social_density()
        self.update_meso_environmental_belief()
        self.update_meso_random_preference()

        # combine beliefs
        self.meso_belief = self.alpha_social * self.meso_soc_density + \
                           self.alpha_env * self.meso_env_belief + \
                           self.alpha_random * self.meso_rand_array

        _, _, meso_x, meso_y = self._micro_slice_from_meso()

        # discount the belief with the distance from the current location
        self.meso_belief = discount_by_distance(self.meso_belief, (meso_x, meso_y), discount_factor=0.5)

        # normalize
        self.meso_belief /= np.sum(self.meso_belief) + 1e-6

    def global_displacement(self):
        self.update_meso_beliefs()

        # get the peak of the relocation map and find the closest empty cell to the destination
        dy, dx = find_peak(self.meso_belief)
        rand_x = dx * self.meso_grid_step + self.model.random.randint(0, self.meso_grid_step - 1)
        rand_y = dy * self.meso_grid_step + self.model.random.randint(0, self.meso_grid_step - 1)
        self.destination = self._closest_empty_cell((rand_x, rand_y))

        # reset local search counter
        self.local_search_count = 0

    def local_displacement(self):
        # select random location within the meso-scale grid cell
        _, _, meso_x, meso_y = self._micro_slice_from_meso()
        x_rand = self.model.random.randint(meso_x * self.meso_grid_step, (meso_x + 1) * self.meso_grid_step - 1)
        y_rand = self.model.random.randint(meso_y * self.meso_grid_step, (meso_y + 1) * self.meso_grid_step - 1)
        self.destination = self._closest_empty_cell((x_rand, y_rand))

    def choose_next_action(self):
        """
        Choose the next action for the agent.
        """
        if self.is_moving and not self.is_sampling:
            self.move()
            return

        if self.is_sampling and not self.is_moving:
            self.sample()

        if not self.is_sampling and not self.is_moving:  # this is also the case when the agent is initialized
            counter_done = self.local_search_count >= self.local_search_counter
            good_spot_found = self.observations[self.pos] > self.relocation_threshold

            if good_spot_found:
                self.is_sampling = True
                self.sample()
            elif not counter_done:
                self.local_displacement()
                self.is_moving = True
            else:
                self.global_displacement()
                self.is_moving = True
                if self.visualization:
                    self.debug_plot()

        if self.is_moving and self.is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def debug_plot(self):
        if self.unique_id == 1:
            self.model.agent_raw_observations = self.observations

            kron_shape = (self.meso_grid_step, self.meso_grid_step)
            d_type = self.meso_belief.dtype

            self.model.relocation_map = np.kron(self.meso_belief, np.ones(kron_shape, dtype=d_type))
            self.model.agent_raw_soc_observations = np.kron(self.meso_soc_density, np.ones(kron_shape, dtype=d_type))
            self.model.agent_raw_env_belief = np.kron(self.meso_env_belief, np.ones(kron_shape, dtype=d_type))
            self.model.agent_raw_rand_array = np.kron(self.meso_rand_array, np.ones(kron_shape, dtype=d_type))

    def step(self):
        self.choose_next_action()

    def _micro_slice_from_meso(self):
        # find max observation in the micro-scale grid
        x, y = self.pos
        meso_x = x // self.meso_grid_step
        meso_y = y // self.meso_grid_step
        x_slice = slice(meso_x * self.meso_grid_step, (meso_x + 1) * self.meso_grid_step)
        y_slice = slice(meso_y * self.meso_grid_step, (meso_y + 1) * self.meso_grid_step)

        return x_slice, y_slice, meso_x, meso_y

    def _closest_empty_cell(self, destination: tuple[int, int]):
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
