from typing import Union

import mesa
import numpy as np

from .utils import discount_by_distance, find_peak


class Agent(mesa.Agent):
    def __init__(
            self,
            unique_id,
            model,
            sampling_length: int = 10,
            relocation_threshold: float = 0.7,
            local_search_counter: int = 4,
            prior_knowledge_corr: float = 0.5,
            prior_knowledge_noize: float = 0.3,
            w_social: float = 0.4,
            w_personal: float = 0.4,
    ):
        super().__init__(unique_id, model)
        # parameters
        # ---- sampling parameters ----
        self.sampling_length: int = sampling_length

        # ---- local search parameters ----
        self.relocation_threshold: float = relocation_threshold
        self.local_search_counter: int = local_search_counter

        # ---- prior knowledge parameters ----
        self.prior_knowledge_corr: float = prior_knowledge_corr
        self.prior_knowledge_noize: float = prior_knowledge_noize

        # ---- global displacement weights ----
        self.w_soc, self.w_env, self.w_rand = self._information_weights(w_social, w_personal)

        # states
        # ---- movement-related states ----
        self._is_moving: bool = False
        self._destination: Union[None, tuple] = None

        # ---- sampling-related states ----
        self._is_sampling: bool = False
        self._sampling_sequence: list[int, ...] = []
        self._observations: np.ndarray = np.ndarray(shape=(model.grid.width, model.grid.height), dtype=float)
        self._observations.fill(1e-6)
        self._local_search_count: int = 0
        self._collected_resource: int = 0

        # ---- meso-scale information ----
        shape = (self.model.grid.meso_width, self.model.grid.meso_height)
        self._meso_env: np.ndarray = np.zeros(shape=shape, dtype=float)
        self._meso_soc: np.ndarray = np.zeros(shape=shape, dtype=float)
        self._meso_combined: np.ndarray = np.zeros(shape=shape, dtype=float)

        self.update_observations_with_prior_knowledge()

    @property
    def meso_pos(self):
        return self.model.grid.meso_coordinate(*self.pos)

    @property
    def meso_soc(self):
        return np.kron(self._meso_soc, np.ones((self.model.grid.meso_step, self.model.grid.meso_step)))

    @property
    def meso_env(self):
        return np.kron(self._meso_env, np.ones((self.model.grid.meso_step, self.model.grid.meso_step)))

    @property
    def meso_combined(self):
        return np.kron(self._meso_combined, np.ones((self.model.grid.meso_step, self.model.grid.meso_step)))

    @property
    def is_moving(self):
        return self._is_moving

    @property
    def is_sampling(self):
        return self._is_sampling

    def update_observations_with_prior_knowledge(self):
        # TODO: implement
        pass
        # self._meso_env.fill(self.prior_knowledge)

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
        """
        Sample the resource at the current location
        """
        x, y = self.pos

        if self.model.random.random() < self.model.resource_distribution[x, y]:
            self._collected_resource += 1
            self._sampling_sequence.append(1)
        else:
            self._sampling_sequence.append(0)

        # finish sampling and update observations
        if len(self._sampling_sequence) == self.sampling_length:
            self.update_local_observations()

            # finish sampling
            self._is_sampling = False

            # reset sampling sequence
            self._sampling_sequence = []

            # update local search counter
            self._local_search_count += 1

    def update_local_observations(self):
        """
        Update the agent's observations with the current resource distribution.
        """
        x, y = self.pos

        # replace previous observation with the new observation
        # TODO: implement a proper running average
        self._observations[x, y] = np.mean([np.mean(self._sampling_sequence), self._observations[x, y]])

    def update_meso_social_density(self):
        # get neighboring agents
        other_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False,
                                                     radius=self.model.grid.width)
        # get discounted smoothed locations map of other agents
        # TODO: use the observation history with the discount factor to incorporate the temporal aspect
        self._meso_soc.fill(0)
        for agent in other_agents:
            if agent.is_sampling:
                self._meso_soc[*agent.meso_pos] += 1

        # normalize to make the sum equal to 1
        # TODO: add information about social density on the previous step
        self._meso_soc = self.normalize(self._meso_soc)

    def update_meso_environmental_belief(self):
        meso_x, meso_y = self.meso_pos
        max_obs = np.max(self._observations[*self.model.grid.micro_slice_from_meso(meso_x, meso_y)])

        # TODO: add information about the previous observation with the discount factor
        self._meso_env[meso_x, meso_y] = np.mean([max_obs, self._meso_env[meso_x, meso_y]])
        return self.normalize(self._meso_env)

    def generate_random_preference(self):
        # generate random values for the preferences
        rand_array = np.random.random(size=(self.model.grid.meso_width, self.model.grid.meso_height))

        # normalize
        return self.normalize(rand_array)

    def update_meso_beliefs(self):
        self.update_meso_social_density()
        env_info = self.update_meso_environmental_belief()
        random_preference = self.generate_random_preference()

        # combine beliefs
        belief = self.w_soc * self._meso_soc + self.w_env * env_info + self.w_rand * random_preference

        # discount the belief with the distance from the current location
        belief_discounted = discount_by_distance(belief, self.meso_pos, discount_factor=0.5)

        self._meso_combined = self.normalize(belief_discounted)

    def global_displacement(self):
        self.update_meso_beliefs()

        # get the peak of the relocation map and find the closest empty cell to the destination
        dy, dx = find_peak(self._meso_combined)
        self._destination = self._get_random_micro_cell_in_meso_grid(dx, dy)

        # reset local search counter
        self._local_search_count = 0

    def local_displacement(self):
        # select random location within the meso-scale grid cell
        self._destination = self._get_random_micro_cell_in_meso_grid(*self.meso_pos)

    def choose_next_action(self):
        """
        Choose the next action for the agent.
        """
        if self._is_moving and not self._is_sampling:
            self.move()
            return

        if self._is_sampling and not self._is_moving:
            self.sample()

        if not self._is_sampling and not self._is_moving:  # this is also the case when the agent is initialized
            counter_done = self._local_search_count >= self.local_search_counter
            good_spot_found = self._observations[self.pos] > self.relocation_threshold

            if good_spot_found:
                self._is_sampling = True
                self.sample()
            elif not counter_done:
                self.local_displacement()
                self._is_moving = True
            else:
                self.global_displacement()
                self._is_moving = True

        if self._is_moving and self._is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def step(self):
        self.choose_next_action()

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

    def _get_random_micro_cell_in_meso_grid(self, meso_x: int, meso_y: int):
        slice_x, slice_y = self.model.grid.micro_slice_from_meso(meso_x, meso_y)
        x = self.model.random.randint(slice_x.start, slice_x.stop - 1)
        y = self.model.random.randint(slice_y.start, slice_y.stop - 1)
        return self._closest_empty_cell((x, y))

    @staticmethod
    def _information_weights(w_soc: float, w_env: float) -> tuple[float, float, float]:
        """
        Calculate the weights of the social and environmental information for the current step.

        :param w_soc: The weight of the social information.
        :param w_env: The weight of the environmental information.
        :return: The weights of the social and environmental information.
        """
        assert -1 <= w_soc <= 1, "The weight of the social information must be between -1 and 1."
        assert 0 <= w_env <= 1, "The weight of the environmental information must be between 0 and 1."

        w_rand = 1 - w_soc - w_env if w_soc + w_env < 1 else 0
        sum_weights = w_soc + w_env + w_rand
        w_soc, w_env, w_rand = w_soc / sum_weights, w_env / sum_weights, w_rand / sum_weights

        return w_soc, w_env, w_rand

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        return arr / (np.sum(arr) + 1e-6)
