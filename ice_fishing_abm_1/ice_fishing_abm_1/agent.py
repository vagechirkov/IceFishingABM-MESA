from typing import Union

import mesa
import numpy as np

from .resource import Resource
from .utils import discount_by_distance, find_peak


class Agent(mesa.Agent):
    def __init__(
            self,
            unique_id,
            model,
            sampling_length: int = 10,
            resource_cluster_radius: int = 5,
            relocation_threshold: float = 0.7,
            local_search_counter: int = 4,
            local_learning_rate: float = 0.5,
            meso_learning_rate: float = 0.5,
            prior_knowledge_corr: float = 0.5,
            prior_knowledge_noize: float = 0.3,
            w_social: float = 0.4,
            w_personal: float = 0.4,
    ):
        super().__init__(unique_id, model)
        # parameters
        # ---- sampling parameters ----
        self.sampling_length: int = sampling_length
        self.resource_cluster_radius: int = resource_cluster_radius

        # ---- local search parameters ----
        self.relocation_threshold: float = relocation_threshold
        self.local_search_counter: int = local_search_counter

        # ---- learning parameters ----
        self.local_learning_rate: float = local_learning_rate
        self.meso_learning_rate: float = meso_learning_rate

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
        self._array_observations: np.ndarray = np.ndarray(shape=(model.grid.height, model.grid.width), dtype=float)
        self._array_observations.fill(0.1)
        self._local_search_count: int = 0
        self._collected_resource: int = 0

        # ---- meso-scale information ----
        shape = (self.model.grid.meso_width, self.model.grid.meso_height)
        self._array_meso_env: np.ndarray = np.zeros(shape=shape, dtype=float)
        self._array_meso_soc: np.ndarray = np.zeros(shape=shape, dtype=float)
        self._array_meso_combined: np.ndarray = np.zeros(shape=shape, dtype=float)

        self.update_observations_with_prior_knowledge()

    @property
    def meso_pos(self):
        return self.model.grid.meso_coordinate(*self.pos)

    @property
    def meso_soc(self):
        return np.kron(self._array_meso_soc,
                       np.ones((self.model.grid.meso_scale_step, self.model.grid.meso_scale_step)))

    @property
    def meso_env(self):
        return np.kron(self._array_meso_env,
                       np.ones((self.model.grid.meso_scale_step, self.model.grid.meso_scale_step)))

    @property
    def meso_combined(self):
        return np.kron(self._array_meso_combined,
                       np.ones((self.model.grid.meso_scale_step, self.model.grid.meso_scale_step)))

    @property
    def observations(self):
        return self._array_observations.copy()

    @property
    def is_moving(self):
        return self._is_moving

    @property
    def is_sampling(self):
        return self._is_sampling

    def update_observations_with_prior_knowledge(self):
        if self.prior_knowledge_corr == 0:
            # flat prior knowledge
            self._array_meso_env.fill(0.1)
        else:
            # mean-pooling resource distribution
            # SEE:https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
            # Getting shape of matrix
            M, N = self.model.resource_distribution.shape

            # Shape of kernel
            K = self.model.grid.meso_scale_step
            L = self.model.grid.meso_scale_step

            # Dividing the image size by kernel size
            MK = M // K
            NL = N // L

            # Creating a pool
            meso_resource = self.model.resource_distribution[:MK * K, :NL * L].reshape(MK, K, NL, L).mean(axis=(1, 3))

            # add noise
            meso_resource += np.random.random(size=meso_resource.shape) * self.prior_knowledge_noize

            self._array_meso_env = meso_resource

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
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False,
                                                  radius=self.resource_cluster_radius)

        # check if there are any resources in the neighborhood
        if len(neighbors) > 0:
            for neighbour in neighbors:
                for agent in self.model.grid.get_cell_list_contents([neighbour]):
                    if isinstance(agent, Resource):
                        if agent.catch():
                            self._collected_resource += 1
                            self._sampling_sequence.append(1)
                            break

        # finish sampling and update observations
        if len(self._sampling_sequence) == self.sampling_length:
            self.update_local_observations()
            self.update_meso_environmental_belief()

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
        i, j = self.model.grid.x_y_to_i_j(*self.pos)

        # update observations
        old_observation = self._array_observations[i, j]
        new_observation = np.mean(self._sampling_sequence)
        time_difference = new_observation - old_observation
        self._array_observations[i, j] = old_observation + self.local_learning_rate * time_difference

    def update_meso_social_density(self):
        # get neighboring agents
        other_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False,
                                                     radius=self.model.grid.width)
        # filter out resources
        other_agents = [agent for agent in other_agents if isinstance(agent, Agent)]

        # get discounted smoothed locations map of other agents
        # TODO: use the observation history with the discount factor to incorporate the temporal aspect
        self._array_meso_soc.fill(0)
        for agent in other_agents:
            if agent.is_sampling:
                i, j = self.model.grid.x_y_to_i_j(*agent.meso_pos)
                self._array_meso_soc[i, j] += 1

        # normalize to make the sum equal to 1
        # TODO: add information about social density on the previous step
        self._array_meso_soc = self.normalize(self._array_meso_soc)

    def update_meso_environmental_belief(self):
        meso_i, meso_j = self.model.grid.x_y_to_i_j(*self.meso_pos)
        i, j = self.model.grid.x_y_to_i_j(*self.pos)

        # update observations
        old_observation = self._array_meso_env[meso_i, meso_j]
        new_observation = self._array_observations[i, j]
        time_difference = new_observation - old_observation
        self._array_meso_env[meso_i, meso_j] = old_observation + self.meso_learning_rate * time_difference

    def generate_random_preference(self):
        # generate random values for the preferences
        rand_array = np.random.random(size=(self.model.grid.meso_height, self.model.grid.meso_width))

        # normalize
        return self.normalize(rand_array)

    def update_meso_beliefs(self):
        self.update_meso_social_density()
        env_info = self.normalize(self._array_meso_env)
        random_preference = self.generate_random_preference()

        # combine beliefs
        belief = self.w_soc * self._array_meso_soc + self.w_env * env_info + self.w_rand * random_preference

        # discount the belief with the distance from the current location
        meso_i, meso_j = self.model.grid.x_y_to_i_j(*self.meso_pos)
        belief_discounted = discount_by_distance(belief, (meso_i, meso_j), discount_factor=0.8)

        self._array_meso_combined = self.normalize(belief_discounted)

    def global_displacement(self):
        # get the peak of the relocation map and find the closest empty cell to the destination
        di, dj = find_peak(self._array_meso_combined)
        dx, dy = self.model.grid.x_y_to_i_j(di, dj)
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
            # if the first step
            if self.model.schedule.steps == 0:
                self.update_meso_beliefs()

            counter_done = self._local_search_count >= self.local_search_counter
            i, j = self.model.grid.x_y_to_i_j(*self.pos)
            good_spot_found = self._array_observations[i, j] > self.relocation_threshold

            if good_spot_found:
                self._is_sampling = True
                # self.sample()
            elif not counter_done:
                self.local_displacement()
                self._is_moving = True
            else:
                self.update_meso_beliefs()
                self.global_displacement()
                self._is_moving = True

        if self._is_moving and self._is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def step(self):
        self.choose_next_action()

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

    def _get_random_micro_cell_in_meso_grid(self, meso_x: int, meso_y: int):
        # TODO: check if it is okay if x and y are switched
        slice_x, slice_y = self.model.grid.micro_slice_from_meso_coordinate(meso_x, meso_y)
        x = self.model.random.randint(slice_x.start + 1, slice_x.stop - 1)
        y = self.model.random.randint(slice_y.start + 1, slice_y.stop - 1)
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
