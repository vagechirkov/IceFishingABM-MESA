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
            social_influence_threshold: float = 1,
            exploration_threshold: float = 0.01,
            prior_knowledge: float = 0.05,
            visualization: bool = False,
            vision_range: int = 1000,
            soc_influence_ratio: float = 0.5
    ):
        super().__init__(unique_id, model)

        # set parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.social_influence_threshold: float = social_influence_threshold  # magnitude of social vector
        self.exploration_threshold: float = exploration_threshold  # choose a random destination with this probability
        self.prior_knowledge: float = prior_knowledge  # prior knowledge about the resource distribution
        self.visualization: bool = visualization
        self.vision_range: int = vision_range  # how far can the agent see other agents
        # how much does social information influence the agent relocation compared to environmental information?
        self.soc_influence_ratio: float = soc_influence_ratio
        self.soc_observations: np.ndarray = np.ndarray(shape=(model.grid.width, model.grid.height), dtype=float)
        # fill observations with small value
        self.soc_observations.fill(1e-6)

        # movement-related states
        self.is_moving: bool = False
        self.destination: Union[None, tuple] = None

        # sampling-related states
        self.is_sampling: bool = False
        self.sampling_sequence: list[int, ...] = []
        self.observations: np.ndarray = np.ndarray(shape=(model.grid.width, model.grid.height), dtype=float)
        # fill observations with small value
        self.observations.fill(1e-6)
        self.collected_resource: int = 0

        self.update_observations_with_prior_knowledge()

    def update_observations_with_prior_knowledge(self):
        if self.prior_knowledge == 0:
            return

        # create a list of all cell indices tuples
        inx = [(x, y) for x in range(self.model.grid.width) for y in range(self.model.grid.height)]

        # select random cells to observe
        inx = self.random.sample(inx, k=int(self.prior_knowledge * len(inx)))
        # inx = np.unravel_index(
        #     np.argsort(self.model.resource_distribution, axis=None)[-self.prior_knowledge:],
        #     self.model.resource_distribution.shape)
        # self.observations[inx] = self.model.resource_distribution[inx]

        for i in inx:
            self.observations[i] = self.model.resource_distribution[i]

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
            self.update_observations()
            self.update_social_observations()
            self.is_sampling = False

    def update_observations(self):
        """
        Update the agent's observations with the current resource distribution.
        """
        x, y = self.pos

        # replace previous observation with the new observation
        # NOTE: here we are assuming that agent completely forgets previous observations in the cell
        self.observations[x, y] = np.mean(self.sampling_sequence)

        # reset sampling sequence
        self.sampling_sequence = []
        self.is_sampling = False

    def update_social_observations(self):
        # get neighboring agents
        other_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False,
                                                     radius=self.vision_range)
        # get discounted smoothed locations map of other agents
        # TODO: use the observation history with the discount factor to incorporate the temporal aspect
        self.soc_observations.fill(0)
        for agent in other_agents:
            if agent.is_sampling:
                self.soc_observations[agent.pos] = 1

    def relocate(self):
        # get discounted smoothed personal observations
        discounted_obs = discount_by_distance(self.observations, self.pos, discount_factor=0.5)
        smoothed_obs = smooth_with_gaussian_filter(discounted_obs, sigma=2)
        smoothed_obs /= np.max(smoothed_obs)

        # get discounted smoothed social observations
        discounted_soc_obs = discount_by_distance(self.soc_observations, self.pos, discount_factor=0.5)
        smoothed_soc_obs = smooth_with_gaussian_filter(discounted_soc_obs, sigma=2)
        smoothed_soc_obs /= np.max(smoothed_soc_obs)

        # combine personal observations and social information
        relocation_map = smoothed_obs * (1 - self.soc_influence_ratio) + smoothed_soc_obs * self.soc_influence_ratio

        # get the peak of the relocation map and find the closest empty cell to the destination
        dx, dy = find_peak(relocation_map)
        self.destination = self.closest_empty_cell((dx, dy))
        self.is_moving = True

    def closest_empty_cell(self, destination: tuple[int, int]):
        radius = 1
        empty_cells = self.get_empty_cells(destination, radius)

        # increase the radius until an empty cell is found
        while len(empty_cells) == 0:
            radius += 1
            empty_cells = self.get_empty_cells(destination, radius=radius)
        return self.random.choice(empty_cells)

    def get_empty_cells(self, destination, radius=1):
        neighbors = self.model.grid.get_neighborhood(destination, moore=True, include_center=True, radius=radius)
        return [cell for cell in neighbors if self.model.grid.is_cell_empty(cell)]

    def random_relocate(self, radius: int = 20):
        """
        Choose a random destination and start moving.
        """
        self.destination = self.random.choice(
            self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=radius))
        self.is_moving = True

    def choose_next_action(self):
        """
        Choose the next action for the agent.
        """
        if self.is_moving and not self.is_sampling:
            self.move()

        if self.is_sampling and not self.is_moving:
            self.sample()

        if not self.is_moving and not self.is_sampling:
            # choose whether and where to move or sample
            x, y = self.pos
            current_observation = self.observations[x, y]

            if current_observation < self.relocation_threshold:
                if self.visualization:
                    self.debug_plot()
                if self.model.random.random() < self.exploration_threshold:
                    if current_observation > 0.3:
                        self.random_relocate(radius=3)
                    else:
                        self.random_relocate(radius=20)
                else:
                    self.relocate()
            else:
                self.is_sampling = True

        if self.is_moving and self.is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def debug_plot(self):
        if self.unique_id == 1:
            # get discounted smoothed personal observations
            self.model.agent_raw_observations = self.observations
            discounted_obs = discount_by_distance(self.observations, self.pos, discount_factor=0.5)
            self.model.agent_discounted_observations = discounted_obs / np.max(discounted_obs)
            smoothed_obs = smooth_with_gaussian_filter(discounted_obs, sigma=2)
            smoothed_obs /= np.max(smoothed_obs)
            self.model.agent_smoothed_observations = smoothed_obs

            # get discounted smoothed social observations
            self.model.agent_raw_soc_observations = self.soc_observations
            discounted_soc_obs = discount_by_distance(self.soc_observations, self.pos, discount_factor=0.5)
            self.model.agent_discounted_soc_observations = discounted_soc_obs / np.max(discounted_soc_obs)
            smoothed_soc_obs = smooth_with_gaussian_filter(discounted_soc_obs, sigma=2)
            smoothed_soc_obs /= np.max(smoothed_soc_obs)
            self.model.agent_smoothed_soc_observations = smoothed_soc_obs

            # combine personal observations and social information
            relocation_map = smoothed_obs * (1 - self.soc_influence_ratio) + smoothed_soc_obs * self.soc_influence_ratio
            self.model.relocation_map = relocation_map

    def step(self):
        self.choose_next_action()
