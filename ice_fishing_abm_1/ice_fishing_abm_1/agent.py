from typing import Union

import mesa
import numpy as np

from .environmental_information import discount_observations_by_distance, estimate_environment_peak, \
    smooth_observations_with_gaussian_filter
from .social_information import estimate_social_vector


class Agent(mesa.Agent):
    def __init__(self,
                 unique_id,
                 model,
                 sampling_length: int = 10,
                 relocation_threshold: float = 0.7,
                 social_influence_threshold: float = 1,
                 exploration_threshold: float = 0.01,
                 prior_knowledge: float = 0.05):
        super().__init__(unique_id, model)

        # set parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.social_influence_threshold: float = social_influence_threshold  # magnitude of social vector
        self.exploration_threshold: float = exploration_threshold  # choose a random destination with this probability
        self.prior_knowledge: float = prior_knowledge  # prior knowledge about the resource distribution

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

    def relocate(self):
        # get neighboring agents
        other_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=20)

        # estimate social vector
        social_vector = estimate_social_vector(self.pos, [agent.pos for agent in other_agents])

        if np.linalg.norm(social_vector) >= self.social_influence_threshold:
            # choose a destination that is correlated with social vector
            x, y = self.pos
            dx = x + int(np.round(social_vector[0])) * self.random.randint(1, 3)
            dy = y + int(np.round(social_vector[1])) * self.random.randint(1, 3)
        else:
            # estimate environmental vector
            discounted_obs = discount_observations_by_distance(self.observations.copy(), self.pos, discount_factor=0.5)
            smoothed_obs = smooth_observations_with_gaussian_filter(discounted_obs, sigma=2)
            environmental_peak = estimate_environment_peak(self.pos, smoothed_obs)

            # select destination based on the environmental peak
            dx, dy = environmental_peak

        # find the closest empty cell to the destination
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

    def random_relocate(self):
        """
        Choose a random destination and start moving.
        """
        self.destination = self.random.choice(
            self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=20))
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
            if self.model.random.random() < self.exploration_threshold:
                self.random_relocate()
            else:
                # choose whether and where to move or sample
                x, y = self.pos
                current_observation = self.observations[x, y]

                if current_observation < self.relocation_threshold:
                    self.debug_plot()
                    self.relocate()
                else:
                    self.is_sampling = True

        if self.is_moving and self.is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def debug_plot(self):
        if self.unique_id == 1:
            self.model.agent_raw_observations = self.observations
            discounted_obs = discount_observations_by_distance(self.observations.copy(), self.pos, discount_factor=0.5)
            self.model.agent_discounted_observations = discounted_obs / np.max(discounted_obs)
            smoothed_obs = smooth_observations_with_gaussian_filter(discounted_obs, sigma=2)
            self.model.agent_smoothed_observations = smoothed_obs / np.max(smoothed_obs)

    def step(self):
        self.choose_next_action()
