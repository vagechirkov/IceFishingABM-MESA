from typing import Any, Union

import mesa
import numpy as np

from .social_information import estimate_social_vector


class Agent(mesa.Agent):
    def __init__(self,
                 unique_id,
                 model,
                 sampling_length: int = 10,
                 relocation_threshold: float = 0.5,
                 social_influence_threshold: float = 1,
                 exploration_threshold: float = 0.01):
        super().__init__(unique_id, model)

        # set parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.social_influence_threshold: float = social_influence_threshold  # magnitude of social vector
        self.exploration_threshold: float = exploration_threshold  # choose a random destination with this probability

        # movement-related states
        self.is_moving: bool = False
        self.destination: Union[None, tuple] = None

        # sampling-related states
        self.is_sampling: bool = False
        self.sampling_sequence: list[int, ...] = []
        self.observations: np.ndarray = np.ndarray(shape=(model.grid.width, model.grid.height), dtype=float)
        self.collected_resource: int = 0

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
            self.destination = (dx, dy)
        else:
            self.random_relocate()

        self.is_moving = True

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
                    self.relocate()
                else:
                    self.is_sampling = True

        if self.is_moving and self.is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def step(self):
        self.choose_next_action()
