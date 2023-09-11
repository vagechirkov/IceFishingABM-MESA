from typing import Any, Union

import mesa
import numpy as np


class Agent(mesa.Agent):
    def __init__(self, unique_id, model, sampling_length: int = 10):
        super().__init__(unique_id, model)

        # set parameters
        self.sampling_length: int = sampling_length

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

            if current_observation < 0.33:
                # select random destination
                self.destination = self.random.choice(self.model.grid.get_neighborhood(self.pos, moore=True))
                self.is_moving = True
            elif current_observation < 0.66:
                self.destination = self.random.choice(self.model.grid.get_neighborhood(self.pos, moore=True))
                self.is_moving = True
            else:
                self.is_sampling = True

        if self.is_moving and self.is_sampling:
            raise ValueError("Agent is both sampling and moving.")

    def step(self):
        self.choose_next_action()