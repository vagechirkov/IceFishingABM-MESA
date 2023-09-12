import logging

import mesa
import numpy as np
from mesa.space import MultiGrid

from .agent import Agent
from .resource_distribution import ResourceDistribution


class Model(mesa.Model):
    def __init__(self,
                 grid_width: int = 100,
                 grid_height: int = 100,
                 number_of_agents: int = 5):
        super().__init__()
        self.number_of_agents = number_of_agents
        self.current_id = 0
        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        # self.datacollector = mesa.datacollection.DataCollector(model_reporters={},)
        self.schedule = mesa.time.RandomActivation(self)
        self.resource = ResourceDistribution(self, n_samples=100_000, n_clusters=1)

        # resource distribution
        self.resource_distribution = self.resource.resource_distribution

        # add resource distribution to grid colors for visualization
        self.grid_colors = self.resource_distribution

        # Create agents
        for _ in range(self.number_of_agents):
            self.initialize_agent()

    def initialize_agent(self, radius: int = 1) -> None:
        """
        Create an agent and add it to the schedule and grid at a random empty cell in the center of the grid.

        :param radius: The distance from the center of the grid to place the agent.
        """
        assert radius > 0, "Radius must be greater than 0."
        a = Agent(self.next_id(), self)

        self.schedule.add(a)
        x_center, y_center = self.grid.width // 2, self.grid.height // 2
        n = self.grid.get_neighborhood((x_center, y_center), moore=True, include_center=True, radius=radius)

        # select empty cells
        empty_cells = [cell for cell in n if self.grid.is_cell_empty(cell)]

        if len(empty_cells) == 0:
            # print warning that multiple agents are being placed in the same cell
            logging.warning("Multiple agents are being placed in the same cell for initialization.")
            # select random cell (even if it is not empty)
            cell = self.random.choice(n)
        else:
            # select random empty cell
            cell = self.random.choice(empty_cells)

        # place agent
        self.grid.place_agent(a, cell)

    def step(self):
        self.schedule.step()

        # collect data
        # self.datacollector.collect(self)

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
