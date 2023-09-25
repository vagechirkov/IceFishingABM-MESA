import logging

import mesa
import numpy as np
from mesa.space import MultiGrid

from .agent import Agent
from .resource_distribution import ResourceDistribution


class Model(mesa.Model):
    def __init__(
            self,
            grid_width: int = 100,
            grid_height: int = 100,
            number_of_agents: int = 5,
            n_resource_clusters: int = 2,
            visualization: bool = False,
            sampling_length: int = 10,
            relocation_threshold: float = 0.7,
            social_influence_threshold: float = 1,
            exploration_threshold: float = 0.01,
            prior_knowledge: float = 0.05
    ):
        super().__init__()
        # parameters general
        self.number_of_agents: int = number_of_agents
        self.n_resource_clusters = n_resource_clusters
        self.visualization = visualization
        self.current_id = 0

        # agent parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.social_influence_threshold: float = social_influence_threshold
        self.exploration_threshold: float = exploration_threshold
        self.prior_knowledge: float = prior_knowledge

        # initialize grid, datacollector, schedule
        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        self.datacollector = mesa.datacollection.DataCollector(
            # agent_reporters={"Collected resource": lambda a: a.collected_resource}
            model_reporters={"Collected resource": lambda m: np.mean([a.collected_resource for a in m.schedule.agents])}
        )
        self.schedule = mesa.time.RandomActivation(self)

        # initialize resource distribution
        self.resource = ResourceDistribution(self, n_clusters=self.n_resource_clusters)
        self.resource.generate_resource_map()
        self.resource_distribution = self.resource.resource_distribution

        # add resource distribution to grid colors for visualization
        if self.visualization:
            self.grid_colors = self.resource_distribution
            self.agent_raw_observations = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)
            self.agent_smoothed_observations = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)
            self.agent_discounted_observations = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)
            self.agent_raw_soc_observations = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)
            self.agent_discounted_soc_observations = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)
            self.agent_smoothed_soc_observations = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)
            self.relocation_map = np.zeros(shape=(self.grid.width, self.grid.height), dtype=float)

        # Create agents
        for _ in range(self.number_of_agents):
            self.initialize_agent()

    def initialize_agent(self, radius: int = 1) -> None:
        """
        Create an agent and add it to the schedule and grid at a random empty cell in the center of the grid.

        :param radius: The distance from the center of the grid to place the agent.
        """
        assert radius > 0, "Radius must be greater than 0."
        a = Agent(
            self.next_id(),
            self,
            visualization=self.visualization,
            sampling_length=self.sampling_length,
            relocation_threshold=self.relocation_threshold,
            social_influence_threshold=self.social_influence_threshold,
            exploration_threshold=self.exploration_threshold,
            prior_knowledge=self.prior_knowledge
        )

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
        self.datacollector.collect(self)

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
