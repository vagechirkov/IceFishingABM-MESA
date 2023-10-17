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
            resource_cluster_radius: int = 5,
            visualization: bool = False,
            sampling_length: int = 10,
            relocation_threshold: float = 0.7,
            local_search_counter: int = 4,
            alpha_social: float = 0.4,
            alpha_env: float = 0.4,
            prior_knowledge: float = 0.05,
            meso_grid_step: int = 10,
    ):
        super().__init__()
        # set numpy random seed
        np.random.seed(self.random.randint(0, 1000000))

        # parameters general
        self.number_of_agents: int = number_of_agents
        self.n_resource_clusters = n_resource_clusters
        self.resource_cluster_radius = resource_cluster_radius
        self.visualization = visualization
        self.current_id = 0

        # agent parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.local_search_counter: int = local_search_counter
        self.alpha_social: float = alpha_social
        self.alpha_env: float = alpha_env
        self.prior_knowledge: float = prior_knowledge
        self.meso_grid_step: int = meso_grid_step

        # initialize grid, datacollector, schedule
        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        self.datacollector = mesa.datacollection.DataCollector(
            # agent_reporters={"Collected resource": lambda a: a.collected_resource}
            model_reporters={"Collected resource": lambda m: np.mean([a.collected_resource for a in m.schedule.agents])}
        )
        self.schedule = mesa.time.RandomActivation(self)

        # initialize resource distribution
        self.resource = ResourceDistribution(self,
                                             n_clusters=self.n_resource_clusters,
                                             cluster_radius=self.resource_cluster_radius,
                                             noize_level=0.01)
        self.resource.generate_resource_map()
        self.resource_distribution = self.resource.resource_distribution

        # add resource distribution to grid colors for visualization
        if self.visualization:
            shape = (self.grid.width, self.grid.height)
            self.grid_colors = self.resource_distribution
            self.agent_raw_observations = np.zeros(shape=shape, dtype=float)
            self.agent_raw_env_belief = np.zeros(shape=shape, dtype=float)
            self.agent_raw_soc_observations = np.zeros(shape=shape, dtype=float)
            self.agent_raw_rand_array = np.zeros(shape=shape, dtype=float)
            self.relocation_map = np.zeros(shape=shape, dtype=float)

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
            sampling_length=self.sampling_length,
            relocation_threshold=self.relocation_threshold,
            local_search_counter=self.local_search_counter,
            meso_grid_step=self.meso_grid_step,
            prior_knowledge=self.prior_knowledge,
            alpha_social=self.alpha_social,
            alpha_env=self.alpha_env,
            visualization=self.visualization
        )

        self.schedule.add(a)
        # find a random location
        cell = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))

        # place agent
        self.grid.place_agent(a, cell)

    def step(self):
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
