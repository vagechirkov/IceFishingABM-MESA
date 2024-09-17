from typing import Union

import mesa
import numpy as np

from .movement_destination_subroutine import ExplorationStrategy
from .patch_evaluation_subroutine import PatchEvaluationSubroutine
from .resource import Resource, make_resource_centers
from .agent import Agent


class Model(mesa.Model):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy = ExplorationStrategy(),
            exploitation_strategy: PatchEvaluationSubroutine = PatchEvaluationSubroutine(threshold=10),
            grid_size: int = 100,
            number_of_agents: int = 5,
            n_resource_clusters: int = 2,
            resource_quality: Union[float, tuple[float]] = 0.8,
            resource_cluster_radius: int = 5,
            keep_overall_abundance: bool = True, ):
        super().__init__()
        self.grid_size = grid_size
        self.number_of_agents = number_of_agents
        self.n_resource_clusters = n_resource_clusters
        self.resource_quality = resource_quality
        self.resource_cluster_radius = resource_cluster_radius
        self.keep_overall_abundance = keep_overall_abundance

        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(grid_size, grid_size, False)

        # initialize resources
        centers = make_resource_centers(self, self.n_resource_clusters, self.resource_cluster_radius)
        for n, (center) in enumerate(centers):
            quality = self.resource_quality if isinstance(self.resource_quality, float) else self.resource_quality[n]
            r = Resource(
                self.next_id(),
                self,
                radius=self.resource_cluster_radius,
                max_value=100,
                current_value=int(quality * 100),
                keep_overall_abundance=self.keep_overall_abundance,
                neighborhood_radius=40,
            )
            r.collected_resource = None
            r.is_sampling = None
            r.is_moving = None
            self.schedule.add(r)
            self.grid.place_agent(r, center)

        # initialize agents
        for _ in range(self.number_of_agents):
            a = Agent(self.next_id(),
                      self,
                      self.resource_cluster_radius,
                      exploration_strategy,
                      exploitation_strategy)
            self.schedule.add(a)
            # find a random location
            cell = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))

            # place agent
            self.grid.place_agent(a, cell)

        # Data collector
        model_reporters = {}
        agent_reporters = {
            "pos": "pos",
            "collected_resource": "collected_resource",
            "is_sampling": "is_sampling",
            "is_moving": "is_moving"}

        self.datacollector = mesa.datacollection.DataCollector(
            agent_reporters=agent_reporters,
            model_reporters=model_reporters
        )

    @property
    def resource_distribution(self) -> np.ndarray:
        # NB: resource distribution is a 2D array with the same shape as the grid and in x,y coordinates system
        return np.sum([a.resource_map() for a in self.schedule.agents if isinstance(a, Resource)], axis=0).T

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
