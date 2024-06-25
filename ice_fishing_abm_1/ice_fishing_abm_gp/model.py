import mesa
import numpy as np

from .resource import Resource, make_resource_centers
from .agent import Agent


class Model(mesa.Model):
    def __init__(
            self,
            grid_size: int = 100,
            number_of_agents: int = 5,
            n_resource_clusters: int = 2,
            resource_quality: float = 0.8,
            resource_cluster_radius: int = 5,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.number_of_agents = number_of_agents
        self.n_resource_clusters = n_resource_clusters
        self.resource_quality = resource_quality
        self.resource_cluster_radius = resource_cluster_radius

        self.w_social = 0.4
        self.w_success = 0.3
        self.w_failure = 0.3

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
                keep_overall_abundance=True,
                neighborhood_radius=40,
            )
            self.schedule.add(r)
            self.grid.place_agent(r, center)

        # initialize agents
        for _ in range(self.number_of_agents):
            a = Agent(self.next_id(), self, resource_cluster_radius=self.resource_cluster_radius)
            self.schedule.add(a)
            # find a random location
            cell = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))

            # place agent
            self.grid.place_agent(a, cell)

    @property
    def resource_distribution(self) -> np.ndarray:
        # NB: resource distribution is a 2D array with the same shape as the grid and in x,y coordinates system
        return np.sum([a.resource_map() for a in self.schedule.agents if isinstance(a, Resource)], axis=0).T

    def step(self):
        self.schedule.step()
