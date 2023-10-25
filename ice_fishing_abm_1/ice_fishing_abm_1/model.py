import mesa
import numpy as np

from .agent import Agent
from .resource_distribution import ResourceDistribution
from .space import MultiMicroMesoGrid


class Model(mesa.Model):
    def __init__(
            self,
            grid_width: int = 100,
            grid_height: int = 100,
            number_of_agents: int = 5,
            n_resource_clusters: int = 2,
            resource_cluster_radius: int = 5,
            sampling_length: int = 10,
            relocation_threshold: float = 0.7,
            local_search_counter: int = 4,
            prior_knowledge_corr: float = 0.0,
            prior_knowledge_noize: float = 0.2,
            w_social: float = 0.4,
            w_personal: float = 0.4,
            meso_grid_step: int = 10,
    ):
        super().__init__()
        # set numpy random seed
        np.random.seed(self.random.randint(0, 1000000))

        # parameters general
        self.number_of_agents: int = number_of_agents
        self.n_resource_clusters = n_resource_clusters
        self.resource_cluster_radius = resource_cluster_radius
        self.current_id = 0

        # agent parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.local_search_counter: int = local_search_counter
        self.prior_knowledge_corr: float = prior_knowledge_corr
        self.prior_knowledge_noize: float = prior_knowledge_noize
        self.w_social: float = w_social
        self.w_personal: float = w_personal
        self.meso_grid_step: int = meso_grid_step

        # initialize grid, datacollector, schedule
        self.grid = MultiMicroMesoGrid(grid_width, grid_height, torus=False, meso_scale_step=self.meso_grid_step)
        self.datacollector = mesa.datacollection.DataCollector(
            # agent_reporters={"Collected resource": lambda a: a.collected_resource}
            model_reporters={
                "Collected resource": lambda m: np.mean([a._collected_resource for a in m.schedule.agents])}
        )
        self.schedule = mesa.time.RandomActivation(self)

        # initialize resource distribution
        self.resource = ResourceDistribution(self,
                                             n_clusters=self.n_resource_clusters,
                                             cluster_radius=self.resource_cluster_radius,
                                             noize_level=0.01)
        self.resource.generate_resource_map()
        self.resource_distribution = self.resource.resource_distribution
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
            prior_knowledge_corr=self.prior_knowledge_corr,
            prior_knowledge_noize=self.prior_knowledge_noize,
            w_social=self.w_social,
            w_personal=self.w_personal,
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
