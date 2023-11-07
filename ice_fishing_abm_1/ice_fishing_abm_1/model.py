import mesa
import numpy as np

from .agent import Agent
from .resource import Resource, make_resource_centers
from .space import MultiMicroMesoGrid


class Model(mesa.Model):
    def __init__(
            self,
            grid_width: int = 100,
            grid_height: int = 100,
            number_of_agents: int = 5,
            n_resource_clusters: int = 2,
            resource_quality: float = 0.8,
            resource_cluster_radius: int = 5,
            sampling_length: int = 10,
            relocation_threshold: float = 0.7,
            local_search_counter: int = 4,
            prior_knowledge_corr: float = 0.0,
            prior_knowledge_noize: float = 0.2,
            w_social: float = 0.4,
            w_personal: float = 0.4,
            local_learning_rate: float = 0.5,
            meso_learning_rate: float = 0.5,
            meso_grid_step: int = 10,
    ):
        super().__init__()
        # set numpy random seed
        np.random.seed(self.random.randint(0, 1000000))

        # parameters general
        self.number_of_agents: int = number_of_agents
        self.current_id = 0

        # resource parameters
        self.n_resource_clusters = n_resource_clusters
        self.resource_cluster_radius = resource_cluster_radius
        self.resource_quality = resource_quality

        # agent parameters
        self.sampling_length: int = sampling_length
        self.relocation_threshold: float = relocation_threshold
        self.local_search_counter: int = local_search_counter
        self.prior_knowledge_corr: float = prior_knowledge_corr
        self.prior_knowledge_noize: float = prior_knowledge_noize
        self.w_social: float = w_social
        self.w_personal: float = w_personal
        self.meso_grid_step: int = meso_grid_step
        self.local_learning_rate: float = local_learning_rate
        self.meso_learning_rate: float = meso_learning_rate

        # initialize grid, datacollector, schedule
        self.grid = MultiMicroMesoGrid(grid_width, grid_height, torus=False, meso_scale_step=self.meso_grid_step)
        self.datacollector = mesa.datacollection.DataCollector(
            # agent_reporters={"Collected resource": lambda a: a.collected_resource}
            model_reporters={
                "Collected resource": lambda m: np.mean(
                    [a._collected_resource for a in m.schedule.agents if isinstance(a, Agent)])}
        )
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for _ in range(self.number_of_agents):
            self.initialize_agent()

        # initialize resource distribution
        self.initialize_resource()

    @property
    def resource_distribution(self) -> np.ndarray:
        """
        Return a resource distribution.
        """
        return np.sum([a.resource_map() for a in self.schedule.agents if isinstance(a, Resource)], axis=0)

    def initialize_resource(self):
        centers = make_resource_centers(self, self.n_resource_clusters, self.resource_cluster_radius)
        for center in centers:
            r = Resource(
                self.next_id(),
                self,
                radius=self.resource_cluster_radius,
                max_value=100,
                current_value=int(self.resource_quality * 100),
                keep_overall_abundance=True,
                neighborhood_radius=20,
            )
            self.schedule.add(r)
            self.grid.place_agent(r, center)

    def initialize_agent(self, radius: int = 1) -> None:
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
            local_learning_rate=self.local_learning_rate,
            meso_learning_rate=self.meso_learning_rate,
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
