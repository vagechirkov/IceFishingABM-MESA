from typing import Union

import mesa
import numpy as np

from .exploitation_strategy import ExploitationStrategy, IceFishingExploitationStrategy
from .exploration_strategy import ExplorationStrategy, GPExplorationStrategy
from .resource import Resource, make_resource_centers
from .agent import Agent


class Model(mesa.Model):
    def __init__(
        self,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy(),
        exploitation_strategy: ExploitationStrategy = ExploitationStrategy(
            threshold=10
        ),
        grid_size: int = 100,
        number_of_agents: int = 5,
        n_resource_clusters: int = 2,
        resource_quality: Union[float, tuple[float]] = 0.8,
        resource_cluster_radius: int = 5,
        keep_overall_abundance: bool = True,
        social_info_quality = None,
        resource_max_value: int = 100, 
    ):
        super().__init__()
        self.grid_size = grid_size
        self.number_of_agents = number_of_agents
        self.n_resource_clusters = n_resource_clusters
        self.resource_quality = resource_quality
        self.resource_cluster_radius = resource_cluster_radius
        self.keep_overall_abundance = keep_overall_abundance
        self.social_info_quality = social_info_quality
        
        # Add resource tracking
        self.total_initial_resource = n_resource_clusters * resource_max_value  # Total possible resource
        self.total_consumed_resource = 0                                        # Track consumed resources
        self.running = True  
        self.consumption_criterion = 0.30

        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(grid_size, grid_size, False)

        # agent parameters
        

        # initialize resources
        centers = make_resource_centers(
            self, self.n_resource_clusters, self.resource_cluster_radius
        )
        for n, (center) in enumerate(centers):
            quality = (
                self.resource_quality
                if isinstance(self.resource_quality, float)
                else self.resource_quality[n]
            )
            r = Resource(
                self.next_id(),
                self,
                radius=self.resource_cluster_radius,
                max_value=resource_max_value,
                current_value=int(quality * resource_max_value),
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
            a = Agent(
                self.next_id(),
                self,
                self.resource_cluster_radius,
                self.social_info_quality,
                exploration_strategy,
                exploitation_strategy,
            )
            self.schedule.add(a)
            # find a random location
            cell = (
                self.random.randint(0, self.grid.width - 1),
                self.random.randint(0, self.grid.height - 1),
            )

            # place agent
            self.grid.place_agent(a, cell)

        self.datacollector = mesa.datacollection.DataCollector(
        agent_reporters={
            "collected_resource": lambda agent: (
                agent.collected_resource
                if hasattr(agent, "collected_resource")
                else None
            ),  # Safeguard for non-agent objects
            "traveled_distance": lambda agent: (
                agent.traveled_distance_euclidean
                if hasattr(agent, "traveled_distance_euclidean")
                else None
            ),  # Safeguard for non-agent objects
            "pos": "pos",
            "is_sampling": "is_sampling",
            "is_moving": "is_moving",
            "step_sizes": lambda a: a._step_sizes if hasattr(a, '_step_sizes') else None,
            "time_to_first_catch": lambda a: a._time_to_first_catch if hasattr(a, '_time_to_first_catch') else None,
            "total_sampling_time": "total_sampling_time",
            "total_consuming_time": "total_consuming_time",
            "cluster_catches": "cluster_catches"
        },
        model_reporters={
            "steps_completed": lambda m: m.schedule.steps,
            "resources_consumed": lambda m: m.total_consumed_resource,
            "total_resources": lambda m: m.total_initial_resource,
        },
       )
        

    @property
    def resource_distribution(self) -> np.ndarray:
        # NB: resource distribution is a 2D array with the same shape as the grid and in x,y coordinates system
        return np.sum(
            [a.resource_map() for a in self.schedule.agents if isinstance(a, Resource)],
            axis=0,
        ).T

    def step(self):
        """
        Run one step of the model. This includes updating the schedule, collecting data, etc.
        """
        self.schedule.step()
        self.datacollector.collect(self)

        # Check if 30% of total resource has been consumed
        if self.total_consumed_resource >= self.consumption_criterion * self.total_initial_resource:
            print(f"\nSimulation stopped at step {self.schedule.steps}")
            print(f"Total initial resource: {self.total_initial_resource}")
            print(f"Resources consumed: {self.total_consumed_resource} ({(self.total_consumed_resource/self.total_initial_resource)*100:.1f}%)")
            self.running = False
            

class IceFishingModel(mesa.Model):
    def __init__(
            self,
            grid_size: int = 100,
            number_of_agents: int = 5,
            fish_densities: np.ndarray = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = number_of_agents
        self.fish_densities = fish_densities
        exploration_strategy = GPExplorationStrategy()
        exploitation_strategy = IceFishingExploitationStrategy(step_minutes=1.0)

        self.grid = mesa.discrete_space.OrthogonalMooreGrid((grid_size, grid_size), torus=False)
        self.datacollector = mesa.datacollection.DataCollector()

        Agent.create_agents(
            self,
            self.num_agents,
            cell=self.rng.choice(
                self.grid.all_cells, replace=False, size=self.num_agents
            ),
            exploration_strategy=exploration_strategy,
            exploitation_strategy=exploitation_strategy,
        )

    def sample_fish_density(self, i, j):
        """Return True if a fish is caught, False otherwise."""
        p_catch = self.fish_densities[int(i), int(j), self.steps - 1]
        return np.random.random() < p_catch

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)