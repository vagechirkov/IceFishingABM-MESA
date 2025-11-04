from typing import Union

import mesa
import numpy as np

from abm.exploitation_strategy import ExploitationStrategy, IceFishingExploitationStrategy
from abm.exploration_strategy import ExplorationStrategy, KernelBeliefExploration
from abm.resource import Resource, make_resource_centers, spatiotemporal_fish_density
from abm.agent import Agent


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
        simulation_length_minutes: int = 120,  # minutes
        steps_per_minute: int = 6,
        fish_length_scale_minutes: float = 15.0,
        fish_length_scale_meters: float = 6.0,
        fish_abundance: float = 3.5,  # 0.008/10s -> 0.15/200s
        fish_density_sharpness: float = 0.5,
        spot_leaving_baseline_weight: float = -3,
        spot_leaving_fish_catch_weight: float = -1.7,
        spot_leaving_time_weight: float = 0.13,  # 0.8 for 1-minute dt
        spot_leaving_social_weight: float = -0.33,
        spot_selection_tau: float = 1.0,
        spot_selection_social_length_scale: float = 25.0,
        spot_selection_success_length_scale: float = 10.0,
        spot_selection_failure_length_scale: float = 10.0,
        spot_selection_w_social: float = 0.2,
        spot_selection_w_success: float = 0.2,
        spot_selection_w_failure: float = 0.4,
        spot_selection_w_locality: float = 0.2,
        spot_selection_social_info_quality: str = "sampling",  # "sampling" or "consuming"
        agent_speed_m_per_min: float = 15.0,
        agent_margin_from_others: float = 5.0,
        sample_from_prior = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.simulation_length_minutes = simulation_length_minutes
        self.num_agents = number_of_agents
        self.steps_per_minute = steps_per_minute
        self.steps_min = 0

        self.fish_length_scale_minutes = fish_length_scale_minutes
        self.fish_length_scale_meters = fish_length_scale_meters
        self.fish_abundance = fish_abundance
        self.fish_density_sharpness = fish_density_sharpness

        self.spot_leaving_baseline_weight = spot_leaving_baseline_weight
        self.spot_leaving_fish_catch_weight = spot_leaving_fish_catch_weight
        self.spot_leaving_time_weight = spot_leaving_time_weight
        self.spot_leaving_social_weight = spot_leaving_social_weight

        self.spot_selection_tau = spot_selection_tau
        self.spot_selection_social_length_scale = spot_selection_social_length_scale
        self.spot_selection_success_length_scale = spot_selection_success_length_scale
        self.spot_selection_failure_length_scale = spot_selection_failure_length_scale
        self.spot_selection_social_info_quality = spot_selection_social_info_quality

        self.spot_selection_w_social = spot_selection_w_social
        self.spot_selection_w_success = spot_selection_w_success
        self.spot_selection_w_failure = spot_selection_w_failure
        self.spot_selection_w_locality = spot_selection_w_locality
        self._normalize_spot_selection_weights()

        self.agent_speed_m_per_min = agent_speed_m_per_min
        self.agent_margin_from_others = agent_margin_from_others

        if sample_from_prior is not None:
            prior_vals = sample_from_prior() if callable(sample_from_prior) else dict(sample_from_prior)

            # Map any provided keys to attributes (only those present will override)
            for k, v in prior_vals.items():
                if hasattr(self, k):
                    setattr(self, k, v)

            # (optional) keep a copy for logging/analysis
            self.sampled_prior = prior_vals

        fish_density, _, _, _ = spatiotemporal_fish_density(
            length_scale_time=self.fish_length_scale_minutes,
            length_scale_space=self.fish_length_scale_meters,
            n_x=self.grid_size,
            n_y=self.grid_size,
            n_time=self.simulation_length_minutes + 1,
            n_samples=1,
            temperature=self.fish_density_sharpness,
            bias=self.fish_abundance,
            # rng=self.rng
        )
        self.fish_density = fish_density[0]

        # initialize exploration and exploitation models
        exploitation_strategy_list = [IceFishingExploitationStrategy(
            step_minutes=1 / self.steps_per_minute,
            baseline_weight=self.spot_leaving_baseline_weight,
            fish_catch_weight=self.spot_leaving_fish_catch_weight,
            time_weight=self.spot_leaving_time_weight,
            social_feature_weight=self.spot_leaving_social_weight
            # rng=self.rng
        ) for _ in range(number_of_agents)]

        exploration_strategy_list = [KernelBeliefExploration(
            grid_size=self.grid_size,
            tau=self.spot_selection_tau,
            social_length_scale=self.spot_selection_social_length_scale,
            success_length_scale=self.spot_selection_success_length_scale,
            failure_length_scale=self.spot_selection_failure_length_scale,
            w_social=self.spot_selection_w_social,
            w_success=self.spot_selection_w_success,
            w_failure=self.spot_selection_w_failure,
            w_locality=self.spot_selection_w_locality,
            w_as_attention_shares=True,
            model_type="kde",
            normalize_features=True,
            # rng=self.rng
        ) for _ in range(self.num_agents)]

        self.grid = mesa.space.MultiGrid(grid_size, grid_size, False)

        Agent.create_agents(
            self,
            self.num_agents,
            initial_position=(grid_size // 2, grid_size// 2),
            exploration_strategy=exploration_strategy_list,
            exploitation_strategy=exploitation_strategy_list,
            speed_m_per_step=self.agent_speed_m_per_min / self.steps_per_minute,
            margin_from_others=self.agent_margin_from_others,
            social_info_quality=self.spot_selection_social_info_quality,
            resource_cluster_radius=None
        )

        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "catch": lambda m: np.mean([a.collected_resource for a in m.agents]),
                "travel_distance": lambda m: np.mean(
                    [a.traveled_distance_euclidean for a in m.agents]
                ),
                "successful_locations": lambda m: np.mean(
                    [a.n_successful_locations for a in m.agents]
                ),
                "failure_locations": lambda m: np.mean(
                    [a.n_failure_locations for a in m.agents]
                ),
                "sampling_time_successful_spot": lambda m: np.mean(
                    [a.time_sampling_successful_spots for a in m.agents]
                ) / self.steps_per_minute,
                "sampling_time_failure_spot": lambda m: np.mean(
                    [a.time_sampling_failure_spots for a in m.agents]
                ) / self.steps_per_minute,

                # parameters
                "spot_selection_w_social": lambda m: m.spot_selection_w_social,
                "spot_selection_w_success": lambda m: m.spot_selection_w_success,
                "spot_selection_w_failure": lambda m: m.spot_selection_w_failure,
                "spot_selection_w_locality": lambda m: m.spot_selection_w_locality,
                "spot_selection_tau": lambda m: m.spot_selection_tau,
                "spot_leaving_baseline_weight": lambda m: m.spot_leaving_baseline_weight,
                "spot_leaving_fish_catch_weight": lambda m: m.spot_leaving_fish_catch_weight,
                "spot_leaving_time_weight": lambda m: m.spot_leaving_time_weight,
                "spot_leaving_social_weight": lambda m: m.spot_leaving_social_weight,
                "fish_abundance": lambda m: m.fish_abundance,
            }
        )

    def sample_fish_density(self, i, j):
        """Return True if a fish is caught, False otherwise."""
        # TODO: implement constant depletion rate
        p_catch = self.fish_density[int(i), int(j), self.steps_min]
        return np.random.random() < p_catch

    def step(self):
        self.steps_min = self.steps // self.steps_per_minute
        assert self.steps_min <= self.simulation_length_minutes
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def _normalize_spot_selection_weights(self):
        """
        Normalizes the spot selection weights (social, success, failure, locality)
        to ensure they sum to 1.0.
        """
        weights = [
            self.spot_selection_w_social,
            self.spot_selection_w_success,
            self.spot_selection_w_failure,
            self.spot_selection_w_locality,
        ]

        total_weight = sum(weights)

        if total_weight != 1.0:
            # Normalize weights if they don't sum to 1
            self.spot_selection_w_social = self.spot_selection_w_social / total_weight
            self.spot_selection_w_success = self.spot_selection_w_success / total_weight
            self.spot_selection_w_failure = self.spot_selection_w_failure / total_weight
            self.spot_selection_w_locality = self.spot_selection_w_locality / total_weight