import numpy as np, scipy.stats as st
import pandas as pd
import mesa
from datetime import datetime
from pathlib import Path
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy
from visualization.visualize_agent_movement import save_agent_movement_gif
from abm.resource import make_resource_centers

class ResourcePatchModel(RandomWalkerModel):
    """Extension of RandomWalkerModel that places one agent directly on a resource patch"""
    
    def __init__(self, *args, **kwargs):
        # First call the parent's __init__ to complete all initialization
        super().__init__(*args, **kwargs)
        
        # Get resource centers
        resource_centers = make_resource_centers(
            self, self.n_resource_clusters, self.resource_cluster_radius
        )
        
        # Find the agents in the scheduler after initialization is complete
        agents = [agent for agent in self.schedule.agents 
                  if hasattr(agent, "is_agent") and agent.is_agent]
        
        if agents:
            # Get the first agent
            first_agent = agents[0]
            
            # Remove the agent from its current position
            self.grid.remove_agent(first_agent)
            
            # Place the agent on the first resource center
            self.grid.place_agent(first_agent, resource_centers[0])
            print(f"Placed agent at resource center: {resource_centers[0]}")


def run_parameter_sweep(social_info, study_name, num_resource_clusters=5, max_sim_steps=1_000, resource_radius=2): 
    # Define parameter ranges
    mu_values = [2.0]
    alpha_values = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.999])
    
    # Constants
    NUM_AGENTS = 10
    GRID_SIZE = 100
    NUM_RESOURCE_CLUSTERS = num_resource_clusters 
    RESOURCE_CLUSTER_RADIUS = resource_radius 
    RESOURCE_QUALITY = 1.0
    D_MIN = 1
    NUM_ITERATIONS = 1_000
    RESOURCE_UNITS = int(500 / NUM_RESOURCE_CLUSTERS)
    max_sim_steps = max_sim_steps
    threshold = 1
    
    # Store results
    results_list = []
    
    # Create directory for results
    base_dir = Path('parameter_sweep_results')
    base_dir.mkdir(exist_ok=True)
    
    # Create specific run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = base_dir / f'on_patch_test_parameter_sweep_{study_name}_clusters{NUM_RESOURCE_CLUSTERS}_radius{RESOURCE_CLUSTER_RADIUS}_steps{max_sim_steps}_{timestamp}'
    save_dir.mkdir(exist_ok=True)
    
    # Save GIF for each mu-alpha combination
    save_gif = False
    gif_dir = save_dir / 'gifs'
    gif_dir.mkdir(exist_ok=True)

    for mu in mu_values:
        # Set up strategies
        exploration_strategy_list = []
        for i, alpha in enumerate(alpha_values):
            exploration_strategy = RandomWalkerExplorationStrategy(
                mu=mu,
                dmin=D_MIN,
                L=GRID_SIZE,
                alpha=alpha,
                grid_size=GRID_SIZE,
            )
            exploration_strategy_list.append(exploration_strategy)
            
        exploitation_strategy = ExploitationStrategy(threshold=threshold)
        
        # Run simulation with agents placed on resource patches
        batch_results = mesa.batch_run(
            ResourcePatchModel,  # Use our custom model class
            parameters={
                "exploration_strategy": exploration_strategy_list,
                "exploitation_strategy": exploitation_strategy,
                "grid_size": GRID_SIZE,
                "number_of_agents": NUM_AGENTS,
                "n_resource_clusters": NUM_RESOURCE_CLUSTERS,
                "resource_quality": RESOURCE_QUALITY,
                "resource_cluster_radius": RESOURCE_CLUSTER_RADIUS,
                "resource_max_value": RESOURCE_UNITS,
                "keep_overall_abundance": True,
                "social_info_quality": "sampling",
            },
            iterations=NUM_ITERATIONS,
            max_steps=max_sim_steps,
            data_collection_period=-1,
            number_processes=None,
            display_progress=True,
        )
        
        # Process results
        df = pd.DataFrame(batch_results)
        df['alpha'] = df['exploration_strategy'].apply(lambda x: x.alpha if x is not None else None)

        # Save GIFs if enabled
        if save_gif:
            for alpha_val in df['alpha'].unique():
                model = ResourcePatchModel(
                    exploration_strategy=RandomWalkerExplorationStrategy(
                        mu=mu, dmin=D_MIN, L=GRID_SIZE, alpha=alpha_val, grid_size=GRID_SIZE,
                    ),
                    exploitation_strategy=exploitation_strategy,
                    grid_size=GRID_SIZE,
                    number_of_agents=NUM_AGENTS,
                    n_resource_clusters=NUM_RESOURCE_CLUSTERS,
                    resource_quality=RESOURCE_QUALITY,
                    resource_cluster_radius=RESOURCE_CLUSTER_RADIUS,
                    resource_max_value=RESOURCE_UNITS,
                    keep_overall_abundance=True,
                    social_info_quality="sampling",
                )
                
                save_agent_movement_gif(
                    model,
                    steps=max_sim_steps,
                    filename=gif_dir / f'agent_movement_mu_{mu}_alpha_{alpha_val:.5f}.gif',
                    resource_cluster_radius=RESOURCE_CLUSTER_RADIUS,
                )

        # Group by AgentID and alpha for analysis
        agent_mask = df.AgentID != 0
        agent_metrics = df[agent_mask].groupby(["AgentID", "alpha"]).agg({
            "collected_resource": "max",
            "traveled_distance": "max",
            "time_to_first_catch": "first",
        }).reset_index()

        # Calculate group statistics
        results = agent_metrics.groupby("alpha").agg({
            "collected_resource": ["mean", "std"],
            "traveled_distance": ["mean", "std"],
            "time_to_first_catch": ["mean", "std"]
        }).reset_index()

        # Calculate and append statistics with confidence intervals
        for _, row in results.iterrows():
            current_alpha = float(row['alpha'])
            group_data = agent_metrics[agent_metrics['alpha'].astype(float) == current_alpha]
            n = len(group_data)

            # Calculate confidence intervals (95%)
            ci_factor = st.t.ppf(0.975, n-1)
            
            # Calculate means
            mean_efficiency = float(group_data['collected_resource'].mean() / (group_data['traveled_distance'].mean() + 1e-10))
            mean_resources = float(group_data['collected_resource'].mean())
            mean_time = float(group_data['time_to_first_catch'].mean())
            
            # Calculate standard deviations
            std_efficiency = float(row["collected_resource"]["std"] / (row["traveled_distance"]["std"] + 1e-10))
            std_resources = float(row["collected_resource"]["std"])
            std_time = float(row["time_to_first_catch"]["std"])
            
            # Calculate standard errors
            se_efficiency = std_efficiency / np.sqrt(n)
            se_resources = std_resources / np.sqrt(n)
            se_time = std_time / np.sqrt(n)

            # Calculate confidence intervals
            ci_efficiency = se_efficiency * ci_factor
            ci_resources = se_resources * ci_factor
            ci_time = se_time * ci_factor

            results_list.append({
                "mu": float(mu),
                "alpha": current_alpha,
                "avg_efficiency": mean_efficiency,
                "avg_resources": mean_resources,
                "avg_time_to_first": mean_time,
                "std_efficiency": std_efficiency,
                "std_resources": std_resources,
                "std_time_to_first": std_time,
                "se_efficiency": se_efficiency,
                "se_resources": se_resources,
                "se_time_to_first": se_time,
                "ci_efficiency": ci_efficiency,
                "ci_resources": ci_resources,
                "ci_time": ci_time
            })
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(save_dir / f'{study_name}_sweep_results.csv', index=False)
    
    # Save parameters for reference
    params = {
        "study_name": study_name,
        "mu_values": mu_values,
        "alpha_values": alpha_values.tolist(),
        "social_info_quality": social_info,
        "agent_on_resource": True,  # Flag indicating one agent starts on resource
        "simulation": {
            "num_agents": NUM_AGENTS,
            "grid_size": GRID_SIZE,
            "num_resource_clusters": NUM_RESOURCE_CLUSTERS,
            "resource_cluster_radius": RESOURCE_CLUSTER_RADIUS,
            "resource_units": RESOURCE_UNITS,
            "resource_quality": RESOURCE_QUALITY,
            "d_min": D_MIN,
            "num_iterations": NUM_ITERATIONS,
            "max_steps": max_sim_steps,
            "threshold": threshold,
        }
    }
    
    import json
    with open(save_dir / f'{study_name}_parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
        
    return save_dir

if __name__ == "__main__":
    social_info = "filtering"  
    RESOURCE_RADIUS_PARAMS = [2, 5] 
    NUMBER_CLUSTERS_LIST = [1]  
    STEPS_LIST = [100_000]
    
    for RESOURCE_RADIUS_PARAM in RESOURCE_RADIUS_PARAMS:
        for NUMBER_CLUSTERS in NUMBER_CLUSTERS_LIST: 
            for STEPS in STEPS_LIST:
                print(f"\nRunning Resource Patch experiment with:")
                print(f"- Number of clusters: {NUMBER_CLUSTERS}")
                print(f"- Resource radius: {RESOURCE_RADIUS_PARAM}")
                print(f"- Time steps: {STEPS:,}")
                print(f"- Social info: {social_info}")
                print(f"- One agent starting on resource")
                print("-" * 50)

                study_name = "rw_patch_experiment"
                save_dir = run_parameter_sweep(social_info, study_name, NUMBER_CLUSTERS, STEPS, RESOURCE_RADIUS_PARAM)
                
                print(f"\nResults saved in: {save_dir}")
                print("=" * 50)