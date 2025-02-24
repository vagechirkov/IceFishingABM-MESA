import numpy as np  scipy.stats as st
import pandas as pd
import mesa
from datetime import datetime
from pathlib import Path
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy
from visualization.visualize_agent_movement import save_agent_movement_gif

def run_parameter_sweep(social_info,study_name, num_resource_clusters = 5,  max_sim_steps=1_000): 
    # Define parameter ranges
    
    mu_values = [2.0]
    alpha_values = np.array([0.00001, 0.0001, 0.001 ,0.01 ,0.1,0.999])#np.linspace(0, 1, num=5)  # 20 points from 10^-5 to 1
    
    # Constants
    NUM_AGENTS = 10
    GRID_SIZE =  100
    NUM_RESOURCE_CLUSTERS = num_resource_clusters 
    RESOURCE_CLUSTER_RADIUS = 2
    RESOURCE_QUALITY = 1.0
    D_MIN = 1
    NUM_ITERATIONS = 10_000
    RESOURCE_UNITS  = int(500 / NUM_RESOURCE_CLUSTERS)
    max_sim_steps = max_sim_steps
    threshold = 1  # Or could be made variable if needed

    # Store results
    results_list = []
    
    # Create directory for results
    base_dir = Path('parameter_sweep_results')
    base_dir.mkdir(exist_ok=True)
    
    # Create specific run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = base_dir / f'parameter_sweep_{study_name}_clusters{NUM_RESOURCE_CLUSTERS}_steps{max_sim_steps}_{timestamp}'
    save_dir.mkdir(exist_ok=True)
    
    
    # Save GIF for each mu-alpha combination
    save_gif = False
    gif_dir = save_dir / 'gifs'
    gif_dir.mkdir(exist_ok=True)

    for mu in mu_values:
        if True:
            #print(f"Running simulation with mu={mu}, alpha={alpha}")
            
            # Set up strategies
            exploration_strategy_list = []
            alpha_index_map = {}
            for i, alpha in enumerate(alpha_values):
                
                exploration_strategy = RandomWalkerExplorationStrategy(
                    mu=mu,
                    dmin=D_MIN,
                    L=GRID_SIZE,
                    alpha=alpha,
                    grid_size=GRID_SIZE,
                )
                exploration_strategy_list.append(exploration_strategy)
                #alpha_index_map[i] = alpha
                
            exploitation_strategy = ExploitationStrategy(threshold=threshold)

            
            
            # Run simulation
            batch_results = mesa.batch_run(
                RandomWalkerModel,
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


            # Replace the existing GIF saving loop with:
            for alpha_val in df['alpha'].unique():
                # Get first model instance for this alpha
                model_row = df[df['alpha'] == alpha_val].iloc[0]
                model = RandomWalkerModel(
                    exploration_strategy=RandomWalkerExplorationStrategy(
                        mu=mu,
                        dmin=D_MIN,
                        L=GRID_SIZE,
                        alpha=alpha_val,
                        grid_size=GRID_SIZE,
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
                
                if save_gif:
                    save_agent_movement_gif(
                        model,
                        steps=max_sim_steps,
                        filename=gif_dir / f'agent_movement_mu_{mu}_alpha_{alpha_val:.5f}.gif',
                        resource_cluster_radius=RESOURCE_CLUSTER_RADIUS,
                    )


            # Now group by both alpha and AgentID
            agent_mask = df.AgentID != 0
            agent_metrics = df[agent_mask].groupby(["AgentID", "alpha"]).agg({
                "collected_resource": "max",
                "traveled_distance": "max",
                "time_to_first_catch": "first",
            }).reset_index()

            # Calculate group statistics
            results = agent_metrics.groupby("alpha").agg({
                "collected_resource": ["mean", "std" , "ci"],
                "traveled_distance": ["mean", "std", "ci"],
                "time_to_first_catch": ["mean", "std", "ci"]
            }).reset_index()

            # Append to results_list with proper format
            for _, row in results.iterrows():
                # Get current alpha value
                current_alpha = float(row['alpha'])  # Ensure alpha is a single float value
                n = len(agent_metrics[agent_metrics['alpha'].astype(float) == current_alpha])
    
                
                # Calculate means and standard deviations
                mean_efficiency = float(row["collected_resource"]["mean"] / (row["traveled_distance"]["mean"] + 1e-10))
                mean_resources = float(row["collected_resource"]["mean"])
                mean_time = float(row["time_to_first_catch"]["mean"])
                
                # Calcutae std 
                std_efficiency = float(row["collected_resource"]["std"] / (row["traveled_distance"]["std"] + 1e-10))
                std_resources = float(row["collected_resource"]["std"])
                std_time = float(row["time_to_first_catch"]["std"])
                
                # calcuate confidence interval 

                results_list.append({
                    "mu": float(mu),
                    "alpha": current_alpha,
                    "avg_efficiency": mean_efficiency,
                    "avg_resources": mean_resources,
                    "avg_time_to_first": mean_time,
                    "std_efficiency": std_efficiency,
                    "std_resources": std_resources,
                    "std_time_to_first": std_time,
                    "se_efficiency": std_efficiency / np.sqrt(n),
                    "se_resources": std_resources / np.sqrt(n),
                    "se_time_to_first": std_time / np.sqrt(n)
                })
    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(save_dir / f'{study_name}_sweep_results.csv', index=False)
    
    # Save parameters for reference
    params = {
        # Existing parameters
        "study_name": study_name,
        "mu_values": mu_values,
        "alpha_values": alpha_values.tolist(),  # Add full array
        "social_info_quality": social_info,     # Add social info type
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
    NUMBER_CLUSTERS_LIST = [1, 5, 10,  15, 20, 30]  
    STEPS_LIST = [100_000]
    #social_info = "all-agents"

    for NUMBER_CLUSTERS in NUMBER_CLUSTERS_LIST: 
        for STEPS in STEPS_LIST:

            print(f"\nRunning simulation with:")
            print(f"- Number of clusters: {NUMBER_CLUSTERS}")
            print(f"- Time steps: {STEPS:,}")
            print(f"- Social info: {social_info}")
            print("-" * 50)

            if social_info == "all-agents":
                save_dir = run_parameter_sweep(social_info, "rw_default_all_agents", NUMBER_CLUSTERS, STEPS)
            elif social_info == "filtering": 
                save_dir = run_parameter_sweep(social_info, "rw_default_filtering", NUMBER_CLUSTERS, STEPS)
            
                print(f"\nResults saved in: {save_dir}")
                print("=" * 50)

