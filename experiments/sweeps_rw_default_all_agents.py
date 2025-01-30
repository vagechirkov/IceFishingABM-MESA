import numpy as np
import pandas as pd
import mesa
from datetime import datetime
from pathlib import Path
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy

def run_parameter_sweep(study_name="rw-default_filtering"):
    # Define parameter ranges
    mu_values = [1.1, 2.0, 3.1]
    alpha_values = np.logspace(-5, 0, num=5)  # 20 points from 10^-5 to 1
    
    # Constants
    NUM_AGENTS = 10
    GRID_SIZE = 100
    NUM_RESOURCE_CLUSTERS = 5
    RESOURCE_CLUSTER_RADIUS = 2
    RESOURCE_QUALITY = 1.0
    D_MIN = 1
    NUM_ITERATIONS = 100
    max_sim_steps = 1000
    threshold = 1  # Or could be made variable if needed

    # Store results
    results_list = []
    
    # Create directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f'parameter_sweep_{study_name}_{timestamp}')
    save_dir.mkdir(exist_ok=True)

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
                    "keep_overall_abundance": True,
                    "social_info_quality": "consuming",
                    
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
                


            # Now group by both alpha and AgentID
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

            # Append to results_list with proper format
            for _, row in results.iterrows():
                results_list.append({
                    "mu": mu,
                    "alpha": row["alpha"],
                    "avg_efficiency": float(row["collected_resource"]["mean"] / (row["traveled_distance"]["mean"] + 1e-10)),
                    "avg_resources": float(row["collected_resource"]["mean"]),
                    "avg_time_to_first": float(row["time_to_first_catch"]["mean"]),
                    "std_efficiency": float(row["collected_resource"]["std"] / (row["traveled_distance"]["std"] + 1e-10)),
                    "std_resources": float(row["collected_resource"]["std"]),
                    "std_time_to_first": float(row["time_to_first_catch"]["std"])
                })
    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(save_dir / f'{study_name}_sweep_results.csv', index=False)
    
    # Save parameters for reference
    params = {
        "study_name": study_name,
        "mu_values": mu_values,
        "alpha_range": [float(alpha_values.min()), float(alpha_values.max())],
        "num_alpha_points": len(alpha_values),
        "NUM_AGENTS": NUM_AGENTS,
        "GRID_SIZE": GRID_SIZE,
        "NUM_RESOURCE_CLUSTERS": NUM_RESOURCE_CLUSTERS,
        "RESOURCE_RADIUS": RESOURCE_CLUSTER_RADIUS,
        "NUM_ITERATIONS": NUM_ITERATIONS,
        "max_steps": max_sim_steps,
        "threshold": threshold,
        'resource_quality': RESOURCE_QUALITY,

    }
    
    import json
    with open(save_dir / f'{study_name}_parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
        
    return save_dir

if __name__ == "__main__":
    save_dir = run_parameter_sweep("rw_default_filtering")
    print(f"Results saved in: {save_dir}")
