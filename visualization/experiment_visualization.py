import optuna
import pandas as pd
import mesa
import matplotlib.pyplot as plt
import plotly.io as pio
import json
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy


def load_study_and_params(study_name="foraging"):
    """Load the saved study and best parameters"""
    # Load the study
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # Load best parameters
    with open(f'best_params_{study_name}.json', 'r') as f:
        best_params = json.load(f)
    
    return study, best_params

def create_visualization(study_name="foraging"):
    # Load study and parameters
    study, best_params = load_study_and_params(study_name)
    
    # Set plotly template
    pio.templates["plotly"].layout["autosize"] = False

    # Create directory for visualizations if it doesn't exist
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(height=400, width=1200)
    fig.write_html("visualizations/optimization_history_1.html")

    # Slice plot
    params = ["mu", "threshold"]
    fig = optuna.visualization.plot_slice(study, params=params)
    fig.write_html("visualizations/optimization_history_2.html")

    # Contour plot
    fig = optuna.visualization.plot_contour(study, params=params)
    fig.update_layout(height=800, width=1200)
    fig.write_html("visualizations/optimization_history_3.html")

    # Parameter importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(height=400, width=1200)
    fig.write_html("visualizations/optimization_history_4.html")

    # Create best model with loaded parameters
    best_exploration_strategy = RandomWalkerExplorationStrategy(
        mu=best_params["mu"],
        dmin=best_params["dmin"],
        L=best_params["L"],
        alpha=best_params["alpha"],
        grid_size=best_params["grid_size"]
    )
    
    best_exploitation_strategy = ExploitationStrategy(
        threshold=best_params["threshold"]
    )
    
    best_model = RandomWalkerModel(
        exploration_strategy=best_exploration_strategy,
        exploitation_strategy=best_exploitation_strategy,
        grid_size=best_params["grid_size"],
        number_of_agents=best_params["num_agents"],
        n_resource_clusters=best_params["n_resource_clusters"],
        resource_quality=best_params["resource_quality"],
        resource_cluster_radius=best_params["resource_cluster_radius"],
        keep_overall_abundance=True,
    )

    # Run best model for data collection
    results = mesa.batch_run(
        RandomWalkerModel,
        parameters={
            "exploration_strategy": best_exploration_strategy,
            "exploitation_strategy": best_exploitation_strategy,
            "grid_size": best_params["grid_size"],
            "number_of_agents": best_params["num_agents"],
            "n_resource_clusters": best_params["n_resource_clusters"],
            "resource_quality": best_params["resource_quality"],
            "resource_cluster_radius": best_params["resource_cluster_radius"],
            "keep_overall_abundance": True,
        },
        iterations=1,
        max_steps=1000,  # You might want to make this configurable
        data_collection_period=1,
    )

    # Convert to DataFrame and analyze
    results_df = pd.DataFrame(results)
    agent_mask = results_df["AgentID"] != 0
    agent_results = results_df[agent_mask]

    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_results["traveled_distance"], bins=20)
    plt.axvline(
        agent_results["traveled_distance"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {agent_results["traveled_distance"].mean():.2f}',
    )
    plt.xlabel("Total Distance Traveled")
    plt.ylabel("Count")
    plt.title("Distribution of Agent Travel Distances (Best Parameters)")
    plt.legend()
    plt.savefig("visualizations/distance_distribution_best.png")
    plt.close()

    # Plot time to first catch distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_results["time_to_first_catch"].dropna(), bins=20)
    plt.axvline(
        agent_results["time_to_first_catch"].dropna().mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {agent_results["time_to_first_catch"].dropna().mean():.2f}',
    )
    plt.xlabel("Steps until First Catch")
    plt.ylabel("Count")
    plt.title("Distribution of Time to First Catch (Best Parameters)")
    plt.legend()
    plt.savefig("visualizations/first_catch_distribution_best.png")
    plt.close()

    # Save a GIF of the agent movement
    save_agent_movement_gif(
        best_model,
        steps=1000,  # You might want to make this configurable
        filename="visualizations/agent_movement.gif",
        resource_cluster_radius=best_params["resource_cluster_radius"],
    )
    print("All visualizations have been saved in the 'visualizations' directory")
    print("Visualization process completed successfully...")

if __name__ == "__main__":
    create_visualization()