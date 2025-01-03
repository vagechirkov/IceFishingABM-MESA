import optuna
import pandas as pd
import mesa
import matplotlib.pyplot as plt
import plotly.io as pio

# Import RandomWalker-specific models and strategies
# NOTE: Make sure to add the project root to the Python path:
# export PYTHONPATH=/path/to/myproject:$PYTHONPATH
# this line can be added to the shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.) so that it’s permanent for the
# terminal environment
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy
from visualization.visualize_agent_movement import save_agent_movement_gif


# 5 resources with radius 2 instead of current values, agents : 5,10,50; grid_size =  100.

"""
RUN HYPERPARAMETERS GO HERE:
"""

NUM_AGENTS     = 3                    # Number of agents
D_MIN          = 1                    # Minimum distance for Levy flight
max_sim_steps  = 100                  # Maximum number of steps
GRID_SIZE      = 20                   # Grid size for simulation
MAX_L          = GRID_SIZE / 2        # Maximum distance for Levy flight
NUM_ITERATIONS = 10                   # Number of iterations
ALPHA          = 1e-5                 # Parameter for social cue coupling 
NUM_RESOURCE_CLUSTERS = 5             # Number of resource clusters
RESOURCE_CLUSTER_RADIUS = 2           # Radius of resource clusters    
RESOURCE_QUALITY = 1.0                # Quality of resources    


def objective(trial):
    """
    The objective function that Optuna will optimize.
    It defines the parameters for both GP and Random Walker models
    and computes the average collected resource based on the exploration and exploitation strategies.
    """

    grid_size = GRID_SIZE
    L = MAX_L  # Maximum distance for Levy flight
    dmin = D_MIN  # Minimum distance for Levy flight

    # Actual hyperparameters

    mu = trial.suggest_float("mu", 1.1, 2.1)  # Exponent for Levy flight
    # alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)  # Parameter for social cue adjustment
    alpha = ALPHA
    threshold = trial.suggest_int("threshold", 1, 2)

    print("Model type: Random Walker")

    # Set up Random Walker exploration strategy
    exploration_strategy = RandomWalkerExplorationStrategy(
        mu=mu,
        dmin=dmin,
        L=L,
        alpha=alpha,
        grid_size=grid_size,
    )
    exploitation_strategy = ExploitationStrategy(threshold=threshold)
    model = RandomWalkerModel

    # Run the simulation using Mesa's batch_run
    results = mesa.batch_run(
        model,
        parameters={
            "exploration_strategy": exploration_strategy,
            "exploitation_strategy": exploitation_strategy,
            "grid_size": grid_size,
            "number_of_agents": NUM_AGENTS,
            "n_resource_clusters": NUM_RESOURCE_CLUSTERS,
            "resource_quality": 1.0,
            "resource_cluster_radius": RESOURCE_CLUSTER_RADIUS,
            "keep_overall_abundance": True,
        },
        iterations=NUM_ITERATIONS,
        number_processes=None,  # use all CPUs
        max_steps=max_sim_steps,
        data_collection_period=-1,  # only the last step
    )
    results = pd.DataFrame(
        results
    )  ### MAKE THIS SCALE FREE BY NORMALISING WITH NUM OF STEPS

    # Filter out agents (resource AgentID is usually 0, so we remove it)
    mask = results.AgentID != 0

    agent_metrics = (
        results.groupby("AgentID")
        .agg(
            {
                "collected_resource": "max",
                "traveled_distance": "max",
                "time_to_first_catch": "first",
            }
        )
        .reset_index()
    )

    # Calculate efficiency (resource/distance)
    agent_metrics["efficiency"] = agent_metrics["collected_resource"] / (
        agent_metrics["traveled_distance"] + 1e-10
    )

    # Use mean efficiency as objective
    avg_efficiency = agent_metrics["efficiency"].mean()

    return avg_efficiency


if __name__ == "__main__":
    # Create the Optuna study and optimize the objective function
    study_name = "foraging-db"  # Unique identifier of the study
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(),
        storage=storage_name,
    )

    study.optimize(objective, n_trials=50, n_jobs=1)

    # Print the best trial results
    trial = study.best_trial
    print("Average collected resource: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    # Visualization: Plot the optimization history

    pio.templates["plotly"].layout["autosize"] = False

    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(height=400, width=1200)
    fig.show()

    # Slice plot (visualize the effects of individual parameters)
    params = ["mu", "threshold"]

    fig = optuna.visualization.plot_slice(study, params=params)
    fig.show()

    # Contour plot (visualize parameter interactions)
    fig = optuna.visualization.plot_contour(study, params=params)
    fig.update_layout(height=800, width=1200)
    fig.show()

    # Parameter importances (visualize which parameters contribute most to the objective)
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(height=400, width=1200)
    fig.show()

    # After optimization, generate a GIF with the best parameters
    best_exploration_strategy = RandomWalkerExplorationStrategy(
        mu=trial.params["mu"], dmin=D_MIN, L=MAX_L, alpha=ALPHA, grid_size=GRID_SIZE
    )
    best_exploitation_strategy = ExploitationStrategy(
        threshold=trial.params["threshold"]
    )
    best_model = RandomWalkerModel(
        exploration_strategy=best_exploration_strategy,
        exploitation_strategy=best_exploitation_strategy,
        grid_size=GRID_SIZE,
        number_of_agents=NUM_AGENTS,
        n_resource_clusters=NUM_RESOURCE_CLUSTERS,
        resource_quality=RESOURCE_QUALITY,
        resource_cluster_radius=RESOURCE_CLUSTER_RADIUS,
        keep_overall_abundance=True,
    )

    # Run best model for data collection
    results = mesa.batch_run(
        RandomWalkerModel,
        parameters={
            "exploration_strategy": best_exploration_strategy,
            "exploitation_strategy": best_exploitation_strategy,
            "grid_size": GRID_SIZE,
            "number_of_agents": NUM_AGENTS,
            "n_resource_clusters": NUM_RESOURCE_CLUSTERS,
            "resource_quality": RESOURCE_QUALITY,
            "resource_cluster_radius": RESOURCE_CLUSTER_RADIUS,
            "keep_overall_abundance": True,
        },
        iterations=1,  # Single run for analysis
        max_steps=max_sim_steps,
        data_collection_period=1,  # Collect at every step
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
    plt.savefig("distance_distribution_best.png")
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
    plt.savefig("first_catch_distribution_best.png")
    plt.close()

    # Save a GIF of the agent movement
    save_agent_movement_gif(
        best_model,
        steps=max_sim_steps,
        filename="agent_movement.gif",
        resource_cluster_radius=2,
    )
    print("Agent Movement Visualization Saved successfully...")
    print("Simulation Completed Successfully...")
