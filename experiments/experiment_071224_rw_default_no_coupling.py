import optuna
import pandas as pd
import mesa
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy
import json


# 5 resources with radius 2 instead of current values, agents : 5,10,50; grid_size =  100.

"""
RUN HYPERPARAMETERS GO HERE:
"""

NUM_AGENTS     = 10                   # Number of agents
D_MIN          = 1                    # Minimum distance for Lévy flight
max_sim_steps  = 1000                 # Maximum number of steps
GRID_SIZE      = 100                  # Grid size for simulation
MAX_L          = GRID_SIZE            # Maximum distance for Lévy flight
NUM_ITERATIONS = 1                    # Number of iterations
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
    L = MAX_L  # Maximum distance for Lévy flight
    dmin = D_MIN  # Minimum distance for Lévy flight

    # Actual hyperparameters

    mu = trial.suggest_float("mu", 1.1, 2.1)  # Exponent for Lévy flight
    # alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)  # Parameter for social cue adjustment
    alpha = ALPHA
    threshold = trial.suggest_categorical("threshold", [1, 2])

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

    study = optuna.create_study(direction="maximize", storage=storage_name)

    study.optimize(objective, n_trials=5, n_jobs=1)

    # Print the best trial results
    trial = study.best_trial
    print("Average collected resource: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


    # Save the best model parameters and results
    best_params = {
        'mu': trial.params["mu"],
        'threshold': trial.params["threshold"],
        'dmin': D_MIN,
        'L': MAX_L,
        'alpha': ALPHA,
        'grid_size': GRID_SIZE,
        'num_agents': NUM_AGENTS,
        'n_resource_clusters': NUM_RESOURCE_CLUSTERS,
        'resource_quality': RESOURCE_QUALITY,
        'resource_cluster_radius': RESOURCE_CLUSTER_RADIUS
    }

    # Save the best model parameters in a JSON file

    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)

    