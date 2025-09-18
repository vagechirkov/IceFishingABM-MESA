import optuna
import pandas as pd
import mesa
from abm.model import Model as SocialInfotaxisModel
from abm.exploration_strategy import SocialInfotaxisExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy
from visualization.visualize_agent_movement import save_agent_movement_gif


def objective(trial):
    """
    The objective function that Optuna will optimize.
    It defines the parameters for the Social Infotaxis model and computes
    the average collected resource based on the exploration and exploitation strategies.
    """

    grid_size = 20
    tau = trial.suggest_float("tau", 0.01, 1.0)  # Softmax temperature
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)  # Exploration probability
    threshold = trial.suggest_int("threshold", 1, 2)

    print("Model type: Social Infotaxis")

    # Set up Social Infotaxis exploration strategy
    exploration_strategy = SocialInfotaxisExplorationStrategy(
        tau=tau, epsilon=epsilon, grid_size=grid_size
    )
    exploitation_strategy = ExploitationStrategy(threshold=threshold)
    model = SocialInfotaxisModel

    # Run the simulation using Mesa's batch_run
    results = mesa.batch_run(
        model,
        parameters={
            "exploration_strategy": exploration_strategy,
            "exploitation_strategy": exploitation_strategy,
            "grid_size": grid_size,
            "number_of_agents": 5,
            "n_resource_clusters": 2,
            "resource_quality": 1.0,
            "resource_cluster_radius": 2,
            "keep_overall_abundance": True,
        },
        iterations=100,
        number_processes=None,  # use all CPUs
        max_steps=100,
        data_collection_period=-1,  # only the last step
    )
    results = pd.DataFrame(results)

    # Filter out agents (resource AgentID is usually 0, so we remove it)
    mask = results.AgentID != 0

    # Calculate the average collected resource
    avg_collected_resource = results.loc[mask, "collected_resource"].mean()

    return avg_collected_resource


if __name__ == "__main__":
    # Create the Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=-1)

    # Print the best trial results
    trial = study.best_trial
    print("Average collected resource: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    # After optimization, generate a GIF with the best parameters
    best_exploration_strategy = SocialInfotaxisExplorationStrategy(
        tau=trial.params["tau"], epsilon=trial.params["epsilon"], grid_size=20
    )
    best_exploitation_strategy = ExploitationStrategy(
        threshold=trial.params["threshold"]
    )
    best_model = SocialInfotaxisModel(
        exploration_strategy=best_exploration_strategy,
        exploitation_strategy=best_exploitation_strategy,
        grid_size=20,
        number_of_agents=5,
        n_resource_clusters=2,
        resource_quality=1.0,
        resource_cluster_radius=2,
        keep_overall_abundance=True,
    )

    # Save a GIF of the agent movement
    save_agent_movement_gif(best_model, steps=100, filename="agent_movement.gif")
    print("Agent Movement Visualization Saved successfully...")
    print("Simulation Completed Successfully...")
