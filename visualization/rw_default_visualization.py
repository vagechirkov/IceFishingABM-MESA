import optuna
import pandas as pd
import mesa
import matplotlib.pyplot as plt
import plotly.io as pio
import json
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy
from visualization.visualize_agent_movement import save_agent_movement_gif
import numpy as np


def load_study_and_params(study_name, storage_name="foraging_cloud"):
    """Load the saved study and best parameters"""
    # Load the study
    storage_name = f"sqlite:///{storage_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # Load best parameters
    with open(f'best_params_{study_name}.json', 'r') as f:
        best_params = json.load(f)
    
    return study, best_params

def create_visualization(study_name):
    # Load study and parameters
    
    study, best_params = load_study_and_params(study_name)
    
    print("Best hyperparams loaded for study:")
    # Set plotly template
    pio.templates["plotly"].layout["autosize"] = False

    # Create directory for visualizations if it doesn't exist
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(height=400, width=1200)
    fig.write_html(f"visualizations/optimization_history_1_{study_name}.html")

    # Slice plot
    
    if study_name == 'rw-default-no-coupling': 
        params = ["mu", "threshold"]
    else: 
        params = ["alpha" ,"mu"]
    fig = optuna.visualization.plot_slice(study, params=params)
    fig.write_html(f"visualizations/optimization_history_2_{study_name}.html")

    # Contour plot
    fig = optuna.visualization.plot_contour(study, params=params)
    fig.update_layout(height=800, width=1200)
    fig.write_html(f"visualizations/optimization_history_3_{study_name}.html")

    # Parameter importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(height=400, width=1200)
    fig.write_html(f"visualizations/optimization_history_4_{study_name}.html")

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
        iterations=1_000,
        max_steps=1000,  # You might want to make this configurable
        data_collection_period=-1,
    )

    # Convert to DataFrame and analyze
    results_df = pd.DataFrame(results)
    agent_mask = results_df["AgentID"] != 0
    agent_results = results_df[agent_mask] 

    # print values
    print(np.array(agent_results["time_to_first_catch"]))
    print(np.array(agent_results["traveled_distance"]))

    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_results["traveled_distance"], bins='auto')
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
    plt.savefig(f"visualizations/distance_distribution_{study_name}.png")
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
    #plt.savefig(f"visualizations/first_catch_distribution_{study_name}.png")
    plt.close()


    # Create scatter plot of distance vs time to first catch
    plt.figure(figsize=(10, 6))
    
    # Clean data - remove NaN values
    plot_data = agent_results[["traveled_distance", "time_to_first_catch"]].dropna()
    
    # Plot the basic scatter
    plt.scatter(
        plot_data["traveled_distance"],
        plot_data["time_to_first_catch"],
        alpha=0.6,
        c='blue',
        label='Agents'
    )
    
    # Add trend line and correlation only if we have enough valid points
    if len(plot_data) > 2:  # Need at least 3 points for meaningful statistics
        try:
            # Add trend line
            z = np.polyfit(plot_data["traveled_distance"], 
                          plot_data["time_to_first_catch"], 1)
            p = np.poly1d(z)
            plt.plot(plot_data["traveled_distance"], 
                    p(plot_data["traveled_distance"]), 
                    "r--", alpha=0.8,
                    label=f'Trend line (slope: {z[0]:.2f})')
            
            # Add correlation coefficient
            corr = plot_data["traveled_distance"].corr(plot_data["time_to_first_catch"])
            plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        except np.linalg.LinAlgError:
            print("Could not compute trend line - numerical instability")
        except Exception as e:
            print(f"Could not add statistical analysis: {str(e)}")
    
    plt.xlabel("Distance Traveled")
    plt.ylabel("Time to First Catch")
    plt.title("Distance Traveled vs Time to First Catch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"visualizations/distance_vs_time_{study_name}.png")
    plt.close()


    # Plot sampling time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_results["total_sampling_time"], bins=20)
    plt.axvline(
        agent_results["total_sampling_time"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {agent_results["total_sampling_time"].mean():.2f}',
    )
    plt.xlabel("Total Sampling Time")
    plt.ylabel("Count")
    plt.title("Distribution of Agent Sampling Times")
    plt.legend()
    plt.savefig(f"visualizations/sampling_time_distribution_{study_name}.png")
    plt.close()

    # Plot consuming time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_results["total_consuming_time"], bins=20)
    plt.axvline(
        agent_results["total_consuming_time"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {agent_results["total_consuming_time"].mean():.2f}',
    )
    plt.xlabel("Total Consuming Time")
    plt.ylabel("Count")
    plt.title("Distribution of Agent Consuming Times")
    plt.legend()
    plt.savefig(f"visualizations/consuming_time_distribution_{study_name}.png")
    plt.close()

    # Plot cluster catches distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_results["cluster_catches"], bins=20)
    plt.axvline(
        agent_results["cluster_catches"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {agent_results["cluster_catches"].mean():.2f}',
    )
    plt.xlabel("Number of Cluster Catches")
    plt.ylabel("Count")
    plt.title("Distribution of Cluster Catches per Agent")
    plt.legend()
    plt.savefig(f"visualizations/cluster_catches_distribution_{study_name}.png")
    plt.close()

    # Create scatter plots comparing metrics
    metrics_pairs = [
        ("traveled_distance", "collected_resource", "Distance vs Resources"),
        ("total_sampling_time", "collected_resource", "Sampling Time vs Resources"),
        ("total_consuming_time", "collected_resource", "Consuming Time vs Resources"),
        ("cluster_catches", "collected_resource", "Cluster Catches vs Resources")
    ]

    for x_metric, y_metric, title in metrics_pairs:
        plt.figure(figsize=(10, 6))
        plot_data = agent_results[[x_metric, y_metric]].dropna()
        
        if len(plot_data) > 0:
            plt.scatter(
                plot_data[x_metric],
                plot_data[y_metric],
                alpha=0.6,
                c='blue',
                label='Agents'
            )
            
            # Add trend line if enough points
            if len(plot_data) > 2:
                try:
                    z = np.polyfit(plot_data[x_metric], plot_data[y_metric], 1)
                    p = np.poly1d(z)
                    plt.plot(
                        plot_data[x_metric],
                        p(plot_data[x_metric]),
                        "r--",
                        alpha=0.8,
                        label=f'Trend (slope: {z[0]:.2f})'
                    )
                    
                    # Add correlation coefficient
                    corr = plot_data[x_metric].corr(plot_data[y_metric])
                    plt.text(
                        0.05, 0.95,
                        f'Correlation: {corr:.2f}',
                        transform=plt.gca().transAxes,
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                except Exception as e:
                    print(f"Could not add trend line for {title}: {str(e)}")
        
        plt.xlabel(x_metric.replace('_', ' ').title())
        plt.ylabel(y_metric.replace('_', ' ').title())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"visualizations/{x_metric}_vs_{y_metric}_{study_name}.png")
        plt.close()

    # Save a GIF of the agent movement
    save_agent_movement_gif(
        best_model,
        steps=1000,  # You might want to make this configurable
        filename=f"visualizations/agent_movement_{study_name}.gif",
        resource_cluster_radius=best_params["resource_cluster_radius"],
    )
    print("All visualizations have been saved in the 'visualizations' directory")
    print("Visualization process completed successfully...")

if __name__ == "__main__":
    #create_visualization("rw-default-no-coupling")
    #create_visualization(study_name="rw-default-all-agents")
    create_visualization(study_name="rw-default-filtering")