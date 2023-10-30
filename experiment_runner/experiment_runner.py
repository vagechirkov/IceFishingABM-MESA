from datetime import datetime
import logging
import os
from pathlib import Path

import mesa
import numpy as np
import pandas as pd

from visualization.plot_params import plot_two_params, prepare_heatmap


def _make_folder_name(name: str) -> str:
    folder_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + "_" + name.lower().replace(" ", "_")
    return folder_name


def run_experiment(name, params, meta_params, model):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Running experiment 🐳: {name}")
    logging.info(f"Parameters 🦓: {pd.Series(params).to_string()}")
    logging.info(f"Meta-parameters 🦓: {pd.Series(meta_params).to_string()}")

    # create folder with the name of the experiment and the date
    results_path = Path(_make_folder_name(name))
    results_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved in {results_path}")

    # save the parameters (dict) as a json file in the folder
    json_params = pd.Series(params).to_json()
    with open(results_path / "parameters.json", "w") as f:
        f.write(json_params)
    logging.info(f"Parameters saved in {results_path / 'parameters.json'}")

    # save the meta-parameters (dict) as a json file in the folder
    json_meta_params = pd.Series(meta_params).to_json()
    with open(results_path / "meta_parameters.json", "w") as f:
        f.write(json_meta_params)
    logging.info(f"Meta-parameters saved in {results_path / 'meta_parameters.json'}")

    # Dry run to make sure the experiment is set up correctly
    cpu_count = os.cpu_count() - 1
    logging.info(f"Running on {cpu_count} cores")
    _ = mesa.batch_run(
        model,
        parameters=params,
        iterations=1,
        max_steps=meta_params["n_steps"],
        number_processes=cpu_count,
        data_collection_period=-1,
        display_progress=True,
    )
    logging.info("Dry run successful 🦜")

    # Run the experiment
    results = mesa.batch_run(
        model,
        parameters=params,
        iterations=meta_params["n_repetitions"],
        max_steps=meta_params["n_steps"],
        number_processes=cpu_count,
        data_collection_period=-1,
        display_progress=True,
    )
    logging.info("Experiment run successful 🦨")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path / "results.csv", mode='a')
    logging.info(f"Results saved in {results_path / 'results.csv'}")

    # Plot and save plots
    # find all the iterable parameters
    iterable_params = [k for k, v in params.items() if isinstance(v, (list, np.ndarray))]
    logging.info(f"Iterable parameters: {iterable_params}")

    # plot the results for each combination of the iterable parameters
    for p1 in iterable_params:
        for p2 in iterable_params:
            if p1 == p2:
                continue
            logging.info(f"Plotting {p1} vs. {p2}")
            heatmap_df = prepare_heatmap(results_df, p1, p2, meta_params["n_steps"])
            plot_two_params(heatmap_df, p1, p2, meta_params["n_repetitions"], results_path)

    logging.info("Plots are saved 🦚")
