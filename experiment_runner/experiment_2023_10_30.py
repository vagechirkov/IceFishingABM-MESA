import datetime
import logging
import os
from pathlib import Path

import mesa
import numpy as np
import pandas as pd

from ice_fishing_abm_1.ice_fishing_abm_1.model import Model as model
from visualization.plot_params import plot_two_params, prepare_heatmap

logging.basicConfig(level=logging.INFO)

NAME = "Resources quality vs. local search properties"
PARAMS = {
    "grid_width": 50,
    "grid_height": 50,
    "number_of_agents": 5,
    "n_resource_clusters": 5,
    "resource_cluster_radius": 10,
    "relocation_threshold": 0.1,
    "prior_knowledge_corr": 0,
    "prior_knowledge_noize": 0.1,
    "sampling_length": np.arange(1, 10, 1),
    "w_social": 0,
    "w_personal": 1,
    "meso_grid_step": 10,
    "local_search_counter": np.arange(1, 10, 1)
}

META_PARAMS = {
    "n_repetitions": 500,
    "n_steps": 1000,
}

logging.info(f"Running experiment 🐳: {NAME} ")
logging.info(f"Parameters 🦓: {PARAMS}")
logging.info(f"Meta-parameters 🦓: {META_PARAMS}")

# create folder with the name of the experiment and the date
today = datetime.date.today()
folder_name = today.strftime("%Y-%m-%d") + "_" + NAME.lower().replace(" ", "_")
results_path = Path(folder_name)
results_path.mkdir(parents=True, exist_ok=True)
logging.info(f"Results will be saved in {results_path}")

# save the parameters (dict) as a json file in the folder
json_params = pd.Series(PARAMS).to_json()
with open(results_path / "parameters.json", "w") as f:
    f.write(json_params)
logging.info(f"Parameters saved in {results_path / 'parameters.json'}")

# save the meta-parameters (dict) as a json file in the folder
json_meta_params = pd.Series(META_PARAMS).to_json()
with open(results_path / "meta_parameters.json", "w") as f:
    f.write(json_meta_params)
logging.info(f"Meta-parameters saved in {results_path / 'meta_parameters.json'}")

# Dry run to make sure the experiment is set up correctly
cpu_count = os.cpu_count() - 1
logging.info(f"Running on {cpu_count} cores")
_ = mesa.batch_run(
    model,
    parameters=PARAMS,
    iterations=1,
    max_steps=META_PARAMS["n_steps"],
    number_processes=cpu_count,
    data_collection_period=-1,
    display_progress=True,
)
logging.info("Dry run successful 🦜")

# Run the experiment
results = mesa.batch_run(
    model,
    parameters=PARAMS,
    iterations=META_PARAMS["n_repetitions"],
    max_steps=META_PARAMS["n_steps"],
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
iterable_params = [k for k, v in PARAMS.items() if isinstance(v, (list, np.ndarray))]
logging.info(f"Iterable parameters: {iterable_params}")

# plot the results for each combination of the iterable parameters
for p1 in iterable_params:
    for p2 in iterable_params:
        if p1 == p2:
            continue
        logging.info(f"Plotting {p1} vs. {p2}")
        heatmap_df = prepare_heatmap(results_df, p1, p2, META_PARAMS["n_steps"])
        plot_two_params(heatmap_df, p1, p2, META_PARAMS["n_repetitions"], results_path)

logging.info("Plots are saved 🦚")
