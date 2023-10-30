import numpy as np

from experiment_runner.experiment_runner import run_experiment
from ice_fishing_abm_1.ice_fishing_abm_1.model import Model as model

NAME = "Resources quality vs local search properties"
PARAMS = {
    "grid_width": 50,
    "grid_height": 50,
    "number_of_agents": 5,
    "n_resource_clusters": 5,
    "resource_cluster_radius": 10,
    "relocation_threshold": 0.1,
    "prior_knowledge_corr": 0,
    "prior_knowledge_noize": 0.1,
    "sampling_length": np.arange(1, 3, 1),
    "w_social": 0,
    "w_personal": 1,
    "meso_grid_step": 10,
    "local_search_counter": np.arange(1, 3, 1)
}

META_PARAMS = {
    "n_repetitions": 500,
    "n_steps": 1000,
}

run_experiment(name=NAME, params=PARAMS, meta_params=META_PARAMS, model=model)
