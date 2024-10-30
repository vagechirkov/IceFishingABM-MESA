"""
Date 31.10.2023
"""

import numpy as np

from experiment_runner.experiment_runner import run_experiment
from ice_fishing_abm_1.ice_fishing_abm_1.model import Model as model

NAME = "social vs environmental weight vs prior knowledge"
PARAMS = {
    "grid_width": 50,
    "grid_height": 50,
    "number_of_agents": 5,
    "n_resource_clusters": 5,
    "resource_cluster_radius": 3,
    "resource_quality": 0.5,
    "relocation_threshold": 0.1,
    "prior_knowledge_corr": 0,
    "prior_knowledge_noize": 0.1,
    "sampling_length": 5,
    "w_social": 0,
    "w_personal": 1,
    "meso_grid_step": 10,
    "local_search_counter": 5,
    "local_learning_rate": 0.5,
    "meso_learning_rate": 0.5,
}

META_PARAMS = {
    "n_repetitions": 500,
    "n_steps": 1000,
}

for sampling_length in [2, 10]:
    for local_search_counter in [2, 5, 8, 11]:
        for r_quality in [0.3, 0.5]:
            for prior in [0, 1]:
                PARAMS["w_social"] = np.arange(0, 0.105, 0.005)
                PARAMS["w_personal"] = np.arange(0, 1.05, 0.05)
                PARAMS["prior_knowledge_corr"] = prior
                PARAMS["resource_quality"] = r_quality
                PARAMS["sampling_length"] = sampling_length
                PARAMS["local_search_counter"] = local_search_counter
                n = NAME + f"_{prior}_resource_quality_{r_quality}"
                run_experiment(name=n, params=PARAMS, meta_params=META_PARAMS, model=model)
