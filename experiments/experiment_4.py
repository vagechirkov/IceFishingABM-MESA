"""
Date 7.11.2023
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
    "resource_quality": 0.5,
    "resource_cluster_radius": 2,
    "sampling_length": 5,
    "relocation_threshold": 0.1,
    "meso_grid_step": 10,
    "local_search_counter": 5,
    "w_social": 0.03,
    "w_personal": 0.97,
    "prior_knowledge_corr": 0,
    "prior_knowledge_noize": 0.1,
    "local_learning_rate": 0.5,
    "meso_learning_rate": 0.5,
    "social_learning_rate": 0.5,
}

META_PARAMS = {
    "n_repetitions": 200,
    "n_steps": 1000,
}


for prior in [1, 0]:
    PARAMS["w_social"] = np.arange(0, 0.1, 0.005)
    PARAMS["w_personal"] = np.arange(0, 1.05, 0.05)
    PARAMS["prior_knowledge_corr"] = prior
    PARAMS["resource_quality"] = 0.8
    PARAMS["sampling_length"] = 5
    PARAMS["local_search_counter"] = 5
    PARAMS["social_learning_rate"] = 1
    n = NAME + f"_{prior}_resource_quality_{PARAMS['resource_quality']}"
    run_experiment(name=n, params=PARAMS, meta_params=META_PARAMS, model=model)
