import datetime

import mesa
import numpy as np
import pandas as pd

from ice_fishing_abm_1.model import Model as model

if __name__ == "__main__":
    params = {
        "grid_width": 50,
        "grid_height": 50,
        "number_of_agents": 5,
        "n_resource_clusters": 5,
        "resource_cluster_radius": 10,
        "relocation_threshold": 0.1,
        "prior_knowledge_corr": 0,
        "prior_knowledge_noize": 0.1,
        "sampling_length": 2,
        "w_social": np.arange(0, 0.5, 0.01),
        "w_personal": np.arange(0, 1, 0.05),
        "meso_grid_step": 10,
        "local_search_counter": 5
    }

    n_repetitions = 500
    n_steps = 1000

    results = mesa.batch_run(
        model,
        parameters=params,
        iterations=n_repetitions,
        max_steps=n_steps,
        number_processes=70,
        data_collection_period=-1,
        display_progress=True,
    )

    results_df = pd.DataFrame(results)

    # save dataset
    today = datetime.date.today()
    df_name = today.strftime("%Y-%m-%d")
    results_df.to_csv(f"{df_name}.csv", mode='a')
