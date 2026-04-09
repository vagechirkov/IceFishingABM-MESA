import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from abm.model import IceFishingModel
from mesa.batchrunner import batch_run

def run_simulations_social_info_abundance(n_repetitions: int = 1):
    rng = np.random.default_rng(42)
    rng_values = rng.integers(0, sys.maxsize, size=(n_repetitions,))

    params = {
        "grid_size": 90,
        "number_of_agents": 6,
        "simulation_length_minutes": 180,
        "fish_abundance": [(2.0, 3.0)],
        "drilling_time_cost_minutes": [1.0],
        "spot_selection_tau": [0.1, 0.15, 0.2, 0.25, 0.3],
        "spot_selection_w_locality": [0.0],
        "spot_selection_weights": [
            (0.0, 0.5, 0.5),
            (0.2, 0.4, 0.4),
            (0.4, 0.3, 0.3),
            (0.6, 0.2, 0.2),
            (0.8, 0.1, 0.1)
        ],
        "spot_selection_social_info_quality": ["sampling", "consuming"]
    }

    max_steps = params["simulation_length_minutes"] * 6

    results = batch_run(
        IceFishingModel,
        parameters=params,
        rng=rng_values.tolist(),
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ice fishing simulations and save results.")
    parser.add_argument(
        "n_repetitions",
        nargs="?",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(".")
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default=""
    )

    chunk_size = 200
    args = parser.parse_args()
    total_repetitions = args.n_repetitions
    args.outdir.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%d.%m.%Y")
    file_counter = 1

    for start_rep in range(0, total_repetitions, chunk_size):
        reps_in_this_chunk = min(chunk_size, total_repetitions - start_rep)
        df = run_simulations_social_info_abundance(reps_in_this_chunk)

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        fname = args.outdir / f"ice_fishing_simulations_{today_str}_{args.suffix}_{file_counter}.csv"
        df.to_csv(fname, index=False)
        file_counter += 1