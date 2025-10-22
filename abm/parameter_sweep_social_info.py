import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from abm.model import IceFishingModel
from mesa.batchrunner import batch_run

def run_simulations_social_info_abundance(n_repetitions: int = 1):
    params = {
        "grid_size": 90,
        "number_of_agents": 6,
        "simulation_length_minutes": 180,
        "fish_abundance": [3.0, 3.5, 4.0],
        "spot_selection_w_social": [0.042, 0.2, 0.8, 2.4]  # 1:4, 2:4, 3:4 ratio of soc to private weights
    }

    max_steps = params["simulation_length_minutes"] * 6

    results = batch_run(
        IceFishingModel,
        parameters=params,
        iterations=n_repetitions,
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

    args = parser.parse_args()

    df = run_simulations_social_info_abundance(args.n_repetitions)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%d.%m.%Y")
    fname = args.outdir / f"ice_fishing_simulations_{today_str}_{args.suffix}.csv"
    df.to_csv(fname, index=False)

    print(f"Saved {len(df)} rows to: {fname}")