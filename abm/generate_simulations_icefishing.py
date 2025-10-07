import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from abm.model import IceFishingModel
from mesa.batchrunner import batch_run


def make_prior(seed: int | None = None):
    """
    Returns a zero-arg callable that samples a full parameter set from the prior.
    - Weights ~ Dirichlet(alpha=[1,1,1])  -> sum to 1
    - spot_selection_tau ~ Uniform(0.01, 1.0)
    - spot_leaving_time_weight ~ Uniform(0.1, 1.0)
    - fish_abundance ~ Uniform(2.0, 4.0)
    """
    rng = np.random.default_rng(seed)

    def sample():
        w_social, w_success, w_failure, w_locality = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        return {
            "spot_selection_w_social": float(w_social),
            "spot_selection_w_success": float(w_success),
            "spot_selection_w_failure": float(w_failure),
            "spot_selection_w_locality": float(w_locality),
            "spot_selection_tau": 0.1,  # float(rng.uniform(0.01, 1.0)),
            "spot_leaving_baseline_weight": float(rng.uniform(-10, 0)),
            "spot_leaving_fish_catch_weight": float(rng.uniform(-10, 0)),
            "spot_leaving_time_weight": float(rng.uniform(0.1, 2.0)),
            "spot_leaving_social_weight": float(rng.uniform(-2.0, 2.0)),
            "fish_abundance": float(rng.uniform(2.0, 4.0)),  #  3.5  #
        }

    return sample

def generate_sbi_simulations(n_simulations: int):
    prior = make_prior(seed=None)

    all_priors = [prior() for _ in range(n_simulations)]

    params = {
        "grid_size": 90,
        "number_of_agents": 6,
        "simulation_length_minutes": 180,
        "sample_from_prior": all_priors,
    }

    max_steps = params["simulation_length_minutes"] * 6

    results = batch_run(
        IceFishingModel,
        parameters=params,
        iterations=1,
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ice fishing simulations and save results.")
    parser.add_argument(
        "n_simulations",
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

    df = generate_sbi_simulations(args.n_simulations)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%d.%m.%Y")
    fname = args.outdir / f"ice_fishing_simulations_{today_str}_{args.suffix}.csv"
    df.to_csv(fname, index=False)

    print(f"Saved {len(df)} rows to: {fname}")