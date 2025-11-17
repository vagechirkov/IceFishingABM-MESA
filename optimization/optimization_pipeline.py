import logging
import sys

import optuna


import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from abm.model import IceFishingModel
from mesa.batchrunner import batch_run


def objective(trial):
    """
    Optuna objective function to optimize the IceFishingModel parameters.
    """
    fish_abundance=3.0
    n_iterations=20
    trial.set_user_attr("n_iterations", n_iterations)

    # Spot Selection Weights
    _w_social = trial.suggest_float("_w_social", 0.0, 1.0)
    _w_success = trial.suggest_float("_w_success", 0.0, 1.0)
    _w_failure = trial.suggest_float("_w_failure", 0.0, 1.0)
    _w_locality = trial.suggest_float("_w_locality", 0.0, 1.0)

    # Add a small epsilon to prevent division by zero if all are 0
    total_w = _w_social + _w_success + _w_failure + _w_locality + 1e-6

    w_social = _w_social / total_w
    w_success = _w_success / total_w
    w_failure = _w_failure / total_w
    w_locality = _w_locality / total_w

    # Store the normalized weights as user attributes
    trial.set_user_attr("spot_selection_w_social", w_social)
    trial.set_user_attr("spot_selection_w_success", w_success)
    trial.set_user_attr("spot_selection_w_failure", w_failure)
    trial.set_user_attr("spot_selection_w_locality", w_locality)
    trial.set_user_attr("fish_abundance", fish_abundance)

    tau = 1.0
    trial.set_user_attr("spot_selection_tau", tau)

    # Spot Leaving Weights (Logistic regression on logit scale)
    # Default: -3
    baseline_weight = trial.suggest_float("spot_leaving_baseline_weight", -7.0, -1.0)

    # Default: -1.7 (Catching fish should make you less likely to leave)
    fish_catch_weight = trial.suggest_float("spot_leaving_fish_catch_weight", -5.0, 0.0)

    # Default: 0.13 (More time should make you more likely to leave)
    time_weight = trial.suggest_float("spot_leaving_time_weight", 0.0, 0.5)

    # Default: -0.33 (Social feature, range can be wider)
    social_weight = trial.suggest_float("spot_leaving_social_weight", -2, 1.0)


    params = {
        "grid_size": 90,
        "number_of_agents": 6,
        "simulation_length_minutes": 180,
        "fish_abundance": fish_abundance,

        "spot_selection_w_social": w_social,
        "spot_selection_w_success": w_success,
        "spot_selection_w_failure": w_failure,
        "spot_selection_w_locality": w_locality,

        "spot_selection_tau": tau,

        # Optimized Spot Leaving Weights
        "spot_leaving_baseline_weight": baseline_weight,
        "spot_leaving_fish_catch_weight": fish_catch_weight,
        "spot_leaving_time_weight": time_weight,
        "spot_leaving_social_weight": social_weight,
    }

    max_steps = params["simulation_length_minutes"] * 6

    # Run Batch Simulation
    try:
        results = batch_run(
            IceFishingModel,
            parameters=params,
            iterations=n_iterations,  # Number of iterations per trial
            max_steps=max_steps,
            number_processes=None,
            data_collection_period=-1,
            display_progress=False,
        )

        if not results:
            # Handle cases where the simulation might fail or return no data
            return optuna.TrialPruned()

        results_df = pd.DataFrame(results)

        # Objective Value
        # We want to maximize the median catch across iterations
        median_catch = results_df['catch'].median()

        # Handle potential NaN or empty results
        if pd.isna(median_catch):
            return 0.0 # or optuna.TrialPruned()

        return median_catch

    except Exception as e:
        print(f"Trial failed with exception: {e}")
        # Tell Optuna to prune this trial if it fails
        return optuna.TrialPruned()



def run_optimization(db_name='ice_fishing_model_11_25', n_trials=100):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = db_name  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of optimization trials to run."
    )

    parser.add_argument(
        "--db_name",
        type=str,
        default="ice_fishing_study",
        help="Name of the study (and .db file), e.g., 'ice_fishing_v1'"
    )

    args = parser.parse_args()

    print(f"--- Starting Optuna Optimization ---")
    print(f"Study Name: {args.db_name}")
    print(f"Trials to run: {args.n_trials}")
    print(f"Storage: sqlite:///{args.db_name}.db")

    run_optimization(db_name=args.db_name, n_trials=args.n_trials)
