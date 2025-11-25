import logging
import sys

import optuna


import argparse
import pandas as pd
from functools import partial
from abm.model import IceFishingModel
from mesa.batchrunner import batch_run


def objective(trial, fish_abundance=3.0, tau=0.1, suggest_slw=True):
    """
    Optuna objective function to optimize the IceFishingModel parameters.
    """
    n_iterations = 20

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

    spot_selection_social_length_scale = trial.suggest_float("ssw_soc_ls", 0.5, 50.0)
    spot_selection_success_length_scale = trial.suggest_float("ssw_suc_ls", 0.5, 50.0)
    spot_selection_failure_length_scale = trial.suggest_float("ssw_fail_ls", 0.5, 50.0)

    # Spot Leaving Weights (Logistic regression on logit scale)
    if suggest_slw:
        # Default: -3
        baseline_weight = trial.suggest_float("slw_base", -20.0, -1.0)

        # Default: -1.7 (Catching fish should make you less likely to leave)
        fish_catch_weight = trial.suggest_float("slw_fish", -15.0, 0.0)

        # Default: 0.13 (More time should make you more likely to leave)
        time_weight = trial.suggest_float("slw_time", 0.0, 2.0)

        # Default: -0.33 (Social feature, range can be wider)
        social_weight = trial.suggest_float("slw_soc", -3, 3.0)
    else:
        baseline_weight = -3.0
        fish_catch_weight = -1.7
        time_weight = 0.13
        social_weight = -0.33

    # Store the normalized weights as user attributes
    trial.set_user_attr("n_iterations", n_iterations)
    trial.set_user_attr("fish_abundance", fish_abundance)

    trial.set_user_attr("ssw_soc", w_social)
    trial.set_user_attr("ssw_suc", w_success)
    trial.set_user_attr("ssw_fail", w_failure)
    trial.set_user_attr("ssw_loc", w_locality)
    trial.set_user_attr("ss_tau", tau)

    params = {
        "grid_size": 90,
        "number_of_agents": 6,
        "simulation_length_minutes": 180,
        "fish_abundance": fish_abundance,

        "spot_selection_w_social": w_social,
        "spot_selection_w_success": w_success,
        "spot_selection_w_failure": w_failure,
        "spot_selection_w_locality": w_locality,
        "spot_selection_social_length_scale": spot_selection_social_length_scale,
        "spot_selection_success_length_scale": spot_selection_success_length_scale,
        "spot_selection_failure_length_scale": spot_selection_failure_length_scale,

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
        median_catch = results_df['median_catch'].median()

        # Handle potential NaN or empty results
        if pd.isna(median_catch):
            return 0.0 # or optuna.TrialPruned()

        return median_catch

    except Exception as e:
        print(f"Trial failed with exception: {e}")
        # Tell Optuna to prune this trial if it fails
        return optuna.TrialPruned()



def run_optimization(db_name, study_name, n_trials=100, fish_abundance=3.0, tau=0.1, suggest_slw=True):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(db_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        # sampler=optuna.samplers.RandomSampler()
        # sampler=optuna.samplers.TPESampler()
        sampler=optuna.samplers.CmaEsSampler(n_startup_trials=500)
    )

    objective_with_params = partial(objective, fish_abundance=fish_abundance, tau=tau, suggest_slw=suggest_slw)
    study.optimize(objective_with_params, n_trials=n_trials, n_jobs=5)


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

    parser.add_argument(
        "--abundance",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default=""
    )

    parser.add_argument(
        "--suggest_slw",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    args = parser.parse_args()

    print("--- Starting Optuna Optimization ---")
    print(f"Study Name: {args.db_name}{args.suffix}_{args.abundance}_{args.tau}_slw_{args.suggest_slw}")
    print(f"Trials to run: {args.n_trials}")
    print(f"Storage: sqlite:///{args.db_name}.db")

    study_name = f"{args.db_name}{args.suffix}_{args.abundance}_{args.tau}"
    if not args.suggest_slw:
        study_name += f"_slw_{args.suggest_slw}"

    run_optimization(db_name=f"{args.db_name}",
                     study_name=study_name,
                     n_trials=args.n_trials,
                     fish_abundance=args.abundance,
                     tau=args.tau,
                     suggest_slw=args.suggest_slw)
