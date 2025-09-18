import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=10000, n_jobs=50)

study.best_params  # E.g. {'x': 2.002108042}