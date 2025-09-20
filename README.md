# IceFishingABM-MESA

## Installation Instructions

```bash
# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# freez only the explicitly installed dependencies
 pip list --not-required --format=freeze > requirements.txt
```

## Running the Model

```bash
# activate the virtual environment
source venv/bin/activate

# add python path
export PYTHONPATH=absolute/path/to/IceFishingABM-MESA:$PYTHONPATH

# run experiment
python -m experiments.experiment_experiment_name
```

## View optuna results

```bash
# cd to the shared directory (e.g., cd /Volumes/share/Chirkov_Jana/)
optuna-dashboard sqlite:///foraging-db.db
```
