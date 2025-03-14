# IceFishingABM-MESA

## Installation Instructions

```bash
# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install dependencies for development (optional)
pip install -r dev_requirements.txt

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
