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

## Images for visualization

* [Fisher](https://www.svgrepo.com/svg/36567/fisher)
* [Ice skating](https://www.svgrepo.com/svg/116117/ice-skating)
* [Jackhammer](https://www.svgrepo.com/svg/233732/jackhammer)

## Run simulation on the cluster

Speed:~ 177 runs/s ~ 10k runs/min

```bash
tmux list-sessions 

tmux attach # attach to the session or create a new one if it does not exist: tmux

cd /mesa-simulations/IceFishingABM-MESA

git fetch
git pull

bash run.sh # run the simulation

```


## Run visualization with mesa
```bash
python IceFishingABM-MESA/ice_fishing_abm_1/run_server.py 
```