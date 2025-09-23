import numpy as np

from abm.model import IceFishingModel
from abm.resource import spatiotemporal_fish_density


def generate_simulations(grid_size=90, n_time=120):
    rng = np.random.default_rng(42)
    fish_density, _, _, _ = spatiotemporal_fish_density(
        rng,
        length_scale_time=15,
        length_scale_space=6,
        n_x=grid_size,
        n_y=grid_size,
        n_time=n_time,
        n_samples=1,
        temperature=0.5,
        bias=1,
    )

    _model = IceFishingModel(
        grid_size=grid_size,
        number_of_agents=5,
        fish_density=fish_density[0],
    )

    for _ in range(n_time):
        _model.step()
    results = _model.datacollector.get_model_vars_dataframe()
    print(results)




if __name__ == "__main__":
    generate_simulations()