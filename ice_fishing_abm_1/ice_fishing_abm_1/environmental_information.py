import numpy as np


def estimate_environment_peak(agent_location: tuple[int, int], observations: np.ndarray) -> tuple[int, ...]:
    """
    Estimate the maximum resource density
    """
    # get the location of the highest resource density
    max_location = np.unravel_index(np.argmax(observations, axis=None), observations.shape)
    return max_location
