import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_observations_with_gaussian_filter(observations, sigma=5):
    """
    Smooth observations with a Gaussian kernel.
    """
    return gaussian_filter(observations, sigma=sigma, mode='constant')


def discount_observations_by_distance(observations, agent_location, discount_factor=0.5):
    """
    Exponentially discount observations by distance from the agent.
    """
    # create a meshgrid
    x, y = np.meshgrid(np.arange(observations.shape[0]), np.arange(observations.shape[1]))

    # calculate distance from the agent
    distance = np.sqrt((x - agent_location[0]) ** 2 + (y - agent_location[1]) ** 2)

    # discount observations by distance
    return observations * discount_factor ** distance


def estimate_environment_peak(agent_location: tuple[int, int], observations: np.ndarray) -> tuple[int, ...]:
    """
    Estimate the maximum resource density
    """
    # get the location of the highest resource density
    max_location = np.unravel_index(np.argmax(observations, axis=None), observations.shape)
    return max_location