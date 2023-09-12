import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_observations_with_gaussian_filter(observations, sigma=5):
    """
    Smooth observations with a Gaussian kernel.
    """
    return gaussian_filter(observations, sigma=sigma, mode='constant')


def estimate_environment_peak(agent_location: tuple[int, int], observations: np.ndarray) -> tuple[int, ...]:
    """
    Estimate the maximum resource density
    """
    # get the location of the highest resource density
    max_location = np.unravel_index(np.argmax(observations, axis=None), observations.shape)
    return max_location
