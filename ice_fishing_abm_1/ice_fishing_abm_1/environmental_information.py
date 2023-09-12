from scipy.ndimage import gaussian_filter


def smooth_observations_with_gaussian_filter(observations, sigma=10):
    """
    Smooth observations with a Gaussian kernel.
    """
    return gaussian_filter(observations, sigma=sigma, mode='constant')
