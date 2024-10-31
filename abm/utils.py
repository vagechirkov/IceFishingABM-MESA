import numpy as np

def x_y_to_i_j(x: int, y: int) -> tuple[int, int]:
    """
    Mesa coordinate system counts axis differently than numpy.
    """
    i, j = y, x

    return i, j


def discount_by_distance(array, agent_location, discount_factor=0.5):
    """
    Exponentially discount array by distance from the agent.
    """
    # create a meshgrid
    x, y = np.meshgrid(np.arange(array.shape[0]), np.arange(array.shape[1]), indexing='ij')

    # calculate distance from the agent
    distance = np.sqrt((x - agent_location[0]) ** 2 + (y - agent_location[1]) ** 2)

    # discount observations by distance
    return array * discount_factor ** distance


def find_peak(array: np.ndarray) -> tuple[int, ...]:
    """
    Find the maximum value of an array.
    """
    # get the location of the maximum value
    max_location = np.unravel_index(np.argmax(array, axis=None), array.shape)
    return max_location
