import numpy as np


def estimate_social_vector(agent_location: tuple[int, int], others_locations: list[tuple[int, int], ...]) -> np.ndarray:
    """ Estimate the social vector as a sum of unit vectors from the agent to others"""
    social_vector = np.zeros(2)
    for other_location in others_locations:
        v = np.array(other_location) - np.array(agent_location)
        social_vector += v / np.linalg.norm(v)
    return social_vector


