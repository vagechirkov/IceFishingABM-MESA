import mesa
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs


class ResourceDistribution:
    def __init__(
            self,
            model: mesa.Model,
            n_clusters: int = 1,
            max_value: float = 1,
            min_value: float = 0.01,
            cluster_std: float = 0.2,
            cluster_radius: int = 5,
            noize_level: float = 0.4
    ):
        self.model = model
        self.n_clusters = n_clusters
        self.max_value = max_value
        self.min_value = min_value
        self.cluster_std = cluster_std
        self.cluster_radius = cluster_radius
        self.noize_level = noize_level
        self.resource_distribution = np.zeros(shape=(self.model.grid.width, self.model.grid.height), dtype=float)
        self.centers = self.make_cluster_centers(n_clusters=self.n_clusters, cluster_radius=self.cluster_radius)

    def generate_resource_map(self):

        for center in self.centers:
            one_cluster = make_circle_cluster(radius=self.cluster_radius, max_value=self.max_value)

            # add the cluster to the resource distribution
            x, y = center
            x_slice = slice(x - self.cluster_radius, x + self.cluster_radius)
            y_slice = slice(y - self.cluster_radius, y + self.cluster_radius)
            self.resource_distribution[x_slice, y_slice] += one_cluster

        # add noise (replace self.noize_level portion of the resource map with random values)
        self.add_noise()

        # set minimum value
        self.resource_distribution[self.resource_distribution < self.min_value] = self.min_value

        # threshold the maximum value
        self.resource_distribution[self.resource_distribution > self.max_value] = self.max_value

    def add_noise(self):
        """
        Add noise to the resource distribution.
        """
        for row in range(self.model.grid.width):
            for col in range(self.model.grid.height):
                if self.model.random.random() < self.noize_level:
                    self.resource_distribution[row, col] = self.model.random.randint(0, 30) / 100

    def make_cluster_centers(self, n_clusters: int = 1, cluster_radius: int = 5) -> tuple[tuple[float, float], ...]:
        """make sure that the cluster centers are not too close together"""
        centers = []

        while len(centers) < n_clusters:
            x = self.model.random.randint(cluster_radius, self.model.grid.width - cluster_radius)
            y = self.model.random.randint(cluster_radius, self.model.grid.height - cluster_radius)

            if len(centers) == 0:
                centers.append((x, y))
                continue

            # TODO: fix this to avoid infinite loop in the case when it is not possible to add a new center far enough
            # TODO: from the existing centers
            # check if the new center is not too close to any of the existing centers
            if np.all(np.linalg.norm(np.array(centers) - np.array((x, y)), axis=-1) > cluster_radius):
                centers.append((x, y))
        return tuple(centers)


def make_circle_cluster(radius: int, max_value: float = 1):
    # generate a grid of points
    x, y = np.meshgrid(np.linspace(-radius, radius, radius * 2), np.linspace(-radius, radius, radius * 2))

    # points in the circle have max_value, points outside have 0
    resource_map = np.zeros(x.shape)
    resource_map[np.sqrt(x ** 2 + y ** 2) < radius - 1] = max_value
    return resource_map


def make_resource_cluster(width: int,
                          height: int,
                          cluster_std: float = 0.2,
                          min_value: float = 0.01,
                          max_value: float = 0.8,
                          cov: np.ndarray = np.array([[1, 0], [0, 1]]),
                          uniform: bool = True,
                          uniform_threshold: float = 0.5) -> np.ndarray:
    """
    Generate a cluster of resource samples around the center point.
    :return: 2D array of resource samples
    """
    # generate a grid of points
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    dist = multivariate_normal(cov=cov * cluster_std, mean=[0, 0])

    # Generating the density function
    # for each point in the meshgrid
    resource_map = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            resource_map[i, j] = dist.pdf([x[i, j], y[i, j]])

    # rescale to [0, max_value]
    resource_map = (resource_map / np.max(resource_map)) * max_value

    if uniform:
        # cut off values below min_value
        resource_map[resource_map < uniform_threshold] = min_value
        resource_map[resource_map >= uniform_threshold] = max_value

    return resource_map


def generate_resource_map_with_make_blobs(
        width: int,
        height: int,
        max_value: float = 1,
        min_value: float = 0,
        cluster_std: float = 0.4,
        n_samples: int = 100_000,
        centers: list[tuple[float, float], ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1)),
        normalize: bool = True,
        random_seed: int = 42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_seed,
                      center_box=(-1, 1))
    x_edges = np.linspace(-2, 2, width + 1)
    y_edges = np.linspace(-2, 2, height + 1)
    resource_map, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges), density=False)

    # add missing samples to the random places
    n_missing_samples = n_samples - np.sum(resource_map)
    for _ in range(n_missing_samples.astype(int)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        resource_map[x, y] += 1

    # rescale to [0, max_value]
    if normalize:
        resource_map = (resource_map / np.max(resource_map)) * max_value
        resource_map[resource_map < min_value] = 0.05  # cut off values below min_value

    # # cut off values below min_value
    # density[density < min_value] = 0

    return resource_map
