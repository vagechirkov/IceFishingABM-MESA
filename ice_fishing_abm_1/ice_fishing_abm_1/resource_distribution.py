import mesa
import numpy as np
from sklearn.datasets import make_blobs


class ResourceDistribution:
    def __init__(self, model: mesa.Model, n_samples: int = 100_000, n_clusters: int = 1):
        self.model = model
        self.n_samples = n_samples

        centers = []
        for _ in range(n_clusters):
            x = self.model.random.uniform(-1, 1)
            y = self.model.random.uniform(-1, 1)
            centers.append((x, y))

        # generate initial resource distribution
        self.resource_distribution = generate_resource_map(
            width=self.model.grid.width,
            height=self.model.grid.height,
            max_value=1,
            cluster_std=0.4,
            n_samples=100_000,
            centers=centers,
            normalize=True,
            random_seed=42
        )


def generate_resource_map(
        width: int, height: int,
        max_value: float = 1,
        cluster_std: float = 0.4,
        n_samples: int = 100_000,
        centers: list[tuple[float, float], ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1)),
        normalize: bool = True,
        random_seed: int = 42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_seed)
    x_edges = np.linspace(-2, 2, width + 1)
    y_edges = np.linspace(-2, 2, height + 1)
    resource_count, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges), density=False)

    # add missing samples to the random places
    n_missing_samples = n_samples - np.sum(resource_count)
    for _ in range(n_missing_samples.astype(int)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        resource_count[x, y] += 1

    # rescale to [0, max_value]
    if normalize:
        resource_count = (resource_count / np.max(resource_count)) * max_value

    # # cut off values below min_value
    # density[density < min_value] = 0

    return resource_count
