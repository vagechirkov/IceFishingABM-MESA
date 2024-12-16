import mesa
import numpy as np


class Resource(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        radius: int = 5,
        max_value: int = 100,
        current_value: int = 50,
        keep_overall_abundance: bool = True,
        neighborhood_radius: int = 20,
    ):
        super().__init__(unique_id, model)
        self.radius: int = radius
        self.model = model
        self.max_value: int = max_value
        self.current_value: int = current_value
        self.keep_overall_abundance: bool = keep_overall_abundance
        self.neighborhood_radius: int = neighborhood_radius
        self.is_resource = True
        self.const_resource: bool = True
        self.const_catch_probability: bool = True

    @property
    def catch_probability(self):
        # Linear relation for catch probability
        if self.const_catch_probability:
            return 1
        else:
            return self.current_value / self.max_value

    def catch(self):
        if (
            self.model.random.random() < self.catch_probability
            and self.current_value > 0
        ):
            if self.const_resource:  # Constant resource doesnt deplet.
                pass
            elif self.current_value > 0:
                self.current_value -= 1
            if self.keep_overall_abundance:
                self._add_resource_to_neighbour()
            return True
        else:
            return False

    def _add_resource_to_neighbour(self):
        """Add one resource to the closest neighbor."""
        # Comment : Need to update this so that the resource is added to a new randomly created resource
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        resources = [n for n in neighbors if isinstance(n, Resource)]

        if len(resources) > 0:
            closest_resource = min(
                resources, key=lambda x: self.model.grid.get_distance(self.pos, x.pos)
            )
            closest_resource.current_value += 1

    def resource_map(self) -> np.ndarray:
        # Generate resource map based on resource radius
        _resource_map = np.zeros((self.model.grid.width, self.model.grid.height))
        iter_neighborhood = self.model.grid.iter_neighborhood(
            self.pos, moore=False, include_center=True, radius=self.radius
        )
        for n in iter_neighborhood:
            _resource_map[n] = 1
        return _resource_map.astype(float) * self.catch_probability


def make_resource_centers(
    model, n_clusters: int = 1, cluster_radius: int = 5
) -> tuple[tuple[int, int], ...]:
    """make sure that the cluster centers are not too close together"""
    centers = []

    while len(centers) < n_clusters:
        x = model.random.randint(cluster_radius, model.grid.width - cluster_radius)
        y = model.random.randint(cluster_radius, model.grid.height - cluster_radius)

        if len(centers) == 0:
            centers.append((x, y))
            continue

        # TODO: fix this to avoid infinite loop in the case when it is not possible to add a new center far enough
        # TODO: from the existing centers
        # check if the new center is not too close to any of the existing centers
        if np.all(
            np.linalg.norm(np.array(centers) - np.array((x, y)), axis=-1)
            > cluster_radius * 2
        ):
            centers.append((x, y))
    return tuple(centers)
