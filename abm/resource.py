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
        self.const_resource: bool = False 
        self.const_catch_probability: bool = True
        self.is_depleted = False

    @property
    def catch_probability(self):
        # Linear relation for catch probability
        if self.const_catch_probability:
            return 1
        else:
            return self.current_value / self.max_value

    def catch(self):
        if (self.model.random.random() < self.catch_probability
            and self.current_value > 0):
            
            if self.const_resource:  # Constant resource doesn't deplete
                return True
            
            self.current_value -= 1
            self.model.total_consumed_resource += 1 
            
            # Check if resource is depleted
            if self.current_value <= 0:
                if self.keep_overall_abundance:
                    # Instead of adding to neighbor, relocate this resource
                    self.relocate_resource()
                self.is_depleted = True
            
            return True
        else:
            return False
        
    def find_new_location(self):
        """Find a new location that doesn't overlap with existing resources."""
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random coordinates
            x = self.model.random.randint(self.radius, self.model.grid.width - self.radius)
            y = self.model.random.randint(self.radius, self.model.grid.height - self.radius)
            
            # Get all nearby cells within twice the resource radius
            nearby_cells = self.model.grid.get_neighborhood(
                (x, y), 
                moore=True, 
                include_center=True, 
                radius=self.radius * 2
            )
            
            # Check if any nearby cells contain resources
            has_nearby_resource = False
            for cell in nearby_cells:
                cell_contents = self.model.grid.get_cell_list_contents(cell)
                if any(isinstance(agent, Resource) for agent in cell_contents):
                    has_nearby_resource = True
                    break
            
            if not has_nearby_resource:
                return (x, y)
            
            attempts += 1
        
        return None  # Return None if no valid location found

    def relocate_resource(self):
        """Relocate the resource to a new valid location."""
        new_pos = self.find_new_location()
        if new_pos:
            self.model.grid.move_agent(self, new_pos)
            self.current_value = self.max_value  # Reset resource value
            self.is_depleted = False
            return True
        return False

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

        # Check if the new center is not too close to any of the existing centers
        if np.all(
            np.linalg.norm(np.array(centers) - np.array((x, y)), axis=-1)
            > cluster_radius * 2
        ):
            centers.append((x, y))
    return tuple(centers)