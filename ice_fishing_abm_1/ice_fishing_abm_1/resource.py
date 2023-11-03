import mesa
import numpy as np


class Resource(mesa.Agent):
    def __init__(self, unique_id, model, radius=5, max_value=100, current_value=50, keep_overall_abundance=True,
                 neighborhood_radius=20):
        super().__init__(unique_id, model)
        self.radius: int = radius
        self.model = model
        self.max_value: int = max_value
        self.current_value: int = current_value
        self.keep_overall_abundance: bool = keep_overall_abundance
        self.neighborhood_radius: int = neighborhood_radius

    def catch_probability(self):
        # this relation is linear for now but might be more realistic if it is sigmoidal
        return self.current_value / self.max_value

    def catch(self):
        if self.model.random.random() < self.catch_probability():
            self.current_value -= 1
            if self.keep_overall_abundance:
                self._add_resource_to_neighbour()
            return True
        else:
            return False

    def _add_resource_to_neighbour(self):
        """Add one resource to the closest neighbor"""
        neighbours = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False,
                                                      radius=self.neighborhood_radius)
        resources = []
        for neighbour in neighbours:
            for agent in self.model.grid.get_cell_list_contents([neighbour]):
                if isinstance(agent, Resource):
                    resources.append(agent)

        # if the current resource is not the only resource in the neighborhood
        if len(resources) > 0:
            # find closest resource
            closest_resource = min(resources, key=lambda x: self.model.grid.get_distance(self.pos, x.pos))
            closest_resource.current_value += 1

    def place_resource(self, pos: tuple[int, int]):
        self.model.grid.place_agent(self, pos)

    def step(self):
        pass

    def resource_map(self) -> np.ndarray:
        size_x, size_y = self.model.grid.width, self.model.grid.height
        # NB: ij = yx coordinates
        j, i = self.pos
        # create a meshgrid
        x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))

        # draw a circle
        circle = (x - i) ** 2 + (y - j) ** 2 <= self.radius ** 2

        # return a resource map
        return circle.astype(float) * self.catch_probability()
