import mesa


class Resource(mesa.Agent):
    def __init__(self, unique_id, model, radius=5, max_value=100, current_value=50):
        super().__init__(unique_id, model)
        self.radius: int = radius
        self.model = model
        self.max_value: int = max_value
        self.current_value: int = current_value

    def catch_probability(self):
        return self.current_value / self.max_value

    def catch(self):
        if self.model.random.random() < self.catch_probability():
            self.current_value -= 1
            self._add_resource_to_neighbour()
            return True
        else:
            return False

    def _add_resource_to_neighbour(self):
        neighbours = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False,
                                                      radius=self.model.grid.width)
        for neighbour in neighbours:
            neighbour_agent = self.model.grid.get_cell_list_contents([neighbour])[0]
            if isinstance(neighbour_agent, Resource):
                neighbour_agent.current_value += 1
                break

    def step(self):
        pass

    def resource_map(self):
        size_x, size_y = self.model.grid.width, self.model.grid.height
        # NB: ij vs xy coordinates
        x, y = self.pos
        return NotImplementedError
