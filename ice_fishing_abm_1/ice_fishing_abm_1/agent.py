import mesa


class Agent(mesa.Agent):
    def __init__(self, unique_id, model: mesa.Model):
        super().__init__(unique_id, model)
        self.state = "fishing"
