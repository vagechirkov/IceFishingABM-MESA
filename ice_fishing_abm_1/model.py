import mesa
from mesa.space import MultiGrid

from .agent import Agent


class Model(mesa.Model):
    def __init__(self,
                 grid_width: int = 100,
                 grid_height: int = 100,
                 number_of_agents: int = 5):
        super().__init__()
        self.number_of_agents = number_of_agents
        self.current_id = 0
        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        # self.datacollector = mesa.datacollection.DataCollector(model_reporters={},)
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for _ in range(self.number_of_agents):
            self.initialize_agent()

    def initialize_agent(self):
        """Create an agent and add it to the schedule and grid at a random empty cell in the center of the grid."""
        a = Agent(self.next_id(), self)

        self.schedule.add(a)

        assert self.grid.width > 6 and self.grid.height > 6, "Grid is too small to place agents in the center."

        x, y = self.grid.width // 2, self.grid.height // 2
        while self.grid.is_cell_empty((x, y)):
            x = self.random.randrange((self.grid.width // 2) - 3, (self.grid.width // 2) + 3)
            y = self.random.randrange((self.grid.height // 2) - 3, (self.grid.height // 2) + 3)

        self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()

        # collect data
        # self.datacollector.collect(self)

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
