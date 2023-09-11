import mesa

from .model import Model
from .visualization.CanvasGridVisualization import CustomCanvasGrid


def draw_grid(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return

    if agent.state == "fishing":
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "State": agent.state}
    elif agent.state == "moving":
        portrayal = {"Shape": "fisher-moving.svg", "Layer": 1, "State": agent.state}
    else:
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "State": agent.state}
    return portrayal


grid_size = 20
grid_canvas_size = 600

grid = CustomCanvasGrid(draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size)

model_params = {
    "grid_width": grid_size,
    "grid_height": grid_size,
    "number_of_agents": mesa.visualization.Slider("N agents", value=5, min_value=1, max_value=10, step=1),

}

plots = [grid]

server = mesa.visualization.ModularServer(Model, plots, "Ice Fishing Model 1", model_params)
