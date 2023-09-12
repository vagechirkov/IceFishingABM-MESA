import mesa

from .model import Model
from .visualization.CanvasGridVisualization import CustomCanvasGrid


def draw_grid(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return

    if agent.is_sampling:
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "Collected resource": agent.collected_resource}
    elif agent.is_moving:
        portrayal = {"Shape": "fisher-moving.svg", "Layer": 1, "Collected resource": agent.collected_resource}
    else:
        portrayal = {}

    return portrayal


grid_size = 50
grid_canvas_size = 600

grid = CustomCanvasGrid(draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size)

model_params = {
    "grid_width": grid_size,
    "grid_height": grid_size,
    "number_of_agents": mesa.visualization.Slider(
        "The number of agents", value=5, min_value=1, max_value=10, step=1),
    "n_resource_clusters": mesa.visualization.Slider(
        "The number of resource clusters", value=3, min_value=1, max_value=5, step=1)

}

plots = [grid]

server = mesa.visualization.ModularServer(Model, plots, "Ice Fishing Model 1", model_params)
