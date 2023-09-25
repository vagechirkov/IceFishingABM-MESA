import mesa

from .model import Model
from .visualization.CanvasGridVisualization import CustomCanvasGrid

SHOW_ICONS = True


def draw_grid(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return

    if agent.is_sampling:
        portrayal = {
            # "Shape": "fisher-fishing.svg",
            "Shape": "fisher-fishing.svg" if SHOW_ICONS else "circle",
            "Filled": "true",
            "Color": "red",
            "r": 0.8,
            "Layer": 1,
            "Collected resource": agent.collected_resource}
    elif agent.is_moving:
        portrayal = {
            # "Shape": "fisher-moving.svg",
            "Shape": "fisher-moving.svg" if SHOW_ICONS else "circle",
            "Filled": "true",
            "Color": "blue",
            "r": 0.8,
            "Layer": 1,
            "Collected resource": agent.collected_resource}
    else:
        portrayal = {}

    return portrayal


grid_size = 20
grid_canvas_size = 600

grid = CustomCanvasGrid(draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "grid_colors")
# chart_element = mesa.visualization.ChartModule(
#     [{"Label": "Average collected resource", "Color": "#AA0000"}], canvas_width=400, canvas_height=100
# )
agent_raw_observation_element = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "agent_raw_observations")

agent_smoothed_observation_element = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "agent_smoothed_observations")

agent_discounted_observation_element = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "agent_discounted_observations")

agent_raw_soc_observation_element = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "agent_raw_soc_observations")

agent_smoothed_soc_observation_element = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "agent_smoothed_soc_observations")

agent_discounted_soc_observation_element = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "agent_discounted_soc_observations")

relocation_map = CustomCanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size, "relocation_map")

model_params = {
    "grid_width": grid_size,
    "grid_height": grid_size,
    "visualization": True,
    "number_of_agents": mesa.visualization.Slider(
        "The number of agents", value=5, min_value=1, max_value=10, step=1),
    "n_resource_clusters": mesa.visualization.Slider(
        "The number of resource clusters", value=3, min_value=1, max_value=5, step=1)

}

plots = [grid,
         relocation_map,
         agent_raw_observation_element,
         agent_discounted_observation_element,
         agent_smoothed_observation_element,
         agent_raw_soc_observation_element,
         agent_discounted_soc_observation_element,
         agent_smoothed_soc_observation_element
         ]

server = mesa.visualization.ModularServer(Model, plots, "Ice Fishing Model 1", model_params)
