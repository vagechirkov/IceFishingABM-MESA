import mesa
from mesa.visualization.modules import CanvasGrid

from .movement_destination_subroutine import ExplorationStrategy
from .patch_evaluation_subroutine import InfotaxisPatchEvaluationSubroutine
from .model import Model
from .resource import Resource

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
    elif isinstance(agent, Resource):
        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "Color": "grey",
            "r": 0.5,
            "Layer": 0,
            "Radius": agent.model.resource_cluster_radius,
        }
    else:
        portrayal = {}

    return portrayal


grid_size = 20
grid_canvas_size = 600

grid = CanvasGrid(draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size)

model_params = {
    "exploration_strategy": ExplorationStrategy(grid_size=grid_size),
    "exploitation_strategy": InfotaxisPatchEvaluationSubroutine(threshold=10),
    "grid_size": grid_size,
    "number_of_agents": 5,
    "n_resource_clusters": 2,
    "resource_quality": 0.8,
    "resource_cluster_radius": 3,
    "keep_overall_abundance": True,
}

plots = [grid]

server = mesa.visualization.ModularServer(Model, plots, "GP Model", model_params)
