import mesa
import matplotlib as mpl

from .model import Model

cmap = mpl.colormaps['Blues']
norm = mpl.colors.Normalize(vmin=0, vmax=20)
m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

m_belief = mpl.cm.ScalarMappable(cmap=mpl.colormaps['Blues'], norm=mpl.colors.Normalize(vmin=0, vmax=.05))


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

grid = mesa.visualization.CanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size)

model_params = {
    "grid_width": grid_size,
    "grid_height": grid_size,
    "number_of_agents": mesa.visualization.Slider("N agents", value=5, min_value=1, max_value=10, step=1),

}

plots = [grid]

server = mesa.visualization.ModularServer(Model, plots,
                                          "Ice Fishing Model 1", model_params
                                          )