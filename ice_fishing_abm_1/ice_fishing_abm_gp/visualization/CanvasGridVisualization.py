from collections import defaultdict

import matplotlib as mpl
import mesa
import numpy as np

cmap = mpl.colormaps['Greys']


class CustomCanvasGrid(mesa.visualization.CanvasGrid):
    def __init__(
            self,
            portrayal_method,
            grid_width,
            grid_height,
            canvas_width=500,
            canvas_height=500,
            variable_to_visualize="grid_colors",
    ):
        super().__init__(portrayal_method, grid_width, grid_height, canvas_width, canvas_height)
        self.variable_to_visualize = variable_to_visualize
        self.grid_colors = None
        self.grid_colors_map = None

    def draw_cell_color(self, x, y):
        """
        Draw a cell's color on the grid.
        """
        color = mpl.colors.to_hex(self.grid_colors_map.to_rgba(self.grid_colors[x, y]))

        return {"Color": color, "Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "x": x, "y": y}

    def check_grid_colors(self, model):
        if hasattr(model, self.variable_to_visualize) and getattr(model, self.variable_to_visualize) is not None:
            variable_to_visualize = getattr(model, self.variable_to_visualize)
            assert isinstance(variable_to_visualize, np.ndarray)
            assert variable_to_visualize.shape == (model.grid.width, model.grid.height)
            self.grid_colors = variable_to_visualize
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            self.grid_colors_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    def render(self, model):
        # copy grid colors from the model
        self.check_grid_colors(model)

        grid_state = defaultdict(list)
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                cell_objects = model.grid.get_cell_list_contents([(x, y)])
                for obj in cell_objects:
                    portrayal = self.portrayal_method(obj)
                    if portrayal:
                        portrayal["x"] = x
                        portrayal["y"] = y
                        grid_state[portrayal["Layer"]].append(portrayal)

                if self.grid_colors is not None:
                    grid_state[0].append(self.draw_cell_color(x, y))
        return grid_state
