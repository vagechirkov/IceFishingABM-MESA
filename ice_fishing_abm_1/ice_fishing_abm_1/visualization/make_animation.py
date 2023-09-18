import matplotlib.animation as animation
from matplotlib.figure import Figure
from mesa.experimental.jupyter_viz import JupyterContainer


def draw_resource_distribution(model, ax):
    if model.resource_distribution is None:
        return

    # draw a heatmap of the resource distribution
    # fill the whole image with the heatmap
    ax.imshow(model.resource_distribution.T, cmap="Greys", interpolation="nearest", origin="lower")


def plot_n_steps(viz_container: JupyterContainer, n_steps: int = 10):
    model = viz_container.model_class(**viz_container.model_params_input, **viz_container.model_params_fixed)

    space_fig = Figure(figsize=(10, 10))
    space_ax = space_fig.subplots()
    space_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    space_ax.set_axis_off()
    # set limits to grid size
    space_ax.set_xlim(0, model.grid.width)
    space_ax.set_ylim(0, model.grid.height)
    # set equal aspect ratio
    space_ax.set_aspect('equal', adjustable='box')

    draw_resource_distribution(model, space_ax)

    scatter = space_ax.scatter(**viz_container.portray(model.grid))

    def update_grid(_scatter, data):
        _scatter.set_offsets(list(zip(data["x"], data["y"])))
        if "c" in data:
            _scatter.set_color(data["c"])
        if "s" in data:
            _scatter.set_sizes(data["s"])
        return _scatter

    def animate(_):
        if model.running:
            model.step()
        return update_grid(scatter, viz_container.portray(model.grid))

    ani = animation.FuncAnimation(space_fig, animate, repeat=True, frames=n_steps, interval=400)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('scatter.gif', writer=writer)


# run an example
if __name__ == "__main__":
    from ice_fishing_abm_1.ice_fishing_abm_1.model import Model
    from mesa.experimental.jupyter_viz import JupyterContainer


    def agent_portrayal(agent):
        return {
            "color": "tab:blue",
            "size": 50,
        }


    model_params = {
        "grid_width": 100,
        "grid_height": 100,
        "number_of_agents": 5,
        "n_resource_clusters": 5,
        "exploration_threshold": 0.1,
        "prior_knowledge": 0.05,
        "sampling_length": 10,
        "social_influence_threshold": 1,
        "relocation_threshold": 0.4
    }
    container = JupyterContainer(
        Model,
        model_params,
        name="Ice Fishing Model 1",
        agent_portrayal=agent_portrayal,
    )

    # start timer
    import time
    start = time.time()
    plot_n_steps(viz_container=container, n_steps=1000)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
