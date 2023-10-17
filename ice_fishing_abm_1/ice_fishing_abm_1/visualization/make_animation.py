import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mesa.experimental.jupyter_viz import JupyterContainer
import seaborn as sns


def draw_resource_distribution(model, ax):
    if model.resource_distribution is None:
        return

    # draw a heatmap of the resource distribution
    sns.heatmap(model.resource_distribution.T, ax=ax, cmap='Greys', cbar=False, square=True, vmin=0, vmax=1)


def draw_agent_meso_belief(model, ax):
    # select the agent with id 1
    _agent = [a for a in model.schedule.agents if a.unique_id == 1][0]
    if _agent.meso_belief is None:
        return

    ax.cla()
    # draw a heatmap of the resource distribution
    g = sns.heatmap(_agent.meso_belief.T, ax=ax, cmap='viridis', cbar=False, square=True)
    # g.invert_yaxis()
    # remove ticks
    g.set_xticks([])
    g.set_yticks([])
    # remove tick labels
    g.set_xticklabels([])
    g.set_yticklabels([])


def estimate_catch_rate(agent, model, previous_catch_rate: float = 0):
    if agent.is_moving:
        return 0

    # estimate catch rate
    if len(agent.sampling_sequence) >= model.sampling_length - 1:
        return np.mean(agent.sampling_sequence)
    else:
        return previous_catch_rate


def plot_n_steps(viz_container: JupyterContainer, n_steps: int = 10):
    model = viz_container.model_class(**viz_container.model_params_input, **viz_container.model_params_fixed)

    space_fig = Figure(figsize=(20, 10.2))
    axs = space_fig.subplots(ncols=2)
    space_ax = axs[0]
    belief = axs[1]
    space_fig.subplots_adjust(left=0, bottom=-0.05, right=1, top=1, wspace=None, hspace=None)
    space_ax.set_axis_off()
    # set limits to grid size
    space_ax.set_xlim(0, model.grid.width)
    space_ax.set_ylim(0, model.grid.height)
    # set equal aspect ratio
    space_ax.set_aspect('equal', adjustable='box')

    draw_resource_distribution(model, space_ax)
    scatter = space_ax.scatter(**viz_container.portray(model.grid))

    def update_grid(_scatter, data):
        coords = np.array(list(zip(data["x"], data["y"])))
        # center coordinates of the scatter points
        _scatter.set_offsets(coords + 0.5)
        if "c" in data:
            _scatter.set_color(data["c"])
        if "s" in data:
            _scatter.set_sizes(data["s"])
        return _scatter

    catch_rates = [estimate_catch_rate(a, model) for a in model.schedule.agents]

    def animate(_):
        nonlocal catch_rates
        if model.running:
            model.step()
        catch_rates = [estimate_catch_rate(a, model, c) for a, c in zip(model.schedule.agents, catch_rates)]
        space_ax.set_title(f"Step {model.schedule.steps} | "
                           f"Catch rates " + ' '.join(['%.2f'] * len(catch_rates)) % tuple(catch_rates))
        draw_agent_meso_belief(model, belief)
        return update_grid(scatter, viz_container.portray(model.grid))

    ani = animation.FuncAnimation(space_fig, animate, repeat=True, frames=n_steps, interval=400)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('scatter.gif', writer=writer)


# run an example
if __name__ == "__main__":
    from ice_fishing_abm_1.ice_fishing_abm_1.model import Model
    from mesa.experimental.jupyter_viz import JupyterContainer


    def agent_portrayal(agent):
        return {
            "color": "tab:orange" if agent.unique_id == 1 else "tab:blue" if agent.is_moving else "tab:red",
            "size": 30,
        }


    model_params = {
        "grid_width": 100,
        "grid_height": 100,
        "number_of_agents": 5,
        "n_resource_clusters": 10,
        "prior_knowledge": 0.05,
        "sampling_length": 10,
        "resource_cluster_radius": 20,
        "relocation_threshold": 0.1,
        "alpha_social": 0.4,
        "alpha_env": 0.4,
        "meso_grid_step": 10,
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
    plot_n_steps(viz_container=container, n_steps=300)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
