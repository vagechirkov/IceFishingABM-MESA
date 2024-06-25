import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mesa.experimental.jupyter_viz import JupyterContainer
import seaborn as sns

from ice_fishing_abm_1.ice_fishing_abm_gp.agent import Agent
from ice_fishing_abm_1.ice_fishing_abm_gp.resource import Resource


def draw_resource_distribution(model, ax, viz_container: JupyterContainer):
    if model.resource_distribution is None:
        return

    # clear axis without removing scatter
    ax.clear()

    # draw a heatmap of the resource distribution
    sns.heatmap(model.resource_distribution, ax=ax, cmap='Greys', cbar=False, square=True, vmin=-0.2, vmax=1)
    ax.set_axis_off()

    # draw meso grid
    # h_lines = np.arange(0, model.grid.width, model.meso_grid_step)
    # v_lines = np.arange(0, model.grid.height, model.meso_grid_step)
    # ax.hlines(h_lines, *ax.get_xlim(), color='white', linewidth=0.5)
    # ax.vlines(v_lines, *ax.get_ylim(), color='white', linewidth=0.5)

    # draw agents
    scatter = ax.scatter(**viz_container.portray(model.grid))
    data = viz_container.portray(model.grid)
    coords = np.array(list(zip(data["x"], data["y"])))
    # center coordinates of the scatter points
    scatter.set_offsets(coords + 0.5)
    if "c" in data:
        scatter.set_color(data["c"])
    if "s" in data:
        scatter.set_sizes(data["s"])


def draw_agent_meso_belief(model, ax, var_name, vmix=None, vmax=None, cmap='viridis'):
    # select the agent with id 1
    _agent = [a for a in model.schedule.agents if a.unique_id == model.n_resource_clusters + 1][0]
    var = getattr(_agent, var_name)

    ax.clear()
    # draw a heatmap of the resource distribution
    sns.heatmap(var, ax=ax, cmap=cmap, cbar=False, square=True, vmin=vmix, vmax=vmax)
    ax.set_axis_off()

    # draw meso grid
    # h_lines = np.arange(0, model.grid.width, model.meso_grid_step)
    # v_lines = np.arange(0, model.grid.height, model.meso_grid_step)
    # ax.hlines(h_lines, *ax.get_xlim(), color='white', linewidth=0.5)
    # ax.vlines(v_lines, *ax.get_ylim(), color='white', linewidth=0.5)

    if _agent.is_moving and _agent._destination is not None and model.number_of_agents == 1:
        ax.scatter(*_agent._destination, color='tab:red', s=20, marker='x')


def estimate_catch_rate(agent, model, previous_catch_rate: float = 0):
    assert isinstance(agent, Agent)

    if agent.is_moving:
        return 0

    # estimate catch rate
    if len(agent._sampling_sequence) >= model.sampling_length - 1:
        return np.mean(agent._sampling_sequence)
    else:
        return previous_catch_rate


def plot_n_steps(viz_container: JupyterContainer, n_steps: int = 10, interval: int = 400):
    model = viz_container.model_class(**viz_container.model_params_input, **viz_container.model_params_fixed)

    space_fig = Figure(figsize=(13, 7))
    gs = GridSpec(2, 3, figure=space_fig, wspace=0.08, hspace=0.08,
                  width_ratios=[1, 0.5, 0.5], height_ratios=[1, 1], left=0.02, right=0.98, top=0.96, bottom=0.04)
    space_ax = space_fig.add_subplot(gs[:, 0])
    observations_ax = space_fig.add_subplot(gs[0, 1])
    env_belief_ax = space_fig.add_subplot(gs[0, 2])
    soc_info_ax = space_fig.add_subplot(gs[1, 1])
    combined_ax = space_fig.add_subplot(gs[1, 2])

    # remove axis
    for ax in space_fig.get_axes():
        ax.set_axis_off()

    draw_resource_distribution(model, space_ax, viz_container)

    plt.tight_layout()

    # catch_rates = [estimate_catch_rate(a, model) for a in model.schedule.agents if isinstance(a, Agent)]

    def animate(_):
        # nonlocal catch_rates
        if model.running:
            model.step()

        # resource distribution and the main scatter plot
        draw_resource_distribution(model, space_ax, viz_container)
        # catch_rates = [estimate_catch_rate(a, model, c) for a, c in zip(
        #     [a for a in model.schedule.agents if isinstance(a, Agent)], catch_rates)]
        # space_ax.set_title(f"Step {model.schedule.steps} | "
        #                    f"Catch rates " + ' '.join(['%.2f'] * len(catch_rates)) % tuple(catch_rates))

        # meso level plots
        draw_agent_meso_belief(model, soc_info_ax, "social_feature")  # , vmix=0, vmax=1, cmap='Greys'
        soc_info_ax.set_title("Social Feature")
        draw_agent_meso_belief(model, env_belief_ax, "success_feature")
        env_belief_ax.set_title("Success Feature")
        draw_agent_meso_belief(model, observations_ax, "failure_feature")
        observations_ax.set_title("Failure Feature")
        draw_agent_meso_belief(model, combined_ax, "belief")
        combined_ax.set_title("Combined Features")

    ani = animation.FuncAnimation(space_fig, animate, repeat=True, frames=n_steps, interval=interval)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('scatter.gif', writer=writer)


# run an example
if __name__ == "__main__":
    from ice_fishing_abm_1.ice_fishing_abm_gp.model import Model
    from mesa.experimental.jupyter_viz import JupyterContainer


    def agent_portrayal(agent):
        if not isinstance(agent, Resource):
            is_focused = agent.unique_id == agent.model.n_resource_clusters + 1
            return {
                "color": "tab:orange" if is_focused else "tab:blue" if agent._is_moving else "tab:red",
                "size": 100,
            }
        else:
            return {
                "color": "black",
                "size": 1,
            }


    model_params = {
        "grid_size": 80,
        "number_of_agents": 5,
        "n_resource_clusters": 2,
    }
    container = JupyterContainer(
        Model,
        model_params,
        name="Ice Fishing Model GP",
        agent_portrayal=agent_portrayal,
    )

    # start timer
    import time

    start = time.time()
    plot_n_steps(viz_container=container, n_steps=100, interval=800)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
