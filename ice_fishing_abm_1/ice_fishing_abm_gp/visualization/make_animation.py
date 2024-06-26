import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mesa.experimental.jupyter_viz import JupyterContainer
import seaborn as sns
from svgpath2mpl import parse_path
from svgpathtools import svg2paths

from ice_fishing_abm_1.ice_fishing_abm_gp.agent import Agent
from ice_fishing_abm_1.ice_fishing_abm_gp.resource import Resource

_, fisher_fishing_attributes = svg2paths('fisher-fishing-simple.svg')
fisher_fishing_marker = parse_path(fisher_fishing_attributes[0]['d'])

fisher_fishing_marker.vertices -= fisher_fishing_marker.vertices.mean(axis=0)
fisher_fishing_marker = fisher_fishing_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
fisher_fishing_marker = fisher_fishing_marker.transformed(mpl.transforms.Affine2D().scale(-1, 1))

_, fisher_running_attributes = svg2paths('fisher-running-simple.svg')
fisher_running_marker = parse_path(fisher_running_attributes[0]['d'])

fisher_running_marker.vertices -= fisher_running_marker.vertices.mean(axis=0)
fisher_running_marker = fisher_running_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
fisher_running_marker = fisher_running_marker.transformed(mpl.transforms.Affine2D().scale(-1, 1))


def portray(g, agent_type=Agent, state='fishing'):
    x = []
    y = []
    s = []  # size
    c = []  # color
    for i in range(g.width):
        for j in range(g.height):
            content = g._grid[i][j]
            if not content:
                continue
            if not hasattr(content, "__iter__"):
                # Is a single grid
                content = [content]
            for agent in content:
                if not isinstance(agent, agent_type):
                    continue
                if state == 'fishing' and not agent._is_sampling:
                    continue
                if state == 'moving' and not agent._is_moving:
                    continue

                is_focused = agent.unique_id == agent.model.n_resource_clusters + 1
                x.append(i)
                y.append(j)
                s.append(500)
                c.append("red" if is_focused else "blue")

    out = {"x": x, "y": y}
    if len(s) > 0:
        out["s"] = s
    if len(c) > 0:
        out["c"] = c
    return out


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
    data_fishing = portray(model.grid, state='fishing')
    data_moving = portray(model.grid, state='moving')

    for data, marker in zip([data_fishing, data_moving], [fisher_fishing_marker, fisher_running_marker]):
        if len(data["x"]) == 0:
            continue

        scatter = ax.scatter(**data, marker=marker)
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
        space_ax.set_title(f"Step {model.schedule.steps}")

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
                "is_resource": False,
                "focused": is_focused,
                "fishing": agent._is_sampling,
            }
        else:
            return {
                "color": "black",
                "size": 1,
                "marker": "s",
                "is_resource": True,
                "focused": False,
                "fishing": False,
            }


    model_params = {
        "grid_size": 80,
        "number_of_agents": 10,
        "n_resource_clusters": 5,
        "resource_cluster_radius": 7,
        "resource_quality": 0.5,
        "keep_overall_abundance": False,
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
