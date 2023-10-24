import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mesa.experimental.jupyter_viz import JupyterContainer
import seaborn as sns


def draw_resource_distribution(model, ax):
    if model.resource_distribution is None:
        return

    # draw a heatmap of the resource distribution
    sns.heatmap(model.resource_distribution, ax=ax, cmap='Greys', cbar=False, square=True, vmin=-0.2, vmax=1)


def draw_agent_meso_belief(model, ax, var_name, vmix=None, vmax=None, cmap='viridis'):
    # select the agent with id 1
    _agent = [a for a in model.schedule.agents if a.unique_id == 1][0]
    var = getattr(_agent, var_name)

    ax.cla()
    # draw a heatmap of the resource distribution
    g = sns.heatmap(var, ax=ax, cmap=cmap, cbar=False, square=True, vmin=vmix, vmax=vmax)


def estimate_catch_rate(agent, model, previous_catch_rate: float = 0):
    if agent._is_moving:
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

    # remove axis for all ax
    # for ax in [space_ax, observations_ax, env_belief_ax, soc_info_ax, combined_ax]:
    #     ax.set_axis_off()

    draw_resource_distribution(model, space_ax)
    scatter = space_ax.scatter(**viz_container.portray(model.grid))

    plt.tight_layout()

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
        draw_agent_meso_belief(model, soc_info_ax, "meso_soc")
        soc_info_ax.set_title("Social information")
        draw_agent_meso_belief(model, env_belief_ax, "meso_env", vmix=-0.2, vmax=0.5, cmap='Greys')
        env_belief_ax.set_title("Environmental belief")
        draw_agent_meso_belief(model, observations_ax, "observations", vmix=-0.2, vmax=0.5, cmap='Greys')
        observations_ax.set_title("Observations")
        draw_agent_meso_belief(model, combined_ax, "meso_combined")
        combined_ax.set_title("Combined")
        return update_grid(scatter, viz_container.portray(model.grid))

    ani = animation.FuncAnimation(space_fig, animate, repeat=True, frames=n_steps, interval=interval)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('scatter.gif', writer=writer)


# run an example
if __name__ == "__main__":
    from ice_fishing_abm_1.ice_fishing_abm_1.model import Model
    from mesa.experimental.jupyter_viz import JupyterContainer


    def agent_portrayal(agent):
        return {
            "color": "tab:orange" if agent.unique_id == 1 else "tab:blue" if agent._is_moving else "tab:red",
            "size": 150,
        }


    model_params = {
        "grid_width": 50,
        "grid_height": 50,
        "number_of_agents": 1,
        "n_resource_clusters": 1,
        "sampling_length": 10,
        "resource_cluster_radius": 10,
        "relocation_threshold": 0.1,
        "meso_grid_step": 10,
        "local_search_counter": 1,
        "w_social": 0,
        "w_personal": 1,
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
    plot_n_steps(viz_container=container, n_steps=200, interval=800)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
