import matplotlib

from abm.model import IceFishingModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def save_agent_movement_gif(model, steps, filename="ice_fishing_abm.gif", fps=10):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_aspect("equal", "box")

    # Add dotted grid lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(color="gray", linestyle="-", linewidth=0.1, alpha=0.2)

    # Initialize background heatmap (fish density)
    fish_im = ax.imshow(
        model.fish_density[:, :, 0],  # first time slice
        origin="lower",
        cmap="viridis",  # "gray_r",
        alpha=0.5,
        extent=[0, model.grid.width, 0, model.grid.height],
        zorder=0,
    )

    # Initialize scatter plots
    agent_scatter = ax.scatter(
        [],
        [],
        edgecolors="black",
        linewidths=0.7,
        s=60,
        color="blue",
        label="Agent",
        zorder=3,
    )
    # Set marker size to 0 for final figures but keeping it as legacy
    destination_marker = ax.scatter(
        [], [], s=0, color="black", marker="x", label="Destination", zorder=1
    )

    # Add legends with fixed sizes
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.25,
            1.25,
        ),  # Check again if the  position the legend outside the plot
        scatterpoints=1,
        fontsize=10,
        frameon=True,
        framealpha=0.8,
    )
    for handle in legend.legend_handles:
        handle._sizes = [80]  # Set a consistent legend marker size
        handle.set_alpha(1.0)  # Ensure legend markers are opaque

    def update(frame):
        model.step()
        agent_positions = []
        agent_colors = []  # New list to store colors for each agent
        destinations = []

        for agent in model.agents:
            # agent_positions.append((xy2ij(*agent.pos)))
            agent_positions.append(agent.pos)

            # Adjust agent color based on state
            if agent.is_moving:
                agent_colors.append("skyblue")
            elif agent.is_sampling:
                if agent.is_consuming:
                    agent_colors.append("red")
                else:
                    agent_colors.append("#FFA500")

            else:
                agent_colors.append("skyblue")

            # Record destination marker if the agent is moving
            if agent.is_moving and agent.destination is not None:
                # destinations.append((xy2ij(*agent.destination)))
                destinations.append(agent.destination)


        agent_x, agent_y = zip(*agent_positions) if agent_positions else ([], [])
        destination_x, destination_y = zip(*destinations) if destinations else ([], [])

        # Update scatter plots
        agent_scatter.set_offsets(np.c_[agent_x, agent_y])
        agent_scatter.set_facecolors(agent_colors)  # Set individual colors for each agent

        destination_marker.set_offsets(np.c_[destination_x, destination_y])

        if (model.steps_min != 0) and (model.steps % model.steps_min == 0):
            fish_im.set_data(model.fish_density[:, :, model.steps_min])

        ax.set_title(f"Minutes: {model.steps_min}")
        return agent_scatter, destination_marker, fish_im

    ani = animation.FuncAnimation(fig, update, frames=steps - 1, interval=200)
    writer = animation.PillowWriter(fps=fps)
    ani.save(filename, writer=writer)
    plt.close()


if __name__ == "__main__":
    grid_size = 90
    n_time = (120 - 1) * 6  # 30 * 6  #
    rng = np.random.default_rng(42)

    _model = IceFishingModel(
        grid_size=grid_size,
        number_of_agents=5,
        spot_selection_tau=0.1
    )

    save_agent_movement_gif(_model, n_time, fps=10)
