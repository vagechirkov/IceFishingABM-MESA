import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def save_agent_movement_gif(
    model, steps=100, filename="scatter.gif", interval=200, resource_cluster_radius=2
):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)

    # Add dotted grid lines
    ax.set_xticks(np.arange(0, model.grid.width + 1, 1))
    ax.set_yticks(np.arange(0, model.grid.height + 1, 1))
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Calculate marker size for resources
    grid_width_in_points = fig.get_size_inches()[0] * fig.dpi
    points_per_grid_unit = grid_width_in_points / model.grid.width
    resource_size = (resource_cluster_radius * points_per_grid_unit) ** 2

    # Initialize scatter plots
    agent_scatter = ax.scatter([], [], s=80, color="blue", label="Agent", zorder=3)
    resource_scatter = ax.scatter(
        [], [], s=resource_size, color="green", alpha=0.5, label="Resource", zorder=1
    )

    # Add legends with fixed sizes
    legend = ax.legend(
        loc="upper right",
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
        resource_positions = []

        for obj in model.schedule.agents:
            if hasattr(obj, "is_agent") and obj.is_agent:
                agent_positions.append(obj.pos)
            elif hasattr(obj, "is_resource") and obj.is_resource:
                resource_positions.append(obj.pos)

        agent_x, agent_y = zip(*agent_positions) if agent_positions else ([], [])
        resource_x, resource_y = zip(*resource_positions) if resource_positions else (
            [],
            [],
        )

        # Update scatter plots
        resource_scatter.set_offsets(np.c_[resource_x, resource_y])
        agent_scatter.set_offsets(np.c_[agent_x, agent_y])
        ax.set_title(f"Step: {frame+1}")
        return agent_scatter, resource_scatter

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval)
    writer = animation.PillowWriter(fps=10)
    ani.save(filename, writer=writer)
    plt.tight_layout()
    plt.close()
