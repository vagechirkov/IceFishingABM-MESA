import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def save_agent_movement_gif(
    model, steps=100, filename="scatter.gif", interval=200, resource_cluster_radius=2
):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_aspect('equal', 'box')

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
        [], [], s=resource_size, c=[], cmap="Greens", 
        vmin=0, vmax=1,  # Add explicit value range
        alpha=0.8, label="Resource", zorder=2
    )
    destination_marker = ax.scatter([], [], s=100, color="black", marker="x", label="Destination", zorder=1)

    # Add legends with fixed sizes
    legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.25, 1.25),  # Check again if the  position the legend outside the plot
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
        resource_positions = []
        resource_intensities = []
        destinations = []

        for obj in model.schedule.agents:
            if hasattr(obj, "is_agent") and obj.is_agent:
                agent_positions.append(obj.pos)

                # Adjust agent color based on state
                if obj.is_moving:
                    agent_colors.append("blue")
                elif obj.is_sampling:
                    if obj.is_consuming:
                        agent_colors.append("red")
                    else:
                       agent_colors.append("#FFA500")

                else:
                    agent_colors.append("gray")

                

                # Record destination marker if the agent is moving
                if obj.is_moving and obj.destination is not None:
                    destinations.append(obj.destination)

            elif hasattr(obj, "is_resource") and obj.is_resource:
                resource_positions.append(obj.pos)

                # Normalize and ensure visibility for intensities
                if obj.max_value > 0:
                    intensity = obj.current_value / obj.max_value
                else:
                    intensity = 0  # Handle edge case where max_value is 0

                # Set minimum intensity for visibility
                resource_intensities.append(intensity)
        


        agent_x, agent_y = zip(*agent_positions) if agent_positions else ([], [])
        resource_x, resource_y = zip(*resource_positions) if resource_positions else ([], [])
        destination_x, destination_y = zip(*destinations) if destinations else ([], [])

        # Update scatter plots
        agent_scatter.set_offsets(np.c_[agent_x, agent_y])
        agent_scatter.set_color(agent_colors)  # Set individual colors for each agent
        

        #resource_scatter.set_offsets(np.c_[resource_x, resource_y])

        # Ensure intensities are numeric and non-empty
        # Update resource scatter with normalized intensity
        if resource_positions:
            resource_scatter.set_offsets(np.c_[resource_x, resource_y])
            resource_scatter.set_array(np.array(resource_intensities))
        else:
            resource_scatter.set_offsets(np.empty((0, 2)))
            resource_scatter.set_array(np.array([]))

        
        destination_marker.set_offsets(np.c_[destination_x, destination_y])
        ax.set_title(f"Step: {frame+1}")
        return agent_scatter, resource_scatter, destination_marker

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval)
    writer = animation.PillowWriter(fps=10)
    ani.save(filename, writer=writer)
    plt.close()
