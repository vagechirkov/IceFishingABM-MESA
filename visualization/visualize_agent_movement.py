import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from abm.model import Model as RandomWalkerModel


def save_agent_movement_gif(model, steps=100, filename="scatter.gif", interval=200):
    """
    Simulate the agent's movement and save it as a GIF.

    Parameters:
        model (Model): An instance of the simulation model.
        steps (int): Number of steps to simulate.
        filename (str): Name of the output GIF file.
        interval (int): Interval between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)

    # Initialize scatter plots for agents and resources
    agent_scatter = ax.scatter([], [], s=80, color="blue", label="Agent")
    resource_scatter = ax.scatter([], [], s=30, color="green", label="Resource")
    ax.legend(loc="upper right")

    def update(frame):
        # Move the model forward by one step
        model.step()

        # Extract agent and resource locations separately
        agent_positions = []
        resource_positions = []

        for obj in model.schedule.agents:
            if (
                hasattr(obj, "is_agent") and obj.is_agent
            ):  # Assuming agent has `is_agent` attribute
                agent_positions.append(obj.pos)
            elif (
                hasattr(obj, "is_resource") and obj.is_resource
            ):  # Assuming resource has `is_resource` attribute
                resource_positions.append(obj.pos)

        # Update positions for agents and resources
        if agent_positions:
            agent_x, agent_y = zip(*agent_positions)
        else:
            agent_x, agent_y = [], []

        if resource_positions:
            resource_x, resource_y = zip(*resource_positions)
        else:
            resource_x, resource_y = [], []

        # Update scatter plot data for agents and resources
        agent_scatter.set_offsets(np.c_[agent_x, agent_y])
        resource_scatter.set_offsets(np.c_[resource_x, resource_y])

        # Update title with the current step number
        ax.set_title(f"Step: {frame+1}")

    # Animate and save as GIF
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval)
    ani.save(filename, writer="pillow", fps=10)
    plt.close()
