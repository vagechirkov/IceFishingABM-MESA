import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def save_agent_movement_gif(model, steps=100, filename="scatter.gif", interval=200):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)

    agent_scatter = ax.scatter([], [], s=80, color="blue", label="Agent")
    resource_scatter = ax.scatter([], [], s=30, color="green", label="Resource")
    ax.legend(loc="upper right")

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
        resource_x, resource_y = zip(*resource_positions) if resource_positions else ([], [])

        agent_scatter.set_offsets(np.c_[agent_x, agent_y])
        resource_scatter.set_offsets(np.c_[resource_x, resource_y])
        ax.set_title(f"Step: {frame+1}")
        return agent_scatter, resource_scatter

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval)
    writer = animation.PillowWriter(fps=10)
    ani.save(filename, writer=writer)
    plt.close()