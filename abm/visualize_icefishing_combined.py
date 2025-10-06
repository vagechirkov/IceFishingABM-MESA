import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from abm.model import IceFishingModel

FPS = 10
INTERVAL = 1000 // FPS
CMAP_DENSITY = "viridis"
CMAP_FEATURE = "PuOr"
CMAP_SOFTMAX = "cividis"


def new_figure_and_axes():
    """
    Mosaic (fixed):
      sim | explore
      sim | explore
      ts  | ts
    """
    fig = plt.figure(figsize=(15, 12), layout="constrained")
    mosaic = [
        ["sim", "explore"],
        ["sim", "explore"],
        ["ts",  "ts"     ],
    ]
    axes = fig.subplot_mosaic(
        mosaic,
        gridspec_kw=dict(
            width_ratios=[2.5, 2.5],
            height_ratios=[1.5/4, 1.5/4, 1/4]
        )
    )

    sp = axes["explore"].get_subplotspec()
    sg = sp.subgridspec(2, 2, wspace=0.01, hspace=0.01)
    explore_axes = np.array([
        fig.add_subplot(sg[0, 0]),
        fig.add_subplot(sg[0, 1]),
        fig.add_subplot(sg[1, 0]),
        fig.add_subplot(sg[1, 1]),
    ])
    axes["explore"].remove()
    return fig, axes, explore_axes


def get_grid_size(model):
    return getattr(model, "grid_size", getattr(model, "grid", None).width)

def get_catch_times_so_far(model):
    return list(getattr(model, "catch_times", []))

def get_agents(model):
    return list(getattr(model, "agents", []))

def get_agent_positions(model):
    agents = get_agents(model)
    if not agents:
        return np.empty((0,2))
    pos = [getattr(a, "pos") for a in agents]
    return np.asarray(pos, dtype=float)

def get_destinations(model):
    dst = []
    for a in get_agents(model):
        if getattr(a, "is_moving", False) and getattr(a, "destination", None) is not None:
            dst.append(getattr(a, "destination"))
    return np.asarray(dst, dtype=float) if len(dst) else np.empty((0,2))

def get_agent_colors(model):
    colors = []
    for a in get_agents(model):
        if getattr(a, "is_moving", False):
            colors.append("skyblue")
        elif getattr(a, "is_sampling", False):
            colors.append("red" if getattr(a, "is_consuming", False) else "#FFA500")
        else:
            colors.append("skyblue")
    return colors

def get_fish_slice(model):
    fd = getattr(model, "fish_density")
    t_idx = int(getattr(model, "steps_min", 0))
    t_idx = max(0, min(t_idx, fd.shape[2]-1))
    return fd[:, :, t_idx]


def get_exploration_fields_for_agent(agent, grid_size):
    strat = getattr(agent, "exploration_strategy", None)

    social = getattr(strat, "social_feature_kde")
    success = getattr(strat, "success_feature_kde")
    failure = getattr(strat, "failure_feature_kde")
    softmax = getattr(strat, "belief_softmax")

    current_position = np.asarray(
        getattr(agent, "pos", (grid_size / 2, grid_size / 2)), dtype=float
    ).reshape(1, 2)
    other_locs = np.asarray(
        getattr(agent, "other_agent_locs", np.empty((0, 2))), dtype=float
    )
    success_locs = np.asarray(
        getattr(agent, "success_locs", np.empty((0, 2))), dtype=float
    )
    failure_locs = np.asarray(
        getattr(agent, "failure_locs", np.empty((0, 2))), dtype=float
    )

    dest = getattr(agent, "destination", (grid_size / 2, grid_size / 2))
    if isinstance(dest, np.ndarray):
        dest = (float(dest[0]), float(dest[1]))

    return {
        "social": social,
        "success": success,
        "failure": failure,
        "softmax": softmax,
        "current_position": current_position,
        "other_locs": other_locs,
        "success_locs": success_locs,
        "failure_locs": failure_locs,
        "dest": dest,
    }

def get_p_leave_and_sampling_for_agent(agent):
    p = float(getattr(agent.exploitation_strategy, "p_leave"))
    s = bool(getattr(agent, "is_sampling")) or bool(getattr(agent, "is_consuming"))
    return p, s

def build_dynamic_dashboard(model, steps, save_format="gif", agent_idx=0):
    # pick the focus agent
    if not get_agents(model):
        raise RuntimeError("No agents found in model.")
    focus_agent = get_agents(model)[agent_idx]

    fig, axes, explore_axes = new_figure_and_axes()
    ax_sim = axes["sim"]
    ax_ts  = axes["ts"]
    grid_size = get_grid_size(model)

    ax_sim.set_xlim(0, grid_size)
    ax_sim.set_ylim(0, grid_size)
    ax_sim.set_aspect("equal", "box")
    ax_sim.set_xticks([])
    ax_sim.set_yticks([])
    fish_im = ax_sim.imshow(
        getattr(model, "fish_density")[:, :, 0],
        origin="lower",
        extent=[0, grid_size, 0, grid_size],
        cmap=CMAP_DENSITY,
        alpha=0.5,
        zorder=0,
        vmin=0,
        vmax=0.15  # getattr(model, "fish_density").max()
    )
    agent_scat = ax_sim.scatter(
        [], [], edgecolors="black", linewidths=0.7, s=60, color="blue", zorder=3
    )

    extent = (0, grid_size, 0, grid_size)
    exp_titles = ["Social", "Success", "Failure", "Softmax"]
    exp_cmps = [CMAP_FEATURE, CMAP_FEATURE, CMAP_FEATURE, CMAP_SOFTMAX]
    exp_vlims = [(-5, 5), (-5, 5), (-5, 5), (None, None)]
    exp_images = []
    exp_points = []

    fields0 = get_exploration_fields_for_agent(focus_agent, grid_size)
    for k, ax in enumerate(explore_axes):
        title = exp_titles[k]; cmap = exp_cmps[k]; vmin, vmax = exp_vlims[k]
        data = fields0["social"] if k==0 else fields0["success"] if k==1 else fields0["failure"] if k==2 else fields0["softmax"]
        if data is None:
            data = np.zeros((grid_size, grid_size), dtype=float)
        im = ax.imshow(
            data, origin="lower", extent=extent, interpolation="nearest",
            cmap=cmap, **({} if vmin is None else dict(vmin=vmin, vmax=vmax))
        )
        # maximize drawable area: no ticks/labels on right panels
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([])
        # keep titles readable
        ax.set_title(title, fontsize=14, pad=2)
        ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)


        obs_sc = ax.scatter(-1, -1, s=32, facecolors="none", edgecolors="#1b9e77", linewidths=1.1)
        agent_sc = ax.scatter(fields0["current_position"][0,1], fields0["current_position"][0,0],
                              s=60, marker="o", facecolors="none", edgecolors="black", linewidths=1.2)

        dest_sc  = ax.scatter(-1, -1,
                              s=90, marker="*", facecolors="red", edgecolors="black", linewidths=1.0)

        # compact colorbar
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        exp_images.append(im)
        exp_points.append((obs_sc, agent_sc, dest_sc))

    t_hist, p_hist, samp_hist = [], [], []
    (ts_line,) = ax_ts.plot([], [], linewidth=2.0)
    ts_line.set_zorder(2)
    move_fill = [None]
    ax_ts.set_xlabel("Time (minutes)"); ax_ts.set_ylabel("Probability")

    approx_total = getattr(model, "total_minutes", steps / max(getattr(model, "steps_per_minute", 6), 1.0))
    ax_ts.set_xlim(0, max(1, approx_total))
    ax_ts.set_ylim(0, 0.5)
    ax_ts.grid(True, alpha=0.25)

    def update(frame):
        model.step()

        pos = get_agent_positions(model)
        cols = get_agent_colors(model)
        if pos.size:
            agent_scat.set_offsets(pos)
            agent_scat.set_facecolors(cols)
        else:
            agent_scat.set_offsets(np.empty((0,2)))

        fish_im.set_data(get_fish_slice(model))
        ax_sim.set_title(f"Minutes: {model.steps_min}", fontsize=12)

        ef = get_exploration_fields_for_agent(focus_agent, grid_size)
        fields_now = [ef["social"], ef["success"], ef["failure"], ef["softmax"]]
        for k, im in enumerate(exp_images):
            data = fields_now[k]
            if data is not None:
                im.set_data(data)
        for k, (obs_sc, agent_sc, dest_sc) in enumerate(exp_points):
            if k == 0 and ef["other_locs"].size and obs_sc is not None:
                obs_sc.set_offsets(np.c_[ef["other_locs"][:,1], ef["other_locs"][:,0]])
            if k == 1 and ef["success_locs"].size and obs_sc is not None:
                obs_sc.set_offsets(np.c_[ef["success_locs"][:,1], ef["success_locs"][:,0]])
            if k == 2 and ef["failure_locs"].size and obs_sc is not None:
                obs_sc.set_offsets(np.c_[ef["failure_locs"][:,1], ef["failure_locs"][:,0]])
            agent_sc.set_offsets(np.c_[ef["current_position"][0,0], ef["current_position"][0,1]])
            dest_sc.set_offsets(np.c_[ef["dest"][0], ef["dest"][1]])

        p_now, sampling_now = get_p_leave_and_sampling_for_agent(focus_agent)
        t_hist.append(model.steps / model.steps_per_minute)
        p_hist.append(p_now)
        samp_hist.append(bool(sampling_now))

        t_arr = np.asarray(t_hist, dtype=float)
        p_arr = np.asarray(p_hist, dtype=float)
        m_arr = np.asarray(samp_hist, dtype=bool)
        p_plot = p_arr.copy()
        ts_line.set_data(t_arr, p_plot)
        if focus_agent.is_consuming:
            ax_ts.axvline((model.steps - 1) / model.steps_per_minute, linestyle="--", alpha=0.5, color="red")
        if move_fill[0] is not None:
            move_fill[0].remove()

        move_fill[0] = ax_ts.fill_between(
            t_arr, 0.0, 1.0, where=~m_arr,
            color="tab:blue", alpha=0.25, linewidth=0, zorder=1
        )

        return (agent_scat, fish_im, *exp_images, ts_line, *move_fill)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=INTERVAL, blit=False)
    fig.savefig("combined_dashboard_last_frame.png", dpi=150, bbox_inches="tight")

    if save_format.lower() == "git" or save_format.lower() == "both":
        writer = animation.PillowWriter(fps=FPS)
        ani.save("combined_dynamic.gif", writer=writer)

    if save_format.lower() == "gif" or save_format.lower() == "both":
        writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("combined_dynamic.mp4", writer=writer)

    plt.close(fig)


if __name__ == "__main__":
    model = IceFishingModel(grid_size=90, number_of_agents=6, spot_selection_tau=0.1, fish_abundance=3.0)
    build_dynamic_dashboard(model, steps=120*6, save_format="gif", agent_idx=0)
