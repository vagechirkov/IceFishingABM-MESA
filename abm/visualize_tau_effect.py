import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Ensure imports work regardless of where it's executed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from abm.exploration_strategy import KernelBeliefExploration
except ImportError:
    from exploration_strategy import KernelBeliefExploration

def main():
    grid_size = 90
    _rng = np.random.default_rng(123)
    
    # Initialize the same KernelBeliefExploration strategy 
    # to maintain consistency with `visualize_icefishing_combined.py`
    exp_strat = KernelBeliefExploration(
        grid_size=grid_size,
        tau=0.1, # Will be varied manually later
        social_length_scale=25.0,
        success_length_scale=10.0,
        failure_length_scale=10.0,
        w_social=0.2,
        w_success=0.4,
        w_failure=0.4,
        w_locality=0.0,
        w_as_attention_shares=True,
        model_type="kde",
        normalize_features=True,
        rng=_rng
    )
    
    # Static realistic locations to use across all tau valuations
    current_position = np.array([[45, 45]], dtype=float)
    
    def rand_locs(n):
        return _rng.integers(10, grid_size-10, size=(n, 2), endpoint=False, dtype=np.int64).astype(float)
        
    other_agent_locs = rand_locs(4)
    success_locs = rand_locs(2)
    failure_locs = rand_locs(3)
    
    # Calculate features once so they remain identical
    exp_strat.update_features(
        current_position=current_position,
        success_locs=success_locs,
        failure_locs=failure_locs,
        other_agent_locs=other_agent_locs
    )
    
    belief = exp_strat._compute_kde_beliefs()
    taus = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    # Pre-calculate softmaxes to establish a global color scale
    softmaxes = [exp_strat._softmax(belief, tau=tau).reshape(grid_size, grid_size) for tau in taus]
    vmin = min(np.min(s) for s in softmaxes)
    vmax = max(np.max(s) for s in softmaxes)
    
    # Setup Figure: 1 row for Softmax for each tau.
    fig, axes = plt.subplots(1, len(taus), figsize=(4 * len(taus), 4), constrained_layout=True)
    extent = (0, grid_size, 0, grid_size)
    
    for i, tau in enumerate(taus):
        ax = axes[i]
        
        # Use pre-calculated exact softmax distribution
        belief_softmax = softmaxes[i]
        
        im = ax.imshow(
            belief_softmax,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="cividis",
            vmin=vmin,
            vmax=vmax
        )
        
        ax.set_title(rf"Choice Probability ($\tau = {tau}$)", fontsize=14)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot fixed context (other agents, success, failure, self position)
        social_sc = ax.scatter(other_agent_locs[:, 1], other_agent_locs[:, 0],
                   s=36, facecolors="none", edgecolors="#1b9e77", linewidths=1.2, label="social")
        success_sc = ax.scatter(success_locs[:, 1], success_locs[:, 0],
                   s=36, facecolors="none", edgecolors="white", linewidths=1.2, label="success")
        failure_sc = ax.scatter(failure_locs[:, 1], failure_locs[:, 0],
                   s=36, facecolors="none", edgecolors="red", linewidths=1.2, label="failure")
        agent_sc = ax.scatter(current_position[0, 1], current_position[0, 0],
                   s=150, marker="*", facecolors="red", edgecolors="black", linewidths=1.0, label="agent")
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if i == 0:
            ax.legend(handles=[social_sc, success_sc, failure_sc, agent_sc], loc="upper right", frameon=True, fontsize=9)

    date_now = datetime.now().strftime("%H-%M-%d-%m-%Y")
    filename = f"tau_effect_softmax_{date_now}.pdf"
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    print(f"Saved {filename}")

if __name__ == "__main__":
    main()
