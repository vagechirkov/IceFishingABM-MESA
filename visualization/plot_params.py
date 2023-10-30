from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def prepare_heatmap(results_df: pd.DataFrame, param1: str, param2: str, n_steps: int) -> pd.DataFrame:
    results_last_step_df = results_df.groupby([param1, param2]).mean().reset_index()
    results_last_step_df.loc[:, param1] = results_last_step_df[param1].round(3)
    results_last_step_df.loc[:, param2] = results_last_step_df[param2].round(3)
    heatmap_df = results_last_step_df.pivot(index=param1, columns=param2, values='Collected resource') / n_steps
    return heatmap_df


def plot_two_params(heatmap_df: pd.DataFrame, param1: str, param2: str, n_repetitions: int, folder: Path):
    with sns.axes_style("white"):
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(7, 7))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # cmap = "viridis"

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(heatmap_df, cmap=cmap,  # vmax=.3, center=0,
                    square=True, linewidths=0, cbar_kws={"shrink": .5, "label": "Resource Collection Rate"})
    g.invert_yaxis()

    plt.tight_layout()
    # sampling_length 2 local_search_counter 5
    plt.savefig(folder / f"{param1} vs. {param2}.png", dpi=300)
    plt.close()
