from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def prepare_heatmap(results_df: pd.DataFrame, param1: str, param2: str, n_steps: int) -> pd.DataFrame:
    results_last_step_df = results_df.groupby([param1, param2]).mean().reset_index()
    results_last_step_df.loc[:, param1] = results_last_step_df[param1].round(3)
    results_last_step_df.loc[:, param2] = results_last_step_df[param2].round(3)
    heatmap_df = results_last_step_df.pivot(index=param1, columns=param2, values='Collected resource') / n_steps
    return heatmap_df


def plot_two_params(heatmap_df: pd.DataFrame, param1: str, param2: str, folder: Path):
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # cmap = "viridis"

    # Draw the heatmap with the mask and correct aspect ratio
    with sns.axes_style("white"):
        g = sns.heatmap(heatmap_df, cmap=cmap,  # vmax=.3, center=0,
                        square=True, linewidths=0, cbar_kws={"shrink": .5, "label": "Resource Collection Rate"})
    g.invert_yaxis()

    plt.tight_layout()
    # sampling_length 2 local_search_counter 5
    plt.savefig(folder / f"{param1} vs. {param2}.png", dpi=300)
    plt.close()


def plot_relocation_weights(heatmap_df: pd.DataFrame, param1: str, param2: str, folder: Path):
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # cmap = "viridis"

    # iterate over the rows and columns
    # for i, row in enumerate(heatmap_df.index):
    #     for j, col in enumerate(heatmap_df.columns):
    #         if row + col > 1:
    #             heatmap_df.iloc[i, j] = np.NAN

    # Draw the heatmap with the mask and correct aspect ratio
    with sns.axes_style("white"):
        g = sns.heatmap(heatmap_df, cmap=cmap,  # vmax=.3, center=0,
                        square=True, linewidths=0, cbar_kws={"shrink": .5, "label": "Resource Collection Rate"})
    g.invert_yaxis()

    # draw the line where the sum of the weights is < 1
    x = np.arange(len(heatmap_df.columns))
    y = []
    for i in x:
        w_1 = heatmap_df.columns[i]
        w_2_max = 1 - w_1
        # find the closest index to w_2_max
        w_2_max_index = np.argmin(np.abs(heatmap_df.index - w_2_max))
        if w_1 + heatmap_df.index[w_2_max_index] < 1:
            w_2_max_index += 1
        y.append(w_2_max_index)

    g.plot(x, y, drawstyle='steps-post', color='black', linewidth=1)

    plt.tight_layout()
    # sampling_length 2 local_search_counter 5
    plt.savefig(folder / f"{param1} vs. {param2} 2.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # read the results
    results_df = pd.read_csv("results.csv")
    df = prepare_heatmap(results_df, "w_social", "w_personal", 1000)
    plot_relocation_weights(df, "w_social", "w_personal", Path(""))
