import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set visual style
sns.set_theme(style="whitegrid")
sns.set_context("talk")

def main():
    # File configuration
    data_path = "abm/data/temperature_by_compID_wide.csv"
    output_path = "abm/empirical_findings_summary.pdf"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    # 1. Beta to Temperature transformation (T = 1 / beta)
    df["Temperature (T=1/beta)"] = 1.0 / df["temperature"]
    
    # 2. Renormalize weights excluding edge distance
    # Original weights: social, success, loss, edge_distance
    # We want to recompute social, success, loss such that they sum to 1.
    weights_to_keep = ["w_star_social", "w_star_success", "w_star_loss"]
    w_sum = df[weights_to_keep].sum(axis=1)
    
    # Renormalize each in a new column
    for w in weights_to_keep:
        df[f"{w}_norm"] = df[w] / w_sum

    # Variables to plot
    norm_weights = [f"{w}_norm" for w in weights_to_keep]
    vars_to_plot = ["Temperature (T=1/beta)"] + norm_weights
    
    # Reshape data to long format for categorical plotting
    df_long = df.melt(
        id_vars=["compID"], 
        value_vars=vars_to_plot, 
        var_name="Parameter", 
        value_name="Estimate"
    )
    
    # Clean up labels for a nicer plot
    label_map = {
        "Temperature (T=1/beta)": "Decision\nStochasticity\n(Temperature)",
        "w_star_social_norm": "Social Info",
        "w_star_success_norm": "Success Info",
        "w_star_loss_norm": "Failure Info"
    }
    df_long["Parameter"] = df_long["Parameter"].map(label_map)

    # 3. Create the plot on a single axis
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Raw results (individual participants)
    sns.stripplot(
        data=df_long,
        x="Parameter",
        y="Estimate",
        hue="Parameter",
        alpha=0.3, # Semi-transparent for raw data
        jitter=True,
        palette="Set1",
        legend=False,
        ax=ax
    )

    # Summary (Mean and 95% CI)
    sns.pointplot(
        data=df_long,
        x="Parameter",
        y="Estimate",
        hue="Parameter",
        errorbar="ci",
        linestyles="none",
        markers="o",
        markersize=12, # Size of markers for mean
        palette="Set1",
        legend=False,
        ax=ax
    )
    
    # Refine aesthetics
    ax.set_ylabel("Estimate (Mean ± 95% CI)")
    ax.set_xlabel("")
    ax.set_ylim(0, None)
    plt.xticks(rotation=0)
    plt.subplots_adjust(top=0.9)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Summary point plot (single axis) saved to {output_path}")

if __name__ == "__main__":
    main()
