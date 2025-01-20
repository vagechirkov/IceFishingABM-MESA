import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_sweep_results(sweep_dir):
    # Load results
    results_df = pd.read_csv(sweep_dir / 'rw_default_all_agents_sweep_results.csv')
    
    # Load parameters
    with open(sweep_dir / 'parameters.json', 'r') as f:
        params = json.load(f)
    
    # Create visualizations directory
    viz_dir = sweep_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    
    # Plot metrics vs alpha for different mu values
    metrics = [
        ("avg_efficiency", "Average Efficiency"),
        ("avg_resources", "Average Resources Collected"),
        ("avg_time_to_first", "Average Time to First Catch")
    ]
    
    for metric, title in metrics:
        plt.figure(figsize=(10, 6))
        
        # Create line plot
        sns.lineplot(
            data=results_df,
            x="alpha",
            y=metric,
            hue="mu",
            marker='o',
            palette="viridis"
        )
        
        plt.xscale('log')  # Use log scale for alpha
        plt.xlabel('Alpha (log scale)')
        plt.ylabel(title)
        plt.title(f'{title} vs Alpha for Different μ Values')
        
        # Add error bars if needed
        std_col = f"std_{metric.split('avg_')[1]}"
        if std_col in results_df.columns:
            for mu in results_df.mu.unique():
                mu_data = results_df[results_df.mu == mu]
                plt.fill_between(
                    mu_data.alpha,
                    mu_data[metric] - mu_data[std_col],
                    mu_data[metric] + mu_data[std_col],
                    alpha=0.2
                )
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'{metric}_vs_alpha.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Find the most recent sweep directory
    sweep_dirs = list(Path('.').glob('parameter_sweep_*'))
    if not sweep_dirs:
        print("No parameter sweep directories found!")
        exit(1)
    
    latest_sweep = max(sweep_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Processing results from: {latest_sweep}")
    
    plot_sweep_results(latest_sweep)
    print("Visualizations created successfully!")