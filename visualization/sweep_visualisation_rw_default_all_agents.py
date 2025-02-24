import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_sweep_results(sweep_dir):
    # Print all files in directory for debugging
    print("Files in directory:", list(sweep_dir.glob('*')))
    
    # Extract study name correctly - simpler approach
    study_name = "rw_default_filtering"  # Since we know the exact name
    
    # Construct paths using exact filenames
    results_path = sweep_dir / f'{study_name}_sweep_results.csv'
    params_path = sweep_dir / f'{study_name}_parameters.json'
    
    print(f"Looking for results file at: {results_path}")
    
    if not results_path.exists():
        print(f"Results file not found at: {results_path}")
        raise FileNotFoundError(f"Could not find results file: {results_path}")
        
    results_df = pd.read_csv(results_path)
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    
    # Create visualizations directory
    viz_dir = sweep_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    
    # Plot metrics vs alpha for different mu values
    
    metrics = [
        ("avg_efficiency", "se_efficiency", "Average Efficiency"),
        ("avg_resources", "se_resources", "Average Resources Collected"),
        ("avg_time_to_first", "se_time_to_first", "Average Time to First Catch")
    ]
    
    for metric, se_metric, title in metrics:
        plt.figure(figsize=(10, 6))

        plt.xscale('log')
        
        # Create line plot with error bands
        sns.lineplot(
            data=results_df,
            x="alpha",
            y=metric,
            hue="mu",
            marker='o',
            palette="viridis"
        )


        
        # Add error bands manually for each mu value
        for mu_val in results_df['mu'].unique():
            mu_data = results_df[results_df['mu'] == mu_val]
            plt.fill_between(
                mu_data['alpha'],
                mu_data[metric] - mu_data[se_metric],
                mu_data[metric] + mu_data[se_metric],
                alpha=0.2
            )

        # Format x-axis ticks to show scientific notation
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.gca().xaxis.set_major_locator(plt.LogLocator(base=10))
        plt.minorticks_off()
        
        plt.xlabel('Alpha values (log scale)')
        plt.ylabel(title)
        plt.title(f'{title} vs Alpha for Different μ Values\n'
                 f'(Clusters: {params["simulation"]["num_resource_clusters"]}, '
                 f'Steps: {params["simulation"]["max_steps"]})')
        plt.tight_layout()
        plt.savefig(viz_dir / f'{metric}_vs_alpha.png', dpi=300, bbox_inches='tight')
        plt.close()
   

if __name__ == "__main__":
    # Base directory for all parameter sweeps
    base_dir = Path('parameter_sweep_results')
    
    if not base_dir.exists():
        print(f"Base directory {base_dir} not found!")
        exit(1)
    
    # Get all sweep directories
    sweep_dirs = list(base_dir.glob('parameter_sweep_*'))
    
    if not sweep_dirs:
        print("No parameter sweep directories found!")
        exit(1)
    
    print(f"Found {len(sweep_dirs)} sweep directories to process")
    
    # Process each sweep directory
    for sweep_dir in sweep_dirs:
        try:
            plot_sweep_results(sweep_dir)
            print(f"Successfully processed: {sweep_dir.name}")
        except Exception as e:
            print(f"Error processing {sweep_dir.name}: {str(e)}")
    
    print("\nAll visualizations completed!")