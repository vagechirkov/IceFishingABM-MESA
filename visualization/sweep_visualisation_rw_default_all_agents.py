import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

def process_all_directories(base_dir, process_all=True):
    # Get all subdirectories
    sweep_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not sweep_dirs:
        print("No directories found!")
        return
    
    if process_all:
        print(f"Found {len(sweep_dirs)} directories to process")
        for sweep_dir in sweep_dirs:
            try:
                plot_sweep_results(sweep_dir)
                print(f"Successfully processed: {sweep_dir.name}")
            except Exception as e:
                print(f"Error processing {sweep_dir.name}: {str(e)}")
    else:
        # Just the latest
        latest_sweep = max(sweep_dirs, key=lambda x: x.stat().st_mtime)
        try:
            plot_sweep_results(latest_sweep)
            print(f"Successfully processed latest: {latest_sweep.name}")
        except Exception as e:
            print(f"Error processing latest: {str(e)}")

def plot_sweep_results(sweep_dir):
    # Find any CSV and JSON files
    csv_files = list(sweep_dir.glob('*.csv'))
    json_files = list(sweep_dir.glob('*.json'))
    
    if not csv_files or not json_files:
        raise FileNotFoundError(f"Missing required files in {sweep_dir}")
    
    # Use the first files found
    results_path = csv_files[0]
    params_path = json_files[0]
    
    print(f"Found files in {sweep_dir.name}:")
    print(f"  - Results: {results_path.name}")
    print(f"  - Parameters: {params_path.name}")
    
    # Load data
    results_df = pd.read_csv(results_path)
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Replace infinity values with NaN to avoid warnings
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    
    # Create visualizations directory
    viz_dir = sweep_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Check if we have multiple mu values
    has_multiple_mu = len(results_df['mu'].unique()) > 1 if 'mu' in results_df.columns else False
    
    # Plot metrics vs alpha
    metrics = [
        ("avg_efficiency", "ci_efficiency", "Average Efficiency"),
        ("avg_resources", "ci_resources", "Average Resources Collected"),
        ("avg_time_to_first", "ci_time", "Average Time to First Catch")
    ]
    
    flip_alpha = True
    alpha_values = np.logspace(-5, 0, 6)  # Creates 6 points evenly spaced in log scale from 10^-5 to 10^0

    for metric, ci_metric, title in metrics:
        if metric not in results_df.columns:
            print(f"Skipping {metric} - not found in results")
            continue
            
        plt.figure(figsize=(10, 6))
        plt.xscale('log')
        

        plot_df = results_df.copy()
        
        # Create plot with or without hue based on mu values
        if has_multiple_mu:
            sns.lineplot(
                data=plot_df,
                x=alpha_values , 
                y=metric,
                hue="mu",
                marker='o',
                palette="viridis",
                errorbar=("ci", 95)
            )
        else:
            sns.lineplot(
                data=plot_df,
                x="alpha",
                y=metric,
                marker='o',
                color="blue",
                errorbar=("ci", 95)
            )
        
        plt.xticks(np.logspace(-5, 0, 6))
            
        # Labels and title
        plt.xlabel('Alpha values (log scale)')
        plt.ylabel(title)
        
        # Get parameters for title if available
        clusters = params.get("simulation", {}).get("num_resource_clusters", "?")
        steps = params.get("simulation", {}).get("max_steps", "?")
        
        plt.title(f'{title} vs Alpha\n(Clusters: {clusters}, Steps: {steps})')
        plt.tight_layout()
        
        # Save figure
        output_file = viz_dir / f'{metric}_vs_alpha.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")
        plt.close()
    
    print(f"Visualizations saved to {viz_dir}")
    return viz_dir
    

if __name__ == "__main__":
    # Base directory for all parameter sweeps
    base_dir = Path('parameter_sweep_results')
    process_all = True  # Toggle between processing all runs (True) or just latest (False)
    
    if not base_dir.exists():
        print(f"Base directory {base_dir} not found!")
        exit(1)
    
    # Get all subdirectories
    sweep_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not sweep_dirs:
        print("No directories found!")
        exit(1)
    
    if process_all:
        print(f"Found {len(sweep_dirs)} directories to process")
        # Process each directory
        for sweep_dir in sweep_dirs:
            try:
                plot_sweep_results(sweep_dir)
                print(f"Successfully processed: {sweep_dir.name}")
            except Exception as e:
                print(f"Error processing {sweep_dir.name}: {str(e)}")
        print("\nAll visualizations completed!")
    else:
        # Process only the latest directory
        latest_sweep = max(sweep_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Processing latest run: {latest_sweep.name}")
        try:
            plot_sweep_results(latest_sweep)
            print(f"Successfully processed latest run")
        except Exception as e:
            print(f"Error processing latest run: {str(e)}")