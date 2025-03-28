#!/usr/bin/env python3
"""
Generate detailed error analysis tables and visualizations for different entanglement ranges.
This script analyzes prediction errors across different entanglement ranges and creates
visualizations to better understand method performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Define consistent aesthetics for methods
METHOD_STYLES = {
    'mlp': {
        'name': 'MLP',
        'nice_name': 'Multi-Layer Perceptron',
        'color': '#1f77b4',   # Blue
        'marker': 'o',
        'linestyle': '-',
    },
    'cnn': {
        'name': 'CNN',
        'nice_name': 'Convolutional Neural Network',
        'color': '#ff7f0e',   # Orange
        'marker': 's',
        'linestyle': '-',
    },
    'transformer': {
        'name': 'Transformer',
        'nice_name': 'Transformer Network',
        'color': '#2ca02c',   # Green
        'marker': '^',
        'linestyle': '-',
    },
    'mle': {
        'name': 'MLE',
        'nice_name': 'Maximum Likelihood',
        'color': '#d62728',   # Red
        'marker': 'D',
        'linestyle': '-',
    },
    'bayesian': {
        'name': 'Bayesian',
        'nice_name': 'Bayesian Estimation',
        'color': '#9467bd',   # Purple
        'marker': 'P',
        'linestyle': '-',
    }
}

# Set publication-quality matplotlib parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.labelpad': 8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

def analyze_range_specific_errors(data_dir="prediction_outputs/benchmark_data", output_dir=None):
    """
    Analyze prediction errors across different entanglement ranges
    
    Args:
        data_dir: Directory containing prediction CSV files
        output_dir: Directory to save output files (default: derived from data_dir)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(data_dir), "error_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Error analysis results will be saved to: {output_dir}")
    
    # Find all prediction CSV files
    prediction_files = []
    for file in os.listdir(data_dir):
        if file.startswith('predictions_') and file.endswith('.csv'):
            try:
                meas = int(file.replace('predictions_', '').replace('.csv', ''))
                prediction_files.append((meas, os.path.join(data_dir, file)))
            except ValueError:
                continue
    
    # Sort by measurement count
    prediction_files.sort()
    
    if not prediction_files:
        print(f"No prediction files found in {data_dir}")
        return
    
    print(f"Found {len(prediction_files)} prediction files")
    
    # Define entanglement ranges for analysis
    ranges = [(0.0, 0.3, "Low"), (0.3, 0.9, "Medium"), (0.9, 1.5, "High")]
    
    # Results dataframe to store aggregated results
    results_data = {
        'Measurements': [],
        'Method': [],
        'Range': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'Bias': [],
        'Bias Direction': [],
        'Sample Count': []
    }
    
    # Process each prediction file
    for meas, file_path in prediction_files:
        print(f"Analyzing predictions for {meas} measurements...")
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            if 'true_values' not in df.columns:
                print(f"Error: 'true_values' column not found in {file_path}")
                continue
            
            # Extract true values
            true_values = df['true_values'].values
            
            # Create figure for this measurement count
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Error Analysis by Entanglement Range ({meas} Measurements)", fontsize=16)
            axes = axes.flatten()
            ax_idx = 0
            
            # Process each method
            for col in [c for c in df.columns if c != 'true_values']:
                method = col.replace('_pred', '').lower()
                
                if method not in METHOD_STYLES:
                    continue
                
                preds = df[col].values
                errors = preds - true_values
                
                # Calculate global metrics
                overall_mse = np.mean(errors**2)
                overall_rmse = np.sqrt(overall_mse)
                overall_mae = np.mean(np.abs(errors))
                overall_bias = np.mean(errors)
                
                # Add to results
                results_data['Measurements'].append(meas)
                results_data['Method'].append(method)
                results_data['Range'].append("Overall")
                results_data['MSE'].append(overall_mse)
                results_data['RMSE'].append(overall_rmse)
                results_data['MAE'].append(overall_mae)
                results_data['Bias'].append(overall_bias)
                results_data['Bias Direction'].append("Overestimate" if overall_bias > 0 else "Underestimate")
                results_data['Sample Count'].append(len(true_values))
                
                # Plot error vs true value
                if ax_idx < len(axes):
                    ax = axes[ax_idx]
                    ax_idx += 1
                    
                    # Create scatter plot of errors vs true values
                    scatter = ax.scatter(
                        true_values, 
                        errors,
                        s=20,
                        alpha=0.6,
                        c=true_values,  # Color by true value for better visualization
                        cmap='viridis',
                        label=METHOD_STYLES[method]['nice_name']
                    )
                    
                    # Add zero line
                    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    
                    # Add range dividers
                    for min_val, max_val, _ in ranges:
                        ax.axvline(x=min_val, color='gray', linestyle='--', alpha=0.3)
                        ax.axvline(x=max_val, color='gray', linestyle='--', alpha=0.3)
                    
                    # Add range labels
                    for min_val, max_val, label in ranges:
                        ax.text((min_val + max_val) / 2, ax.get_ylim()[0] * 0.9, 
                               label, ha='center', fontsize=10, 
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
                    
                    # Add regression line
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(true_values, errors)
                    reg_x = np.array([min(true_values), max(true_values)])
                    reg_y = intercept + slope * reg_x
                    ax.plot(reg_x, reg_y, 'k--', alpha=0.7)
                    
                    # Add text with statistics
                    stats_text = f"{METHOD_STYLES[method]['nice_name']}\n"
                    stats_text += f"MSE: {overall_mse:.4f}\n"
                    stats_text += f"Bias: {overall_bias:.4f}\n"
                    stats_text += f"Trend: {slope:.2f}x{'+' if intercept>0 else ''}{intercept:.2f}"
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9, 
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    # Labels
                    ax.set_xlabel('True Entanglement', fontsize=12)
                    ax.set_ylabel('Error (Predicted - True)', fontsize=12)
                    ax.set_title(f"{METHOD_STYLES[method]['nice_name']} Error Pattern", fontsize=14)
                    
                    # Add colorbar
                    plt.colorbar(scatter, ax=ax, label='True Entanglement Value')
                
                # Analyze each range
                for min_val, max_val, label in ranges:
                    range_mask = (true_values >= min_val) & (true_values < max_val)
                    
                    if sum(range_mask) > 0:
                        range_true = true_values[range_mask]
                        range_pred = preds[range_mask]
                        range_errors = errors[range_mask]
                        
                        range_mse = np.mean(range_errors**2)
                        range_rmse = np.sqrt(range_mse)
                        range_mae = np.mean(np.abs(range_errors))
                        range_bias = np.mean(range_errors)
                        
                        # Add to results
                        results_data['Measurements'].append(meas)
                        results_data['Method'].append(method)
                        results_data['Range'].append(label)
                        results_data['MSE'].append(range_mse)
                        results_data['RMSE'].append(range_rmse)
                        results_data['MAE'].append(range_mae)
                        results_data['Bias'].append(range_bias)
                        results_data['Bias Direction'].append("Overestimate" if range_bias > 0 else "Underestimate")
                        results_data['Sample Count'].append(sum(range_mask))
            
            # Save the error analysis plot
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(output_dir, f'error_analysis_{meas}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, 'detailed_error_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved detailed error analysis to {csv_path}")
    
    # Create a more readable summary table
    summary_rows = []
    for meas in sorted(results_df['Measurements'].unique()):
        for method in [m for m in METHOD_STYLES.keys() if m in results_df['Method'].unique()]:
            method_data = results_df[(results_df['Measurements'] == meas) & 
                                     (results_df['Method'] == method)]
            
            row = {
                'Measurements': meas,
                'Method': METHOD_STYLES[method]['nice_name']
            }
            
            # Add overall stats
            overall = method_data[method_data['Range'] == 'Overall']
            if len(overall) > 0:
                row['Overall MSE'] = f"{overall['MSE'].values[0]:.6f}"
                row['Overall Bias'] = f"{overall['Bias'].values[0]:.3f}"
            
            # Add range-specific stats
            for _, range_row in method_data[method_data['Range'] != 'Overall'].iterrows():
                prefix = range_row['Range']
                row[f"{prefix} MSE"] = f"{range_row['MSE']:.6f}"
                row[f"{prefix} Bias"] = f"{range_row['Bias']:.3f} ({range_row['Bias Direction']})"
            
            summary_rows.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'error_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved error summary to {summary_path}")
    
    # Create a text file in a format easy to include in the paper
    paper_path = os.path.join(output_dir, 'paper_error_analysis.txt')
    with open(paper_path, 'w') as f:
        f.write("ERROR ANALYSIS BY ENTANGLEMENT RANGE FOR PAPER\n")
        f.write("===========================================\n\n")
        
        for meas in [20, 100]:  # Focus on these key measurement counts for the paper
            f.write(f"\n{meas} MEASUREMENTS\n")
            f.write("-" * 20 + "\n\n")
            
            for range_label in ["Low", "Medium", "High"]:
                f.write(f"{range_label} Entanglement Range:\n")
                
                range_data = results_df[(results_df['Measurements'] == meas) & 
                                      (results_df['Range'] == range_label)]
                
                for _, row in range_data.iterrows():
                    method = row['Method']
                    method_name = METHOD_STYLES[method]['nice_name'] if method in METHOD_STYLES else method
                    
                    f.write(f"  {method_name}: MSE={row['MSE']:.6e}, ")
                    f.write(f"Bias={row['Bias']:.3f} ({row['Bias Direction']})\n")
                
                f.write("\n")
    
    print(f"Saved paper-ready error analysis to {paper_path}")
    return results_df

def main():
    # Default data directory
    data_dir = "prediction_outputs/benchmark_data"
    
    # Check if the default directory exists
    if not os.path.exists(data_dir):
        print(f"Default data directory '{data_dir}' not found.")
        data_dir = input("Enter the path to your data directory: ")
        if not os.path.exists(data_dir):
            print(f"Error: Directory '{data_dir}' does not exist.")
            return 1
    
    try:
        print(f"Analyzing prediction errors across entanglement ranges...")
        results_df = analyze_range_specific_errors(data_dir)
        print("Error analysis completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
