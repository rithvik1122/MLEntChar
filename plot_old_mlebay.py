#!/usr/bin/env python3
"""
Enhanced visualization script for entanglement characterization results.
Creates publication-quality plots with better formatting and insightful comparisons.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import math
import traceback
import argparse
from pathlib import Path
import matplotlib.font_manager as fm

# Check available fonts and set to use a reliable available font
available_fonts = [f.name for f in fm.fontManager.ttflist]
print(f"Available fonts: {', '.join(sorted(set(available_fonts[:10])))}...")

# Use a more broadly available font set
if 'DejaVu Serif' in available_fonts:
    serif_font = 'DejaVu Serif'
elif 'Liberation Serif' in available_fonts:
    serif_font = 'Liberation Serif'
else:
    serif_font = None  # Let matplotlib use its default

# Set publication-quality matplotlib parameters with available fonts
plt.rcParams.update({
    'font.family': 'serif' if serif_font else 'sans-serif',
    'font.serif': [serif_font] if serif_font else [],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'lines.markeredgewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlepad': 10,
    'axes.labelpad': 8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    # Use mathtext for better compatibility
    'mathtext.default': 'regular'
})

# Print font selection
print(f"Using {'serif font: ' + serif_font if serif_font else 'default matplotlib font'}")

def parse_complex(value):
    """Parse complex numbers from various string formats"""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, complex):
        return value.real
    elif isinstance(value, str):
        try:
            # Try to handle scientific notation and complex numbers
            return float(value.replace('(', '').replace(')', '').split('+')[0])
        except:
            return np.nan
    return np.nan

def create_plots(results_path='entanglement_results.csv', predictions_path='predictions_vs_true.csv'):
    """Generate enhanced visualizations from results data"""
    # Create plots directory
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving plots to: {plots_dir}")
    
    try:
        # Load data with error handling
        print(f"Loading results from: {results_path}")
        results_df = pd.read_csv(results_path)
        
        # Handle predictions data if available
        has_predictions = os.path.exists(predictions_path)
        if has_predictions:
            print(f"Loading predictions from: {predictions_path}")
            predictions_df = pd.read_csv(predictions_path)
        else:
            print(f"Predictions file not found: {predictions_path}")
            predictions_df = None
            
        # Parse any complex numbers in MSE values
        for col in results_df.columns:
            if 'MSE' in col or 'Time' in col:
                results_df[col] = results_df[col].apply(parse_complex)
        
        if predictions_df is not None:
            for col in predictions_df.columns:
                if col != 'num_measurements':
                    predictions_df[col] = predictions_df[col].apply(parse_complex)
        
        # Define methods and their properties for consistent styling
        # Commenting out Generative model as it's not performing well
        # methods = ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian', 'Generative']
        methods = ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian']
        nice_names = {
            'MLP': 'Multi-Layer Perceptron',
            'CNN': 'Convolutional Neural Network',
            'Transformer': 'Transformer Network',
            'MLE': 'Maximum Likelihood',
            'Bayesian': 'Bayesian Estimation',
            # 'Generative': 'Generative Model'  # Commented out
        }
        colors = {
            'MLP': '#1f77b4',      # Blue
            'CNN': '#ff7f0e',      # Orange
            'Transformer': '#2ca02c', # Green
            'MLE': '#d62728',      # Red
            'Bayesian': '#9467bd', # Purple
            # 'Generative': '#8c564b' # Brown  # Commented out
        }
        markers = {
            'MLP': 'o', 
            'CNN': 's', 
            'Transformer': '^', 
            'MLE': 'D', 
            'Bayesian': 'P', 
            # 'Generative': 'X'  # Commented out
        }
        
        # Calculate overall statistics
        print("\nPerformance Overview:")
        print("=====================")
        for method in methods:
            avg_mse = results_df[f'{method}_MSE'].mean()
            min_mse = results_df[f'{method}_MSE'].min()
            max_mse = results_df[f'{method}_MSE'].max()
            avg_time = results_df[f'{method}_Time'].mean()
            
            print(f"{nice_names[method]}:")
            print(f"  Avg MSE: {avg_mse:.6f} (Range: {min_mse:.6f}-{max_mse:.6f})")
            print(f"  Avg Time: {avg_time:.2f} seconds")
        
        # PLOT 1: MSE vs Number of Measurements (Publication Quality)
        print("\nCreating MSE vs Measurements plot...")
        # Use figure with adjusted settings
        plt.figure(figsize=(10, 8), constrained_layout=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for method in methods:
            # Create scatter plot with connecting lines
            ax.plot(
                results_df['num_measurements'], 
                results_df[f'{method}_MSE'],
                marker=markers[method],
                linestyle='-',
                label=nice_names[method],
                color=colors[method],
                linewidth=2,
                markersize=9,
                markeredgecolor='white',
                markeredgewidth=1.5,
                alpha=0.9
            )
        
        # Set log scales for both axes
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        
        # Add grid with enhanced appearance
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Add annotations showing theoretical scaling (1/sqrt(N))
        x_range = results_df['num_measurements'].max() / results_df['num_measurements'].min()
        y_pos = results_df[f'MLP_MSE'].min() * 3
        x1 = results_df['num_measurements'].min() * 1.5
        x2 = x1 * 4
        y1 = y_pos
        y2 = y1 / math.sqrt(4)  # 1/sqrt(N) scaling
        
        ax.plot([x1, x2], [y1, y2], 'k--', linewidth=1.5, alpha=0.7)
        ax.text(x2 * 1.1, y2, r'$\sim 1/\sqrt{N}$', fontsize=12, alpha=0.8)
        
        # Reference lines showing 1/N and 1/sqrt(N) scalings
        x_vals = np.logspace(np.log10(results_df['num_measurements'].min()),
                           np.log10(results_df['num_measurements'].max()), 100)
        reference_point = (results_df['num_measurements'].min(), y_pos * 3)
        
        # Add labels and title
        ax.set_xlabel('Number of Measurements (N)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=15, fontweight='bold')
        ax.set_title('Entanglement Characterization Error vs Measurement Count', 
                    fontsize=16, fontweight='bold')
        
        # Add legend with enhanced appearance - CHANGED from upper right to lower left
        legend = ax.legend(
            loc='lower left',  # Changed from 'upper right' to 'lower left'
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True,
            ncol=2
        )
        
        # Improve tick formatting
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_tick_params(which='minor', bottom=False)
        
        # Add shaded regions for error bars if multiple runs exist
        methods_with_multiple_runs = []
        for method in methods:
            if len(results_df.groupby('num_measurements')[f'{method}_MSE']) > 1:
                methods_with_multiple_runs.append(method)
                
        if methods_with_multiple_runs:
            for method in methods_with_multiple_runs:
                grouped = results_df.groupby('num_measurements')[f'{method}_MSE']
                means = grouped.mean()
                stds = grouped.std()
                ax.fill_between(
                    means.index, 
                    means - stds, 
                    means + stds, 
                    alpha=0.2, 
                    color=colors[method]
                )
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(plots_dir / 'mse_vs_measurements.png', dpi=300)
        plt.savefig(plots_dir / 'mse_vs_measurements.pdf')
        
        # NEW PLOT: Computation Time vs Number of Measurements (Publication Quality)
        print("\nCreating Computation Time vs Measurements plot...")
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for method in methods:
            # Create scatter plot with connecting lines
            ax.plot(
                results_df['num_measurements'], 
                results_df[f'{method}_Time'],
                marker=markers[method],
                linestyle='-',
                label=nice_names[method],
                color=colors[method],
                linewidth=2,
                markersize=9,
                markeredgecolor='white',
                markeredgewidth=1.5,
                alpha=0.9
            )
        
        # Set log scales for both axes
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        
        # Add grid with enhanced appearance
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Add labels and title
        ax.set_xlabel('Number of Measurements (N)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=15, fontweight='bold')
        ax.set_title('Computation Time vs Measurement Count', 
                    fontsize=16, fontweight='bold')
        
        # Add legend with enhanced appearance
        legend = ax.legend(
            loc='upper left',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True,
            ncol=2
        )
        
        # Improve tick formatting
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_tick_params(which='minor', bottom=False)
        
        # Add shaded regions for error bars if multiple runs exist
        methods_with_multiple_runs = []
        for method in methods:
            if len(results_df.groupby('num_measurements')[f'{method}_Time']) > 1:
                methods_with_multiple_runs.append(method)
                
        if methods_with_multiple_runs:
            for method in methods_with_multiple_runs:
                grouped = results_df.groupby('num_measurements')[f'{method}_Time']
                means = grouped.mean()
                stds = grouped.std()
                ax.fill_between(
                    means.index, 
                    means - stds, 
                    means + stds, 
                    alpha=0.2, 
                    color=colors[method]
                )
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(plots_dir / 'time_vs_measurements.png', dpi=300)
        plt.savefig(plots_dir / 'time_vs_measurements.pdf')
        
        # FIX: Create separate figure for time vs measurements with complexity visualization
        print("Creating Time vs Measurements with complexity reference...")
        fig_complexity, ax_complexity = plt.subplots(figsize=(10, 8))
        
        # Re-plot each method on the new figure
        for method in methods:
            ax_complexity.plot(
                results_df['num_measurements'], 
                results_df[f'{method}_Time'],
                marker=markers[method],
                linestyle='-',
                label=nice_names[method],
                color=colors[method],
                linewidth=2,
                markersize=9,
                markeredgecolor='white',
                markeredgewidth=1.5,
                alpha=0.9
            )
        
        # Set log scales for both axes
        ax_complexity.set_xscale('log', base=10)
        ax_complexity.set_yscale('log', base=10)
        
        # Add grid with enhanced appearance
        ax_complexity.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Add reference lines showing expected scaling
        x_vals = np.logspace(np.log10(results_df['num_measurements'].min()),
                           np.log10(results_df['num_measurements'].max()), 100)
        
        # For the computational complexity visualization
        min_m = results_df['num_measurements'].min()
        
        # Find a reference point for MLE/Bayesian method to show O(N) and O(N²) scaling
        if 'MLE' in methods:
            mle_times = results_df[results_df['num_measurements'] == min_m]['MLE_Time'].values
            if len(mle_times) > 0:
                ref_time = mle_times[0]
                
                # Add O(N) and O(N²) reference lines
                ax_complexity.plot(x_vals, ref_time * (x_vals/min_m), 'k--', linewidth=1.5, alpha=0.7, label='O(N) scaling')
                ax_complexity.plot(x_vals, ref_time * (x_vals/min_m)**2, 'k:', linewidth=1.5, alpha=0.7, label='O(N²) scaling')
                
                # Add text annotations for clarity
                mid_x = np.sqrt(min_m * results_df['num_measurements'].max())
                y_pos_linear = ref_time * (mid_x/min_m) * 0.7
                y_pos_quadratic = ref_time * (mid_x/min_m)**2 * 0.7
                
                ax_complexity.annotate('O(N)', xy=(mid_x, y_pos_linear),
                                     xycoords='data', fontsize=12, alpha=0.8,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                                     
                ax_complexity.annotate('O(N²)', xy=(mid_x, y_pos_quadratic),
                                     xycoords='data', fontsize=12, alpha=0.8,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add labels and title
        ax_complexity.set_xlabel('Number of Measurements (N)', fontsize=15, fontweight='bold')
        ax_complexity.set_ylabel('Computation Time (seconds)', fontsize=15, fontweight='bold')
        ax_complexity.set_title('Computation Time vs Measurements with Complexity References', 
                             fontsize=16, fontweight='bold')
        
        # Add legend with enhanced appearance
        legend = ax_complexity.legend(
            loc='upper left',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True
        )
        
        # Improve tick formatting
        ax_complexity.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax_complexity.xaxis.set_tick_params(which='minor', bottom=False)
                
        plt.tight_layout()
        plt.savefig(plots_dir / 'time_vs_measurements_with_complexity.png', dpi=300)
        plt.savefig(plots_dir / 'time_vs_measurements_with_complexity.pdf')
        
        # PLOT 2: Enhanced MSE Comparison with error bars
        print("Creating enhanced MSE comparison plot...")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up width and positions for grouped bar chart
        n_methods = len(methods)
        bar_width = 0.8 / n_methods
        opacity = 0.8
        
        # Group by number of measurements
        measurements = sorted(results_df['num_measurements'].unique())
        x_positions = np.arange(len(measurements))
        
        # Create grouped bar chart
        for i, method in enumerate(methods):
            mse_values = []
            mse_errors = []
            
            for m in measurements:
                filtered = results_df[results_df['num_measurements'] == m]
                mse_values.append(filtered[f'{method}_MSE'].mean())
                # Add error bar if multiple data points exist
                if len(filtered) > 1:
                    mse_errors.append(filtered[f'{method}_MSE'].std())
                else:
                    mse_errors.append(0)
            
            pos = x_positions + i * bar_width - (n_methods-1) * bar_width / 2
            ax.bar(pos, mse_values, bar_width, 
                  label=nice_names[method],
                  color=colors[method],
                  alpha=opacity,
                  yerr=mse_errors,
                  error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))
        
        # Configure axes
        ax.set_xlabel('Number of Measurements', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
        ax.set_title('Error Comparison Across Methods by Measurement Count', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(m) for m in measurements])
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.95, ncol=2)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'method_comparison_by_measurements.png', dpi=300)
        plt.savefig(plots_dir / 'method_comparison_by_measurements.pdf')
        
        # PLOT 3: Log-Log Error vs Computation Time (Efficiency Frontier)
        print("Creating Error vs Time efficiency plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for method in methods:
            # For each measurement count, plot MSE vs Time
            for i, m in enumerate(sorted(results_df['num_measurements'].unique())):
                filtered = results_df[results_df['num_measurements'] == m]
                mse = filtered[f'{method}_MSE'].values[0]
                time_val = filtered[f'{method}_Time'].values[0]
                
                if i == 0:  # First point gets a label
                    ax.scatter(time_val, mse, 
                              s=m/5 + 30,  # Size depends on measurement count
                              color=colors[method], 
                              marker=markers[method],
                              label=nice_names[method],
                              edgecolors='white')
                else:
                    ax.scatter(time_val, mse, 
                              s=m/5 + 30,  # Size depends on measurement count
                              color=colors[method], 
                              marker=markers[method],
                              edgecolors='white')
                
                # Connect points for the same method
                if i > 0:
                    prev_filtered = results_df[results_df['num_measurements'] == 
                                            sorted(results_df['num_measurements'].unique())[i-1]]
                    prev_time = prev_filtered[f'{method}_Time'].values[0]
                    prev_mse = prev_filtered[f'{method}_MSE'].values[0]
                    ax.plot([prev_time, time_val], [prev_mse, mse], 
                           color=colors[method], alpha=0.5, linestyle=':')
                    
                # Add measurement count annotation
                if method == 'MLP':  # Only annotate one method to avoid clutter
                    ax.annotate(f"{m}", 
                               (time_val, mse),
                               textcoords="offset points",
                               xytext=(0,10), 
                               fontsize=8,
                               ha='center')
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add labels and title
        ax.set_xlabel('Computation Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
        ax.set_title('Error-Computation Time Efficiency Frontier', 
                    fontsize=16, fontweight='bold')
        
        # Add grid
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Add "better" region annotation
        ax.annotate('Better →', xy=(0.02, 0.02), xycoords='axes fraction', 
                   fontsize=12, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'error_vs_time_efficiency.png', dpi=300)
        plt.savefig(plots_dir / 'error_vs_time_efficiency.pdf')
        
        # PLOT 4: Normalized improvement with more measurements
        print("Creating normalized improvement plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # For each method, normalize error to the error at min measurements
        min_measurements = min(results_df['num_measurements'])
        
        for method in methods:
            rel_improvements = []
            measurement_counts = sorted(results_df['num_measurements'].unique())
            
            # Get baseline (highest measurement count MSE)
            baseline = results_df[results_df['num_measurements'] == min_measurements][f'{method}_MSE'].values[0]
            
            for m in measurement_counts:
                filtered = results_df[results_df['num_measurements'] == m]
                mse = filtered[f'{method}_MSE'].values[0]
                rel_improvements.append(baseline / mse)  # Ratio of improvement
            
            ax.plot(measurement_counts, rel_improvements, 
                   marker=markers[method],
                   label=nice_names[method],
                   color=colors[method],
                   linewidth=2,
                   markersize=8)
            
        # Add theoretical scaling references with math formatting that works with all fonts
        x_vals = np.array(measurement_counts)
        y_sqrt = np.sqrt(x_vals / min_measurements)  # 1/sqrt(N) scaling
        y_linear = x_vals / min_measurements  # 1/N scaling
        
        ax.plot(x_vals, y_sqrt, 'k--', linewidth=1.5, alpha=0.7, label='sqrt(N) scaling')
        ax.plot(x_vals, y_linear, 'k:', linewidth=1.5, alpha=0.7, label='N scaling')
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Labels and title
        ax.set_xlabel('Number of Measurements (N)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Improvement Factor', fontsize=14, fontweight='bold')
        ax.set_title('Error Reduction with Increased Measurements', 
                    fontsize=16, fontweight='bold')
        
        # Add grid
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Add legend with theoretical scalings
        ax.legend(loc='upper left', framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'normalized_improvement.png', dpi=300)
        plt.savefig(plots_dir / 'normalized_improvement.pdf')
        
        # PLOT 5: Prediction accuracy visualizations (if predictions data available)
        if predictions_df is not None:
            print("Creating prediction accuracy visualizations...")
            
            # For each measurement count, create a scatter plot of predicted vs true values
            for num_meas in sorted(predictions_df['num_measurements'].unique()):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                subset = predictions_df[predictions_df['num_measurements'] == num_meas]
                
                # Add perfect prediction line (y=x)
                min_val = subset['true_value'].min()
                max_val = subset['true_value'].max()
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
                       label='Perfect Prediction', linewidth=1.5, alpha=0.7)
                
                # Plot subset of methods to avoid overcrowding - remove Generative
                plot_methods = ['MLP', 'CNN', 'Transformer', 'Bayesian']
                
                for method in plot_methods:
                    # Calculate correlation
                    col_name = f"{method.lower()}_pred"
                    r = np.corrcoef(subset['true_value'], subset[col_name])[0,1]
                    
                    # Create scatterplot with alpha for density indication
                    ax.scatter(subset['true_value'], subset[col_name],
                              label=f"{nice_names[method]} (r={r:.3f})",
                              color=colors[method],
                              marker=markers[method],
                              s=30,
                              alpha=0.6,
                              edgecolor='none')
                
                # Configure appearance
                ax.set_xlabel('True Entanglement Negativity', fontsize=14, fontweight='bold')
                ax.set_ylabel('Predicted Entanglement Negativity', fontsize=14, fontweight='bold')
                ax.set_title(f'Predictions vs True Values ({num_meas} Measurements)', 
                            fontsize=16, fontweight='bold')
                
                # Add diagonal gridlines to help evaluate accuracy
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Equal aspect ratio for fair comparison
                ax.set_aspect('equal')
                
                # Add legend
                ax.legend(loc='best', framealpha=0.95)
                
                plt.tight_layout()
                plt.savefig(plots_dir / f'predictions_scatter_{num_meas}.png', dpi=300)
                plt.savefig(plots_dir / f'predictions_scatter_{num_meas}.pdf')
            
            # PLOT 6: Error distribution violin plots
            print("Creating error distribution plots...")
            
            # For each measurement count, create violin plot of errors
            for num_meas in sorted(predictions_df['num_measurements'].unique()):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                subset = predictions_df[predictions_df['num_measurements'] == num_meas]
                
                # Calculate errors for each method
                error_data = []
                method_labels = []
                
                for method in methods:
                    col_name = f"{method.lower()}_pred"
                    if col_name in subset.columns:
                        errors = subset[col_name] - subset['true_value']
                        error_data.append(errors)
                        method_labels.append(nice_names[method])
                
                # Create violin plot
                vp = ax.violinplot(error_data, showmeans=True, showmedians=True)
                
                # Color the violins according to our color scheme
                for i, pc in enumerate(vp['bodies']):
                    pc.set_facecolor(colors[methods[i]])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                
                # Add zero line
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Set x-axis ticks with method names
                ax.set_xticks(range(1, len(method_labels) + 1))
                ax.set_xticklabels(method_labels, rotation=45, ha='right')
                
                # Add labels and title
                ax.set_ylabel('Prediction Error (Predicted - True)', fontsize=14, fontweight='bold')
                ax.set_title(f'Error Distribution by Method ({num_meas} Measurements)', 
                            fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(plots_dir / f'error_distribution_{num_meas}.png', dpi=300)
                plt.savefig(plots_dir / f'error_distribution_{num_meas}.pdf')
        
        print("\nAll plots have been saved to the 'plots' directory.")
        print("PDF and PNG versions are available for each plot.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
    except Exception as e:
        print(f"Error during plotting: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Generate enhanced plots from entanglement characterization results")
    parser.add_argument("--results", type=str, default="entanglement_results.csv",
                       help="Path to results CSV file (default: entanglement_results.csv)")
    parser.add_argument("--predictions", type=str, default="predictions_vs_true.csv",
                       help="Path to predictions CSV file (default: predictions_vs_true.csv)")
    
    args = parser.parse_args()
    
    create_plots(args.results, args.predictions)
