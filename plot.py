import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data():
    # Load results
    results_df = pd.read_csv('entanglement_results.csv')
    predictions_df = pd.read_csv('predictions_vs_true.csv')
    return results_df, predictions_df

def plot_mse_comparison(results_df):
    plt.figure(figsize=(12, 6))
    methods = ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian', 'Generative']
    
    for method in methods:
        plt.plot(
            results_df['num_measurements'], 
            results_df[f'{method}_MSE'], 
            'o-', 
            label=method,
            markersize=8
        )
    
    plt.xlabel('Number of Measurements')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('MSE vs Number of Measurements')
    plt.tight_layout()
    plt.savefig('plots/mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_computation_time(results_df):
    plt.figure(figsize=(12, 6))
    methods = ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian', 'Generative']
    
    for method in methods:
        plt.plot(
            results_df['num_measurements'], 
            results_df[f'{method}_Time'], 
            'o-', 
            label=method,
            markersize=8
        )
    
    plt.xlabel('Number of Measurements')
    plt.ylabel('Computation Time (seconds)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Computation Time vs Number of Measurements')
    plt.tight_layout()
    plt.savefig('plots/computation_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatters(predictions_df):
    methods = ['mlp', 'cnn', 'transformer', 'mle', 'bayesian', 'generative']
    measurements = predictions_df['num_measurements'].unique()
    
    for n_meas in measurements:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Prediction Comparison for {n_meas} Measurements')
        
        df_subset = predictions_df[predictions_df['num_measurements'] == n_meas]
        
        for idx, method in enumerate(methods):
            row = idx // 3
            col = idx % 3
            
            ax = axes[row, col]
            ax.scatter(
                df_subset['true_value'],
                df_subset[f'{method}_pred'],
                alpha=0.5,
                s=20
            )
            
            # Add perfect prediction line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
            
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.set_title(method.upper())
            ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(f'plots/predictions_{n_meas}_measurements.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_error_distribution_plots(predictions_df):
    methods = ['mlp', 'cnn', 'transformer', 'mle', 'bayesian', 'generative']
    measurements = predictions_df['num_measurements'].unique()
    
    for n_meas in measurements:
        plt.figure(figsize=(12, 6))
        df_subset = predictions_df[predictions_df['num_measurements'] == n_meas]
        
        errors = []
        for method in methods:
            error = df_subset[f'{method}_pred'] - df_subset['true_value']
            errors.append(error)
            
        plt.boxplot(errors, labels=[m.upper() for m in methods])
        plt.title(f'Error Distribution for {n_meas} Measurements')
        plt.ylabel('Prediction Error')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'plots/error_distribution_{n_meas}_measurements.png', dpi=300)
        plt.close()

def main():
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Load data
    results_df, predictions_df = load_data()
    
    # Create plots
    plot_mse_comparison(results_df)
    plot_computation_time(results_df)
    plot_prediction_scatters(predictions_df)
    create_error_distribution_plots(predictions_df)
    
    print("All plots have been generated in the 'plots' directory.")

if __name__ == "__main__":
    main()
