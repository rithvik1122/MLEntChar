#!/usr/bin/env python3
"""
Script to load saved models and make predictions on new data.
This allows using the best neural networks for inference without retraining.
Also includes functionality to compare performance against MLE and Bayesian methods.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Import from ML classes from entchar_old_mlebay.py
from entchar_old_mlebay import (
    MLP, CNN, Transformer, device,
    entanglement_negativity, predict_in_batches, evaluate_generative_model,
    generate_data_with_mixture, parallel_mle_estimator, parallel_bayesian_estimator
)

# Try to import EnhancedDensityMatrixVAE or use regular DensityMatrixVAE as fallback
try:
    from entchar_old_mlebay import EnhancedDensityMatrixVAE
except ImportError:
    from entchar_old_mlebay import DensityMatrixVAE as EnhancedDensityMatrixVAE

def load_model(model_type, num_measurements, model_dir="saved_models"):
    """
    Load a saved model of the specified type and number of measurements
    
    Args:
        model_type: One of 'mlp', 'cnn', 'transformer', or 'generative'
        num_measurements: Number of measurements the model was trained on
        model_dir: Directory containing saved models
        
    Returns:
        Loaded model
    """
    # First load normalization parameters to get input dimension
    norm_params_path = os.path.join(model_dir, f"norm_params_{num_measurements}.npz")
    if not os.path.exists(norm_params_path):
        raise FileNotFoundError(f"Normalization parameters not found at {norm_params_path}")
        
    # Load normalization parameters
    norm_data = np.load(norm_params_path, allow_pickle=True)
    input_dim = int(norm_data['input_dim'])
    
    # Path to model file
    model_path = os.path.join(model_dir, f"{model_type.lower()}_model_{num_measurements}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Create appropriate model architecture
    if model_type.lower() == 'mlp':
        model = MLP(input_dim).to(device)
    elif model_type.lower() == 'cnn':
        model = CNN(input_dim).to(device)
    elif model_type.lower() == 'transformer':
        model = Transformer(input_dim).to(device)
    elif model_type.lower() == 'generative':
        # For generative model, we need to determine latent_dim
        latent_dim = min(256, max(64, input_dim * 3 // 2))
        model = EnhancedDensityMatrixVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dm_dim=16
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    print(f"Successfully loaded {model_type} model trained on {num_measurements} measurements")
    return model, norm_data

def normalize_new_data(X, norm_data):
    """Normalize new data using parameters from training"""
    mean = norm_data['mean']
    std = norm_data['std']
    # Avoid division by zero
    std = np.where(std < 1e-6, 1e-6, std)
    # Apply the same scaling as in training
    scale_factor = np.sqrt(X.shape[1])
    X_norm = (X - mean) / (std * scale_factor)
    return X_norm

def predict_with_model(model, X, model_type, norm_data, batch_size=64):
    """Make predictions using loaded model"""
    # Normalize input data
    X_norm = normalize_new_data(X, norm_data)
    
    if model_type.lower() == 'generative':
        # For generative model we need to use evaluate_generative_model
        predictions = evaluate_generative_model(model, X_norm)
    else:
        # For other models we can use predict_in_batches
        povms = norm_data['povms'] if 'povms' in norm_data else None
        predictions = predict_in_batches(model, X_norm, povms=povms, batch_size=batch_size)
    
    return predictions

def create_test_data(num_measurements, num_samples=100, model_dir="saved_models"):
    """Create some test data using the saved normalization params"""
    norm_params_path = os.path.join(model_dir, f"norm_params_{num_measurements}.npz")
    norm_data = np.load(norm_params_path, allow_pickle=True)
    
    # Generate random measurements within the expected range
    mean = norm_data['mean']
    std = norm_data['std']
    
    # Generate random data around the expected distribution
    X_test = np.random.normal(loc=mean, scale=std, size=(num_samples, len(mean)))
    
    # Rescale to [0, 1] range for measurement probabilities
    X_test = np.clip(X_test, 0, 1)
    
    return X_test

def benchmark_models(measurements_list, test_samples=500, batch_size=64, model_dir="saved_models", 
                     models_to_test=None, include_traditional=True, output_dir="prediction_outputs"):
    """
    Benchmark models across different measurement counts against MLE and Bayesian methods
    
    Args:
        measurements_list: List of measurement counts to test
        test_samples: Number of test samples to generate for each measurement count
        batch_size: Batch size for predictions
        model_dir: Directory where models are stored
        models_to_test: List of model types to test ('mlp', 'cnn', 'transformer', 'generative')
        include_traditional: Whether to include MLE and Bayesian methods
        output_dir: Base directory to save all outputs
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create data directory for CSV files
    data_dir = os.path.join(output_dir, "benchmark_data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Outputs will be saved to '{output_dir}' directory")
    print(f"Plots will be saved to '{plot_dir}' directory")
    print(f"Data files will be saved to '{data_dir}' directory")
    
    # Default to all models if none specified
    if models_to_test is None:
        models_to_test = ['mlp', 'cnn', 'transformer', 'generative']
    
    # Results dictionary to store performance metrics
    results = {
        'measurements': [],
        'method': [],
        'mse': [],
        'time': []
    }
    
    # Define colors for consistent plotting
    colors = {
        'mlp': '#1f77b4',      # Blue
        'cnn': '#ff7f0e',      # Orange
        'transformer': '#2ca02c', # Green
        'generative': '#8c564b', # Brown
        'mle': '#d62728',      # Red
        'bayesian': '#9467bd', # Purple
    }
    
    # Define markers for consistent plotting
    markers = {
        'mlp': 'o', 
        'cnn': 's', 
        'transformer': '^', 
        'generative': 'X',
        'mle': 'D',
        'bayesian': 'P'
    }
    
    # Process each measurement count
    for num_measurements in measurements_list:
        print(f"\n{'='*60}")
        print(f"Testing with {num_measurements} measurements...")
        
        # Generate test data
        print(f"Generating test dataset with {test_samples} samples...")
        X, Y, povms = generate_data_with_mixture(test_samples, num_measurements)
        print(f"Generated data shapes - X: {X.shape}, Y: {Y.shape}")
        
        # Ground truth data for evaluation
        y_true = Y
        
        # Test each neural network model
        for model_type in models_to_test:
            try:
                # Check if model exists
                model_path = os.path.join(model_dir, f"{model_type}_model_{num_measurements}.pt")
                if not os.path.exists(model_path):
                    print(f"Model {model_type} for {num_measurements} measurements not found. Skipping.")
                    continue
                
                print(f"\nEvaluating {model_type} model...")
                
                # Load model
                model, norm_data = load_model(model_type, num_measurements, model_dir)
                
                # Time prediction
                start_time = time.time()
                predictions = predict_with_model(model, X, model_type, norm_data, batch_size)
                pred_time = time.time() - start_time
                
                # Calculate MSE
                mse = np.mean((predictions - y_true) ** 2)
                
                # Store results
                results['measurements'].append(num_measurements)
                results['method'].append(model_type)
                results['mse'].append(mse)
                results['time'].append(pred_time)
                
                print(f"{model_type.upper()} - MSE: {mse:.6f}, Time: {pred_time:.2f}s")
                
            except Exception as e:
                print(f"Error evaluating {model_type} model: {e}")
        
        # Test traditional methods if requested
        if include_traditional:
            # MLE method
            try:
                print("\nEvaluating MLE method...")
                start_time = time.time()
                mle_predictions = parallel_mle_estimator(X, povms)
                mle_time = time.time() - start_time
                mle_mse = np.mean((np.array(mle_predictions) - y_true)**2)
                
                # Store results
                results['measurements'].append(num_measurements)
                results['method'].append('mle')
                results['mse'].append(mle_mse)
                results['time'].append(mle_time)
                
                print(f"MLE - MSE: {mle_mse:.6f}, Time: {mle_time:.2f}s")
            except Exception as e:
                print(f"Error evaluating MLE method: {e}")
            
            # Bayesian method
            try:
                print("\nEvaluating Bayesian method...")
                start_time = time.time()
                bayesian_predictions = parallel_bayesian_estimator(X, povms)
                bayesian_time = time.time() - start_time
                bayesian_mse = np.mean((np.array(bayesian_predictions) - y_true)**2)
                
                # Store results
                results['measurements'].append(num_measurements)
                results['method'].append('bayesian')
                results['mse'].append(bayesian_mse)
                results['time'].append(bayesian_time)
                
                print(f"Bayesian - MSE: {bayesian_mse:.6f}, Time: {bayesian_time:.2f}s")
            except Exception as e:
                print(f"Error evaluating Bayesian method: {e}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    results_file = os.path.join(data_dir, 'benchmark_results.csv')
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Create comparison plots
    # 1. MSE vs Number of Measurements
    plt.figure(figsize=(12, 8))
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(
            method_data['measurements'], 
            method_data['mse'], 
            marker=markers[method],
            linestyle='-',
            color=colors[method],
            label=method.upper(),
            linewidth=2,
            markersize=10
        )
    
    plt.xlabel('Number of Measurements', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.title('MSE vs Number of Measurements Comparison', fontsize=16)
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    
    mse_plot_file = os.path.join(plot_dir, 'mse_vs_measurements_comparison.png')
    plt.savefig(mse_plot_file, dpi=300)
    print(f"MSE comparison plot saved to {mse_plot_file}")
    
    # 2. Time vs Number of Measurements
    plt.figure(figsize=(12, 8))
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(
            method_data['measurements'], 
            method_data['time'], 
            marker=markers[method],
            linestyle='-',
            color=colors[method],
            label=method.upper(),
            linewidth=2,
            markersize=10
        )
    
    plt.xlabel('Number of Measurements', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.title('Computation Time vs Number of Measurements', fontsize=16)
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    
    time_plot_file = os.path.join(plot_dir, 'time_vs_measurements_comparison.png')
    plt.savefig(time_plot_file, dpi=300)
    print(f"Time comparison plot saved to {time_plot_file}")
    
    # 3. Efficiency Plot (MSE vs Time)
    plt.figure(figsize=(12, 8))
    
    # Define markers for different measurement counts
    measurement_markers = {
        10: 'o',
        20: 's',
        50: '^',
        100: 'D',
        250: 'P',
        400: '*'
    }
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        for _, row in method_data.iterrows():
            meas = row['measurements']
            mse = row['mse']
            comp_time = row['time']
            
            marker = measurement_markers.get(meas, 'o')
            
            # Plot point
            plt.scatter(
                comp_time,
                mse,
                s=100,
                marker=marker,
                color=colors[method],
                edgecolors='white',
                label=f"{method.upper()} - {meas} meas." if meas == method_data['measurements'].iloc[0] else ""
            )
            
        # Connect points for this method
        plt.plot(
            method_data['time'], 
            method_data['mse'], 
            linestyle='--',
            alpha=0.5,
            color=colors[method]
        )
    
    plt.xlabel('Computation Time (seconds)', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.title('Efficiency Frontier: Error vs Computation Time', fontsize=16)
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add "better" region annotation
    plt.annotate('Better →', xy=(0.02, 0.02), xycoords='axes fraction', 
               fontsize=12, fontweight='bold', color='green',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.tight_layout()
    
    efficiency_plot_file = os.path.join(plot_dir, 'efficiency_frontier.png')
    plt.savefig(efficiency_plot_file, dpi=300)
    print(f"Efficiency frontier plot saved to {efficiency_plot_file}")
    
    # Save raw data for each measurement count
    for num_measurements in measurements_list:
        measurements_data = {}
        measurements_data['true_values'] = y_true
        
        for method in df['method'].unique():
            method_data = df[(df['method'] == method) & (df['measurements'] == num_measurements)]
            if not method_data.empty:
                # Find the corresponding predictions and save them
                if method == 'mlp' and 'mlp_predictions' in locals():
                    measurements_data['mlp'] = mlp_predictions
                elif method == 'cnn' and 'cnn_predictions' in locals():
                    measurements_data['cnn'] = cnn_predictions
                elif method == 'transformer' and 'transformer_predictions' in locals():
                    measurements_data['transformer'] = transformer_predictions
                elif method == 'generative' and 'generative_predictions' in locals():
                    measurements_data['generative'] = generative_predictions
                elif method == 'mle' and 'mle_predictions' in locals():
                    measurements_data['mle'] = mle_predictions
                elif method == 'bayesian' and 'bayesian_predictions' in locals():
                    measurements_data['bayesian'] = bayesian_predictions
        
        # Save measurements data to file
        measurements_file = os.path.join(data_dir, f'predictions_{num_measurements}.npz')
        np.savez(measurements_file, **measurements_data)
        print(f"Raw prediction data for {num_measurements} measurements saved to {measurements_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Predict with saved models')
    parser.add_argument('--measurements', type=int, default=None, 
                        help='Number of measurements to use (if not specified, will test all available models)')
    parser.add_argument('--model', type=str, default='all',
                        choices=['mlp', 'cnn', 'transformer', 'generative', 'all'],
                        help='Model type to load')
    parser.add_argument('--data-file', type=str, default=None,
                        help='File with measurement data (if not provided, random test data will be used)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to generate if no data file is provided')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for predictions')
    parser.add_argument('--output-dir', type=str, default='prediction_outputs',
                        help='Directory to save all outputs')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison benchmarks against MLE and Bayesian methods')
    parser.add_argument('--no-traditional', action='store_true',
                        help='Skip traditional methods (MLE, Bayesian) in comparisons')
    parser.add_argument('--benchmark-samples', type=int, default=500,
                        help='Number of samples for benchmarking (only used with --compare)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs will be saved to '{output_dir}' directory")
    
    # Create subdirectories for different output types
    plots_dir = os.path.join(output_dir, 'plots')
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if model directory exists
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found.")
        return 1
    
    # Check if we're running comparison benchmarks
    if args.compare:
        print("Running comparison benchmarks...")
        
        # Determine which models to test
        if args.model == 'all':
            models_to_test = ['mlp', 'cnn', 'transformer', 'generative']
        else:
            models_to_test = [args.model]
        
        # Find available measurement counts if not specified
        if args.measurements is None:
            available_measurements = []
            for file in os.listdir(model_dir):
                if file.startswith('mlp_model_') and file.endswith('.pt'):
                    try:
                        meas = int(file.split('_')[-1].split('.')[0])
                        available_measurements.append(meas)
                    except ValueError:
                        pass
            
            if not available_measurements:
                print("Error: No models found in model directory.")
                return 1
            
            available_measurements.sort()
            print(f"Found models for measurements: {available_measurements}")
            measurements_list = available_measurements
        else:
            measurements_list = [args.measurements]
        
        # Run benchmarks with output directory
        benchmark_models(
            measurements_list=measurements_list,
            test_samples=args.benchmark_samples,
            batch_size=args.batch_size,
            model_dir=model_dir,
            models_to_test=models_to_test,
            include_traditional=not args.no_traditional,
            output_dir=output_dir
        )
        
        return 0
    
    # Regular prediction mode
    if args.measurements is None:
        print("Error: For prediction mode, --measurements must be specified.")
        return 1
    
    # Load data or generate test data
    if args.data_file:
        try:
            data = np.load(args.data_file)
            X = data['X'] if 'X' in data else data['measurements']
            print(f"Loaded {len(X)} samples from {args.data_file}")
            
            # Check if data has the right dimension
            norm_params_path = os.path.join(model_dir, f"norm_params_{args.measurements}.npz")
            norm_data = np.load(norm_params_path, allow_pickle=True)
            expected_dim = int(norm_data['input_dim'])
            
            if X.shape[1] != expected_dim:
                print(f"Warning: Model expects {expected_dim} measurements but data has {X.shape[1]}.")
                if X.shape[1] > expected_dim:
                    print(f"Truncating data to first {expected_dim} measurements.")
                    X = X[:, :expected_dim]
                else:
                    print(f"Padding data with zeros to match expected dimension.")
                    pad_width = ((0, 0), (0, expected_dim - X.shape[1]))
                    X = np.pad(X, pad_width, mode='constant')
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
    else:
        print(f"No data file provided. Generating {args.samples} random test samples.")
        X = create_test_data(args.measurements, args.samples)
    
    # List of models to run
    model_types = ['mlp', 'cnn', 'transformer', 'generative'] if args.model == 'all' else [args.model]
    
    # Run predictions with each model
    all_predictions = {}
    
    for model_type in model_types:
        try:
            # Load model
            print(f"\nLoading {model_type} model...")
            model, norm_data = load_model(model_type, args.measurements)
            
            # Make predictions
            print(f"Predicting with {model_type} model...")
            start_time = time.time()
            predictions = predict_with_model(model, X, model_type, norm_data, args.batch_size)
            pred_time = time.time() - start_time
            
            print(f"{model_type} predictions - shape: {predictions.shape}, time: {pred_time:.2f}s")
            print(f"Stats - Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Avg: {predictions.mean():.4f}")
            
            # Store predictions
            all_predictions[model_type] = predictions
            
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
    
    # Store all predictions in output directory
    output_file = os.path.join(data_dir, f'predictions_{args.measurements}.npz')
    np.savez(output_file, **all_predictions)
    print(f"\nAll predictions saved to {output_file}")
    
    # Plot histogram of predictions and save to plots directory
    plt.figure(figsize=(12, 8))
    for model_type, preds in all_predictions.items():
        plt.hist(preds, alpha=0.5, bins=30, label=model_type)
    
    plt.xlabel('Predicted Entanglement Negativity')
    plt.ylabel('Frequency')
    plt.title(f'Prediction Distribution ({args.measurements} measurements)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot in plots directory
    plot_file = os.path.join(plots_dir, f'predictions_{args.measurements}m.png')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    
    # Also generate CSV file for easy analysis
    csv_file = os.path.join(data_dir, f'predictions_{args.measurements}.csv')
    
    # Create a DataFrame from predictions for CSV export
    csv_data = {}
    for model_type, preds in all_predictions.items():
        csv_data[f'{model_type}_pred'] = preds
    
    # If we generated random data, also include the input measurements
    if args.data_file is None:
        for i in range(X.shape[1]):
            csv_data[f'measurement_{i+1}'] = X[:, i]
    
    pd.DataFrame(csv_data).to_csv(csv_file, index=True)
    print(f"CSV data saved to {csv_file}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExecution terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
