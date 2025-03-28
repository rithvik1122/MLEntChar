#!/usr/bin/env python3
"""
Script to load saved models and make predictions on new data.
This generates CSV files with true values and predictions for all methods,
which can then be visualized using plot_predictions.py.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import traceback
from pathlib import Path

# Import from ML classes from entchar_old_mlebay.py
from entchar_old_mlebay import (
    MLP, CNN, Transformer, device,
    entanglement_negativity, predict_in_batches, 
    generate_data_with_mixture, parallel_mle_estimator, parallel_bayesian_estimator
)

def load_model(model_type, num_measurements, model_dir="saved_models"):
    """
    Load a saved model of the specified type and number of measurements
    
    Args:
        model_type: One of 'mlp', 'cnn', or 'transformer'
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
    
    # For all models we can use predict_in_batches
    povms = norm_data['povms'] if 'povms' in norm_data else None
    predictions = predict_in_batches(model, X_norm, povms=povms, batch_size=batch_size)
    
    return predictions

def benchmark_models(measurements_list, test_samples=500, batch_size=64, model_dir="saved_models",
                     models_to_test=None, include_traditional=True, output_dir="prediction_outputs"):
    """
    Benchmark models across different measurement counts against MLE and Bayesian methods.
    Generates CSV files with predictions for further analysis.
    
    Args:
        measurements_list: List of measurement counts to test
        test_samples: Number of test samples to generate for each measurement count
        batch_size: Batch size for predictions
        model_dir: Directory where models are stored
        models_to_test: List of model types to test ('mlp', 'cnn', 'transformer')
        include_traditional: Whether to include MLE and Bayesian methods
        output_dir: Base directory to save all outputs
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "benchmark_data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Predictions and benchmark results will be saved to '{data_dir}' directory")
    
    # Default to all models if none specified
    if models_to_test is None:
        models_to_test = ['mlp', 'cnn', 'transformer']
    
    # Results dictionary to store performance metrics
    results = {
        'measurements': [],
        'method': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'time': []
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
        
        # Dictionary to store predictions for each method
        predictions_dict = {'true_values': y_true}
        
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
                
                # Store predictions for CSV export
                predictions_dict[model_type] = predictions
                
                # Calculate metrics for benchmark results
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                mse = mean_squared_error(y_true, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)
                
                # Store results
                results['measurements'].append(num_measurements)
                results['method'].append(model_type)
                results['mse'].append(mse)
                results['rmse'].append(rmse)
                results['mae'].append(mae)
                results['r2'].append(r2)
                results['time'].append(pred_time)
                
                print(f"{model_type.upper()} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}, Time: {pred_time:.2f}s")
                
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
                
                # Store predictions for CSV export
                predictions_dict['mle'] = np.array(mle_predictions)
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                mle_mse = mean_squared_error(y_true, mle_predictions)
                mle_rmse = np.sqrt(mle_mse)
                mle_mae = mean_absolute_error(y_true, mle_predictions)
                mle_r2 = r2_score(y_true, mle_predictions)
                
                # Store results
                results['measurements'].append(num_measurements)
                results['method'].append('mle')
                results['mse'].append(mle_mse)
                results['rmse'].append(mle_rmse)
                results['mae'].append(mle_mae)
                results['r2'].append(mle_r2)
                results['time'].append(mle_time)
                
                print(f"MLE - MSE: {mle_mse:.6f}, RMSE: {mle_rmse:.6f}, MAE: {mle_mae:.6f}, R²: {mle_r2:.6f}, Time: {mle_time:.2f}s")
            except Exception as e:
                print(f"Error evaluating MLE method: {e}")
            
            # Bayesian method
            try:
                print("\nEvaluating Bayesian method...")
                start_time = time.time()
                bayesian_predictions = parallel_bayesian_estimator(X, povms)
                bayesian_time = time.time() - start_time
                
                # Store predictions for CSV export
                predictions_dict['bayesian'] = np.array(bayesian_predictions)
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                bayesian_mse = mean_squared_error(y_true, bayesian_predictions)
                bayesian_rmse = np.sqrt(bayesian_mse)
                bayesian_mae = mean_absolute_error(y_true, bayesian_predictions)
                bayesian_r2 = r2_score(y_true, bayesian_predictions)
                
                # Store results
                results['measurements'].append(num_measurements)
                results['method'].append('bayesian')
                results['mse'].append(bayesian_mse)
                results['rmse'].append(bayesian_rmse)
                results['mae'].append(bayesian_mae)
                results['r2'].append(bayesian_r2)
                results['time'].append(bayesian_time)
                
                print(f"Bayesian - MSE: {bayesian_mse:.6f}, RMSE: {bayesian_rmse:.6f}, MAE: {bayesian_mae:.6f}, R²: {bayesian_r2:.6f}, Time: {bayesian_time:.2f}s")
            except Exception as e:
                print(f"Error evaluating Bayesian method: {e}")
        
        # Save predictions to CSV file
        if predictions_dict:
            pred_df = pd.DataFrame(predictions_dict)
            pred_file = os.path.join(data_dir, f'predictions_{num_measurements}.csv')
            pred_df.to_csv(pred_file, index=False)
            print(f"Saved predictions data to {pred_file}")
            
            # Also save as npz file for compatibility
            np.savez(os.path.join(data_dir, f'predictions_{num_measurements}.npz'), **predictions_dict)
            print(f"Saved predictions data to {os.path.join(data_dir, f'predictions_{num_measurements}.npz')}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    results_file = os.path.join(data_dir, 'benchmark_results.csv')
    df.to_csv(results_file, index=False)
    print(f"\nBenchmark results saved to {results_file}")
    print(f"Run 'python plot_predictions.py' to create visualization plots from the generated data.")
    
    return df

def main():
    """
    Simple entry point with common default parameters.
    For more customization, import and call benchmark_models directly.
    """
    # Set default parameters
    model_dir = "saved_models"
    output_dir = "prediction_outputs"
    measurements_list = [10, 20, 50, 100, 250, 400]
    models_to_test = ['mlp', 'cnn', 'transformer']
    include_traditional = True
    test_samples = 500
    batch_size = 64
    
    try:
        print("Starting model prediction benchmark...")
        print(f"Will test models for measurement counts: {measurements_list}")
        print(f"Models to test: {models_to_test}")
        print(f"Include traditional methods (MLE/Bayesian): {include_traditional}")
        
        df = benchmark_models(
            measurements_list=measurements_list,
            test_samples=test_samples,
            batch_size=batch_size,
            model_dir=model_dir,
            models_to_test=models_to_test,
            include_traditional=include_traditional,
            output_dir=output_dir
        )
        
        print("\nBenchmark completed successfully!")
        print("To create plots from the generated data, run:")
        print("python plot_predictions.py")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
