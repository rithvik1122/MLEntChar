# Quantum Entanglement Characterization Tools

This repository contains a suite of tools for quantum entanglement characterization using classical machine learning and traditional methods (MLE, Bayesian estimation).

## Overview

The project implements and compares several approaches for entanglement characterization from quantum measurement data:

- **Neural Networks**: MLP, CNN, Transformer, and Generative models
- **Traditional Methods**: Maximum Likelihood Estimation (MLE) and Bayesian Estimation
- **Utilities**: Data generation, model saving/loading, and visualization tools

## Setup and Requirements

### Dependencies

```bash
pip install numpy torch matplotlib scikit-learn tqdm pandas seaborn psutil
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running the Scripts

### 1. Full Training Pipeline

Run the main script to train and evaluate all models:

```bash
python entchar_old_mlebay.py --measurements 50 --sim-size 15000 --save-models
```

Arguments:
- `--measurements`: Number of quantum measurements to use (default: tests multiple values)
- `--sim-size`: Number of simulated quantum states (default: 15000)
- `--save-models`: Save trained models for future use
- `--batch-size`: Customize batch size for training (auto-selected by default)

If you experience memory issues with MLE/Bayesian methods, try:
```bash
python entchar_old_mlebay.py --measurements 50 --max-workers 4
```

For debugging issues with parallel processing:
```bash
python entchar_old_mlebay.py --disable-parallel
```

### 2. Loading Saved Models for Inference

After training models, you can use them for predictions:

```bash
python predict_with_saved_models.py --measurements 50 --model all
```

Arguments:
- `--measurements`: Number of measurements the model was trained on
- `--model`: Model type to use (mlp, cnn, transformer, generative, all)
- `--data-file`: Path to file with measurement data (optional)
- `--samples`: Number of samples to generate if no data file provided
- `--output`: Output filename for predictions

To compare performance across methods:
```bash
python predict_with_saved_models.py --compare --benchmark-samples 500
```

### 3. Traditional Methods Only (MLE & Bayesian)

To run only the MLE and Bayesian methods without neural networks:

```bash
python mle_bayesian_plot.py
```

This script generates plots comparing MLE and Bayesian estimation across different measurement counts.

### 4. Creating Plots

Generate publication-quality plots from results:

```bash
python plot_old_mlebay.py --results entanglement_results.csv
```

Arguments:
- `--results`: Path to results CSV file (default: entanglement_results.csv)
- `--predictions`: Path to predictions CSV file (optional)

### 5. Debugging Memory Issues

If you encounter memory errors or process crashes:

```bash
python debug_memory.py --workers 4 --array-size 1000 --tasks 10
```

Arguments:
- `--workers`: Number of workers for multiprocessing test
- `--array-size`: Size of test arrays (higher values test memory limits)
- `--tasks`: Number of parallel tasks to launch
- `--monitor`: Duration to monitor system memory (seconds)
- `--skip-mp`: Skip multiprocessing tests

## Workflow Examples

### Basic Research Workflow

1. **Train models and save them**:
   ```bash
   python entchar_old_mlebay.py --measurements 100 --sim-size 10000 --save-models
   ```

2. **Generate plots from results**:
   ```bash
   python plot_old_mlebay.py
   ```

3. **Use models for new data**:
   ```bash
   python predict_with_saved_models.py --measurements 100 --model all
   ```

### Comparing Methods

To benchmark all methods against each other across different measurement counts:

```bash
python predict_with_saved_models.py --compare --benchmark-samples 500
```

This will generate comparison plots in the `comparison_plots` directory, showing:
- MSE vs number of measurements
- Computation time vs number of measurements
- Efficiency frontier (error vs time)

## Troubleshooting

### Memory Errors

If you experience "BrokenProcessPool" errors:

1. Reduce the number of workers:
   ```bash
   python entchar_old_mlebay.py --max-workers 2
   ```

2. Run the diagnostics tool:
   ```bash
   python debug_memory.py
   ```

3. Disable parallelization for debugging:
   ```bash
   python entchar_old_mlebay.py --disable-parallel
   ```

### CUDA Out of Memory

If you experience GPU memory issues:

1. Reduce batch size:
   ```bash
   python entchar_old_mlebay.py --batch-size 32
   ```

2. For larger models, try using only one model type:
   ```bash
   python entchar_old_mlebay.py --model transformer
   ```

## File Descriptions

- `entchar_old_mlebay.py`: Main script for training and evaluating models
- `predict_with_saved_models.py`: Loads saved models and makes predictions
- `mle_bayesian_plot.py`: Standalone script for MLE and Bayesian methods
- `plot_old_mlebay.py`: Creates publication-quality plots from results
- `debug_memory.py`: Diagnostic tool for memory and multiprocessing issues

## Output Files

- `saved_models/`: Directory containing trained models and normalization parameters
- `entanglement_results.csv`: Results from model training and evaluation
- `plots/`: Directory containing generated plots
- `comparison_plots/`: Directory with method comparison plots
- `predictions.npz`: Prediction outputs from saved models
