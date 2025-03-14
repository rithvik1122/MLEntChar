#!/usr/bin/env python3
"""
Simple script to plot MSE vs number of measurements for MLE and Bayesian methods.
Includes data generation, MLE and Bayesian methods directly without imports.
"""

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import concurrent.futures
from tqdm import tqdm
from functools import partial

# -------------------- DATA GENERATION CODE --------------------
# Directly copied from entchar_old_mlebay.py

def get_optimal_workers():
    """Get optimal number of workers for parallel processing"""
    cpu_cores = os.cpu_count()
    optimal_cpu_workers = max(2, int(cpu_cores * 0.9))
    print(f"Detected {cpu_cores} CPU cores, using {optimal_cpu_workers} workers")
    return optimal_cpu_workers

def generate_mubs():
    """Generate Mutually Unbiased Bases"""
    M0 = np.eye(4)
    M1 = np.array([[1, 1, 1, 1],
                   [1, 1, -1, -1],
                   [1, -1, -1, 1],
                   [1, -1, 1, -1]]) / 2
    M2 = np.array([[1, -1, -1j, -1j],
                   [1, -1, 1j, 1j],
                   [1, 1, 1j, -1j],
                   [1, 1, -1j, 1j]]) / 2
    M3 = np.array([[1, -1j, -1j, -1],
                   [1, -1j, 1j, 1],
                   [1, 1j, 1j, -1],
                   [1, 1j, -1j, 1]]) / 2
    M4 = np.array([[1, -1j, -1, -1j],
                   [1, -1j, 1, 1j],
                   [1, 1j, -1, 1j],
                   [1, 1j, 1, -1j]]) / 2
    return [M0, M1, M2, M3, M4]

def construct_povms(mubs):
    """Construct POVMs from MUBs"""
    # Collect local measurement vectors from all bases
    local_vectors = []
    for basis in mubs:
        # Each row of the basis is a measurement vector
        for vec in basis:
            local_vectors.append(vec)
    # Form bipartite POVMs via tensor products of all vector pairs
    bipartite_povms = []
    for vecA in local_vectors:
        for vecB in local_vectors:
            combined = np.kron(vecA, vecB)
            op = np.outer(combined, combined.conj())
            bipartite_povms.append(op)
    return bipartite_povms

def randomHaarState(dim, rank):
    """Generate a random mixed state using Haar measure"""
    A = np.random.normal(0, 1, (dim, dim)) + 1j * np.random.normal(0, 1, (dim, dim))
    q, r = np.linalg.qr(A, mode='complete')
    r = np.diag(np.divide(np.diagonal(r), np.abs(np.diagonal(r)))) @ np.eye(dim)
    rU = q @ r
    B = np.random.normal(0, 1, (dim, rank)) + 1j * np.random.normal(0, 1, (dim, rank))
    B = B @ B.T.conj()
    I = np.eye(dim)
    rho = (I + rU) @ B @ (I + rU.T.conj())
    return rho / np.trace(rho)

def randompure(dim, n):
    """Generate n random pure states in dimension dim"""
    rpure = np.random.normal(0, 1, [dim, n]) + 1j * np.random.normal(0, 1, [dim, n])
    rpure = rpure / np.linalg.norm(rpure, axis=0)
    rhon = []
    for i in range(n):
        rhon.append(np.outer(rpure[:, i], rpure[:, i].conj()))
    return rhon

def partial_transpose(rho, dims, subsystem=0):
    """Compute the partial transpose of a density matrix"""
    d1, d2 = dims
    rho_reshaped = rho.reshape(d1, d2, d1, d2)
    if subsystem == 0:
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
    else:
        rho_pt = rho_reshaped.transpose(0, 3, 2, 1)
    rho_pt = rho_pt.reshape(rho.shape)
    return rho_pt

def entanglement_negativity(rho, dims=[4,4]):
    """Calculate the entanglement negativity of a density matrix"""
    try:
        rho_pt = partial_transpose(rho, dims)
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        
        # Filter out small imaginary components
        eigenvalues = np.real(eigenvalues[np.abs(np.imag(eigenvalues)) < 1e-10])
        
        # Ensure eigenvalues are finite
        if not np.all(np.isfinite(eigenvalues)):
            return 0.0
            
        trace_norm = np.sum(np.abs(eigenvalues))
        return max(0, (trace_norm - 1) / 2)  # Ensure non-negative
        
    except np.linalg.LinAlgError:
        return 0.0  # Return 0 for problematic cases

def generate_data_with_mixture(sim_size, num_measurements):
    """Generate quantum data with mixed states for entanglement negativity predictions"""
    mubs = generate_mubs()
    povms = construct_povms(mubs)
    
    # Sort POVMs by information content (approximated by eigenvalue spread)
    povm_ranks = []
    for povm in povms:
        try:
            eigenvals = np.linalg.eigvalsh(povm).real
            povm_ranks.append(np.max(eigenvals) - np.min(eigenvals))
        except RuntimeError:
            povm_ranks.append(0.0)
    
    # Sort indices and reorganize POVMs
    sorted_indices = np.argsort(povm_ranks)[::-1]  # Descending order
    povms = [povms[i] for i in sorted_indices]
    
    # Use the most informative POVMs first
    used_povms = povms[:num_measurements]
    print(f"Using {len(used_povms)} most informative POVMs per state.")
    
    X = []
    Y = []
    
    # Use bins for better balance 
    entanglement_bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5)]
    samples_per_bin = sim_size // len(entanglement_bins)
    
    print(f"Target samples per bin: {samples_per_bin}")
    
    # Define parameters for different entanglement ranges
    bin_parameters = {
        (0.0, 0.3): [{'purity': 0.5}, {'purity': 0.7}, {'purity': 'random'}],
        (0.3, 0.6): [{'purity': 0.8}, {'purity': 0.85}, {'purity': 'random'}],
        (0.6, 0.9): [{'purity': 0.9}, {'purity': 0.93}, {'purity': 'random'}],
        (0.9, 1.2): [{'purity': 0.95}, {'purity': 0.98}, {'purity': 'random'}],
        (1.2, 1.5): [{'purity': 0.99}, {'purity': 0.995}, {'purity': 'random'}]
    }
    
    # Generated mixed states for each bin
    for bin_start, bin_end in entanglement_bins:
        bin_negativities = []
        attempts = 0
        param_idx = 0
        max_time_per_bin = 120  # 2 minutes timeout per bin
        start_time = time.time()
        
        pbar = tqdm(total=samples_per_bin, desc=f"Bin [{bin_start:.2f}, {bin_end:.2f}]")
        
        while len(bin_negativities) < samples_per_bin:
            # Check timeout
            if time.time() - start_time > max_time_per_bin:
                print(f"\nTimeout for bin [{bin_start:.2f}, {bin_end:.2f}]. "
                      f"Generated {len(bin_negativities)}/{samples_per_bin} states.")
                break
                
            # Get generation parameters
            if param_idx >= len(bin_parameters[(bin_start, bin_end)]):
                param_idx = 0
            params = bin_parameters[(bin_start, bin_end)][param_idx]
            
            if params['purity'] == 'random':
                if bin_start >= 0.9:
                    purity = np.random.uniform(0.93, 0.99)
                elif bin_start >= 0.6:
                    purity = np.random.uniform(0.85, 0.95)
                else:
                    purity = np.random.uniform(0.5, 0.9)
            else:
                purity = params['purity']
                
            # Create a batch of states
            batch_size = 10
            for _ in range(batch_size):
                # Generate random quantum state
                if np.random.rand() < 0.5:
                    # Pure state
                    pure_states = randompure(16, 1)
                    state = pure_states[0]
                else:
                    # Mixed state
                    state = randomHaarState(16, 2)
                
                # Mix with maximally mixed state to target entanglement range
                mixed_state = np.eye(16) / 16
                rho = purity * state + (1 - purity) * mixed_state
                
                # Calculate negativity
                neg = entanglement_negativity(rho)
                
                # If in target range, compute measurements and save
                if bin_start <= neg < bin_end:
                    # Calculate measurements for this state
                    measurements = []
                    for povm in used_povms:
                        prob = np.real(np.trace(rho @ povm))
                        # Add some measurement noise
                        n_shots = 200
                        noisy_count = np.random.binomial(n_shots, prob) / n_shots
                        measurements.append(noisy_count)
                    
                    X.append(measurements)
                    Y.append(neg)
                    bin_negativities.append(neg)
                    pbar.update(1)
                
                # Check if we have enough samples
                if len(bin_negativities) >= samples_per_bin:
                    break
                    
            # Update parameters periodically
            attempts += batch_size
            if attempts > 100:
                param_idx += 1
                attempts = 0
        
        pbar.close()
        if bin_negativities:
            print(f"  Bin [{bin_start:.2f}, {bin_end:.2f}]: Generated {len(bin_negativities)} states")
            print(f"  Average negativity: {np.mean(bin_negativities):.4f} ± {np.std(bin_negativities):.4f}")
    
    X = np.array(X)
    Y = np.array(Y)
    print(f"\nFinal dataset shape - X: {X.shape}, Y: {Y.shape}")
    
    return X, Y, povms

# -------------------- MLE AND BAYESIAN METHODS --------------------
# Directly copied from benchmark implementation

def _single_mle_estimation(measurements, povms):
    """
    Single MLE estimation - takes measurements from one quantum state and 
    returns the estimated entanglement negativity.
    """
    # Initialize with maximally mixed state
    rho = np.eye(16) / 16
    
    # Use only available measurements 
    povms = povms[:len(measurements)]
    max_iter = 9000 # Changed from 15000 to 9000
    
    # Simple MLE iteration
    for _ in range(max_iter):
        R = np.zeros((16, 16), dtype=np.complex128)
        
        for m, povm in zip(measurements, povms):
            prob = max(np.real(np.trace(rho @ povm)), 1e-10)
            R += (m / prob) * povm
        
        # Update density matrix
        new_rho = R @ rho @ R
        new_rho = 0.5 * (new_rho + new_rho.conj().T)  # Ensure Hermitian
        trace = np.trace(new_rho)
        if trace < 1e-10:
            break
        new_rho = new_rho / trace
        
        # Stricter convergence criteria for MLE
        if np.max(np.abs(new_rho - rho)) < 1e-8:
            break
            
        rho = new_rho
    
    # Calculate entanglement negativity
    return entanglement_negativity(rho)

def _single_bayesian_estimation(measurements, povms):
    """
    Single Bayesian estimation - takes measurements from one quantum state and
    returns the estimated entanglement negativity.
    
    This implementation differs from MLE by:
    1. Using a prior distribution (mixture of pure and mixed states)
    2. Applying Bayesian updates with regularization
    3. Using different convergence criteria
    """
    # Initialize with a stronger prior distribution
    # Use 3 different pure states to create a more complex prior
    pure_states = randompure(16, 3)
    mixed_part = np.eye(16, dtype=np.complex128) / 16
    
    # Create a more distinctive prior
    prior_weight = 0.5  # Stronger weight for prior knowledge (was 0.3)
    rho = (1 - prior_weight) * mixed_part
    for i, pure_state in enumerate(pure_states):
        # Add multiple pure states with different weights
        weight = 0.5 * (i + 1) / len(pure_states)
        rho += prior_weight * weight * pure_state
    
    rho = 0.5 * (rho + rho.conj().T)  # Ensure Hermitian
    rho = rho / np.trace(rho)  # Ensure trace 1
    
    # Use only available measurements
    povms = povms[:len(measurements)]
    max_iter = 6000  # Changed from 800 to 6000
    
    # Bayesian update loop with regularization
    alpha = 0.8  # More conservative learning rate (was 0.95)
    reg_param = 0.05  # Stronger regularization (was 0.01)
    
    # Keep track of the original prior for regularization
    prior_rho = rho.copy()
    
    for iter_count in range(max_iter):
        R = np.zeros((16, 16), dtype=np.complex128)
        
        for m, povm in zip(measurements, povms):
            prob = max(np.real(np.trace(rho @ povm)), 1e-10)
            R += (m / prob) * povm
        
        # Bayesian update with regularization toward prior
        new_rho = R @ rho @ R
        
        # Add regularization toward prior throughout the iterations
        # but with decreasing influence
        reg_factor = reg_param * np.exp(-iter_count / 100)
        new_rho = (1 - reg_factor) * new_rho + reg_factor * prior_rho
        
        new_rho = 0.5 * (new_rho + new_rho.conj().T)  # Ensure Hermitian
        trace = np.trace(new_rho)
        if trace < 1e-10:
            break
        new_rho = new_rho / trace
        
        # Gradually decrease learning rate for more stable convergence
        updated_rho = alpha * new_rho + (1 - alpha) * rho
        
        # Looser convergence check for Bayesian method
        # This will cause earlier stopping
        fidelity = np.abs(np.trace(rho @ updated_rho)) / np.sqrt(np.trace(rho @ rho) * np.trace(updated_rho @ updated_rho))
        if 1.0 - fidelity < 1e-3 and iter_count > 20:  # Much looser threshold
            break
        
        rho = updated_rho
    
    # Calculate entanglement negativity
    return entanglement_negativity(rho)

def _parallel_worker(method, measurements, povms):
    """Worker function for parallel processing"""
    return method(measurements, povms)

def mle_estimator(X, povms):
    """Parallel MLE estimator for multiple measurement sets"""
    optimal_workers = get_optimal_workers()
    print(f"Running MLE estimation with {optimal_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        worker_func = partial(_parallel_worker, _single_mle_estimation, povms=povms)
        results = list(executor.map(worker_func, X))
    return results

def bayesian_estimator(X, povms):
    """Parallel Bayesian estimator for multiple measurement sets"""
    optimal_workers = get_optimal_workers()
    print(f"Running Bayesian estimation with {optimal_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        worker_func = partial(_parallel_worker, _single_bayesian_estimation, povms=povms)
        results = list(executor.map(worker_func, X))
    return results

# -------------------- EVALUATION CODE --------------------

def calculate_mse_for_measurements(num_measurements, sim_size=1000, test_size=0.2):
    """
    Calculate MSE for a given number of measurements using both methods
    """
    print(f"\n{'='*50}")
    print(f"Generating data with {num_measurements} measurements...")
    
    # Generate data
    X, Y, povms = generate_data_with_mixture(sim_size, num_measurements)
    print(f"Generated data with shape X: {X.shape}, Y: {Y.shape}")
    
    # Split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )
    print(f"Test set size: {len(X_test)}")
    
    # Run MLE estimator
    print("Running MLE estimator...")
    start_time = time.time()
    mle_results = mle_estimator(X_test, povms[:num_measurements])
    mle_time = time.time() - start_time
    mle_mse = np.mean((np.array(mle_results) - Y_test)**2)
    print(f"MLE MSE: {mle_mse:.6f}, Time: {mle_time:.2f}s")
    
    # Run Bayesian estimator
    print("Running Bayesian estimator...")
    start_time = time.time()
    bayesian_results = bayesian_estimator(X_test, povms[:num_measurements])
    bayesian_time = time.time() - start_time
    bayesian_mse = np.mean((np.array(bayesian_results) - Y_test)**2)
    print(f"Bayesian MSE: {bayesian_mse:.6f}, Time: {bayesian_time:.2f}s")
    
    # Create scatter plots for visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test, mle_results, alpha=0.5)
    plt.plot([0, 1.5], [0, 1.5], 'r--')
    plt.title(f'MLE Predictions (MSE: {mle_mse:.6f})')
    plt.xlabel('True Negativity')
    plt.ylabel('Predicted Negativity')
    
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, bayesian_results, alpha=0.5)
    plt.plot([0, 1.5], [0, 1.5], 'r--')
    plt.title(f'Bayesian Predictions (MSE: {bayesian_mse:.6f})')
    plt.xlabel('True Negativity')
    plt.ylabel('Predicted Negativity')
    
    plt.tight_layout()
    os.makedirs('plots_mlebay', exist_ok=True)
    plt.savefig(f'plots_mlebay/scatter_plot_{num_measurements}_measurements.png')
    plt.close()
    
    return {
        'num_measurements': num_measurements,
        'mle_mse': mle_mse,
        'bayesian_mse': bayesian_mse,
        'mle_time': mle_time,
        'bayesian_time': bayesian_time
    }

def main():
    # List of measurement counts to test
    measurement_counts = [10, 20, 50, 100, 250, 400]
    
    # Update simulation size to match entchar_old_mlebay.py for fair comparison
    sim_size = 15000
    
    results = []
    
    for n in measurement_counts:
        result = calculate_mse_for_measurements(n, sim_size)
        results.append(result)
    
    # Create MSE vs measurements plot
    plt.figure(figsize=(10, 6))
    
    mle_mse = [r['mle_mse'] for r in results]
    bayesian_mse = [r['bayesian_mse'] for r in results]
    
    plt.plot(measurement_counts, mle_mse, 'o-', label='MLE')
    plt.plot(measurement_counts, bayesian_mse, 's-', label='Bayesian')
    
    plt.xlabel('Number of Measurements')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Number of Measurements')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    os.makedirs('plots_mlebay', exist_ok=True)
    plt.savefig('plots_mlebay/mse_vs_measurements.png')
    
    # Also plot execution time
    plt.figure(figsize=(10, 6))
    
    mle_time = [r['mle_time'] for r in results]
    bayesian_time = [r['bayesian_time'] for r in results]
    
    plt.plot(measurement_counts, mle_time, 'o-', label='MLE')
    plt.plot(measurement_counts, bayesian_time, 's-', label='Bayesian')
    
    plt.xlabel('Number of Measurements')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Number of Measurements')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('plots_mlebay/time_vs_measurements.png')
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Measurements':^15} | {'MLE MSE':^15} | {'Bayesian MSE':^15} | {'MLE Time (s)':^15} | {'Bayesian Time (s)':^15}")
    print("-"*80)
    for r in results:
        print(f"{r['num_measurements']:^15} | {r['mle_mse']:<15.6f} | {r['bayesian_mse']:<15.6f} | {r['mle_time']:<15.2f} | {r['bayesian_time']:<15.2f}")
    
    # Save results to file
    with open('plots_mlebay/mse_results.txt', 'w') as f:
        f.write("Measurements,MLE_MSE,Bayesian_MSE,MLE_Time,Bayesian_Time\n")
        for r in results:
            f.write(f"{r['num_measurements']},{r['mle_mse']:.6f},{r['bayesian_mse']:.6f},{r['mle_time']:.2f},{r['bayesian_time']:.2f}\n")
    
    print("\nResults saved to plots_mlebay directory")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
