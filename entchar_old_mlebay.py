# Standard library imports
import os
import csv
import time
import math
import concurrent.futures
from functools import partial

# Third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import torch.nn.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_gpu_availability():
    if torch.cuda.is_available():
        # Set per-process GPU memory fraction to 80%
        torch.cuda.set_per_process_memory_fraction(0.8, torch.cuda.current_device())
        gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        # Get GPU properties
        print(f"\nGPU Details:")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.zeros(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("GPU memory test passed successfully!")
        except Exception as e:
            print(f"Warning: GPU memory test failed: {str(e)}")
            print("Falling back to CPU...")
            global device
            device = torch.device("cpu")
    else:
        print("\nWarning: CUDA is not available. Using CPU.")
        print(f"PyTorch version: {torch.__version__}")

def get_optimal_workers():
    cpu_cores = os.cpu_count()
    # Use 90% of available CPU cores for CPU-bound operations
    optimal_cpu_workers = max(2, int(cpu_cores * 0.9))
    print(f"Detected {cpu_cores} CPU cores, using {optimal_cpu_workers} workers")
    return optimal_cpu_workers

def get_dataloader_workers():
    cpu_cores = os.cpu_count()
    if torch.cuda.is_available():
        # For GPU operations, use fewer workers to avoid CPU-GPU bottleneck
        return max(2, min(8, cpu_cores // 4))
    else:
        # For CPU operations, use more workers
        return max(2, int(cpu_cores * 0.8))

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_cuda_cores = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_cuda_cores += props.multi_processor_count * 64  # Approximate CUDA cores per SM
            print(f"GPU {i}: {props.name}")
            print(f"Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"CUDA Cores: ~{props.multi_processor_count * 64}")
        return gpu_count, total_cuda_cores
    return 0, 0

# REPLACED: Using data generation functions from mle_bayesian_plot.py
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

# REPLACED: Using data generation function from mle_bayesian_plot.py
def generate_data_with_mixture(sim_size, num_measurements, mixture_ratio=0.5, rank=2):
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

def generate_mixture_of_states(dim=16, mixture_ratio=0.5, rank=2):
    # Keep this function as a utility to maintain backward compatibility
    if np.random.rand() < mixture_ratio:
        pure_states = randompure(dim, 1)
        return pure_states[0]
    else:
        return randomHaarState(dim, rank)

def normalize_data(X_train, X_test):
    # Improved normalization strategy
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Avoid division by zero and handle small values better
    std = np.where(std < 1e-6, 1e-6, std)
    
    # Add scaling factor based on number of measurements
    scale_factor = np.sqrt(X_train.shape[1])  # Scale by sqrt of measurement count
    X_train_norm = (X_train - mean) / (std * scale_factor)
    X_test_norm = (X_test - mean) / (std * scale_factor)
    
    return X_train_norm, X_test_norm

def _single_mle_estimation(measurements, povms):
    """
    Single MLE estimation - takes measurements from one quantum state and 
    returns the estimated entanglement negativity.
    """
    # Initialize with maximally mixed state
    rho = np.eye(16) / 16
    
    # Use only available measurements 
    povms = povms[:len(measurements)]
    max_iter = 9000  # Fixed number of iterations
    
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

def _parallel_mle_worker(m, povms):
    return _single_mle_estimation(m, povms)

def parallel_mle_estimator(X, povms):
    """Parallel MLE estimator with improved error handling"""
    # Use more workers for CPU-bound MLE computation
    optimal_workers = get_optimal_workers()
    print(f"Running MLE estimation with {optimal_workers} workers")
    
    # Add more robust error handling
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Use a try/except block inside the context manager
            try:
                # Process data in smaller chunks to reduce memory pressure
                chunk_size = min(250, len(X))  # Process at most 250 items at a time
                results = []
                
                # Create chunks of data
                for i in range(0, len(X), chunk_size):
                    chunk = X[i:i + chunk_size]
                    print(f"Processing MLE chunk {i//chunk_size + 1}/{(len(X)-1)//chunk_size + 1} ({len(chunk)} items)")
                    
                    # Process chunk and handle errors
                    chunk_results = list(executor.map(partial(_parallel_mle_worker, povms=povms), chunk))
                    results.extend(chunk_results)
                    
                    # Free memory and check for interrupt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                
                return results
                
            except concurrent.futures.process.BrokenProcessPool:
                print("WARNING: Process pool broke. Falling back to sequential computation.")
                # Fall back to sequential processing if the process pool breaks
                results = []
                for i, x in enumerate(tqdm(X, desc="MLE Sequential")):
                    results.append(_single_mle_estimation(x, povms))
                return results
                
            except KeyboardInterrupt:
                print("\nMLE computation interrupted by user. Partial results will be returned.")
                # Return partial results on keyboard interrupt
                return results if results else [0.0] * len(X)
                
    except Exception as e:
        print(f"ERROR in MLE computation: {e}")
        print("Returning zeros as fallback")
        return [0.0] * len(X)

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
    prior_weight = 0.5  # Stronger weight for prior knowledge
    rho = (1 - prior_weight) * mixed_part
    for i, pure_state in enumerate(pure_states):
        # Add multiple pure states with different weights
        weight = 0.5 * (i + 1) / len(pure_states)
        rho += prior_weight * weight * pure_state
    
    rho = 0.5 * (rho + rho.conj().T)  # Ensure Hermitian
    rho = rho / np.trace(rho)  # Ensure trace 1
    
    # Use only available measurements
    povms = povms[:len(measurements)]
    max_iter = 6000  # Fixed number of iterations
    
    # Bayesian update loop with regularization
    alpha = 0.8  # More conservative learning rate
    reg_param = 0.05  # Stronger regularization
    
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

def _parallel_bayesian_worker(m, povms):
    return _single_bayesian_estimation(m, povms)

def parallel_bayesian_estimator(X, povms):
    """Parallel implementation of improved Bayesian estimator with better error handling"""
    optimal_workers = get_optimal_workers()
    print(f"Running Bayesian estimation with {optimal_workers} workers")
    
    # Add more robust error handling
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Use a try/except block inside the context manager
            try:
                # Process data in smaller chunks to reduce memory pressure
                chunk_size = min(250, len(X))  # Process at most 250 items at a time
                results = []
                
                # Create chunks of data
                for i in range(0, len(X), chunk_size):
                    chunk = X[i:i + chunk_size]
                    print(f"Processing Bayesian chunk {i//chunk_size + 1}/{(len(X)-1)//chunk_size + 1} ({len(chunk)} items)")
                    
                    # Process chunk and handle errors
                    chunk_results = list(executor.map(partial(_parallel_bayesian_worker, povms=povms), chunk))
                    results.extend(chunk_results)
                    
                    # Free memory and check for interrupt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                
                return results
                
            except concurrent.futures.process.BrokenProcessPool:
                print("WARNING: Process pool broke. Falling back to sequential computation.")
                # Fall back to sequential processing if the process pool breaks
                results = []
                for i, x in enumerate(tqdm(X, desc="Bayesian Sequential")):
                    results.append(_single_bayesian_estimation(x, povms))
                return results
                
            except KeyboardInterrupt:
                print("\nBayesian computation interrupted by user. Partial results will be returned.")
                # Return partial results on keyboard interrupt
                return results if results else [0.0] * len(X)
                
    except Exception as e:
        print(f"ERROR in Bayesian computation: {e}")
        print("Returning zeros as fallback")
        return [0.0] * len(X)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        # Important change: Scale network architecture with input size
        # Better scaling ensures the network can properly utilize more measurements
        hidden_size = max(1024, min(4096, input_size * 32))
        
        # Add input normalization and scaling based on input_size
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Important change: More sophisticated input attention mechanism
        # This allows the network to focus on the most informative measurements
        self.input_attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.Sigmoid()
        )
        
        # Architecture that scales properly with more measurements
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(max(0.05, 0.2 - 0.001 * input_size)),  # Less dropout with more measurements
            nn.BatchNorm1d(hidden_size),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(max(0.05, 0.15 - 0.001 * input_size)),
            nn.BatchNorm1d(hidden_size // 2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size // 4),
            
            nn.Linear(hidden_size // 4, 1),
            nn.ReLU()
        )
        
        # Improved weight initialization that scales with measurement count
        self._initialize_weights(input_size)
    
    def _initialize_weights(self, input_size):
        """Physics-aware initialization that improves with more measurements"""
        scale_factor = 1.0 / np.sqrt(input_size)  # Smaller initial weights with more measurements
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:  # Output layer
                    nn.init.zeros_(m.bias)
                    nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small weights for output
                else:
                    # Scale initialization based on input size for better convergence
                    bound = max(0.05, scale_factor)
                    nn.init.uniform_(m.weight, -bound, bound)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Apply measurement-aware attention that improves with more measurements
        x = self.input_norm(x)
        weights = self.input_attention(x)
        x = x * weights  # Weight measurements by learned importance
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        # Important change: Adjust grid dimensions based on input size
        self.grid_dim = int(np.ceil(np.sqrt(input_size)))
        
        # Scale feature extraction based on measurement count
        base_filters = min(128, max(32, input_size // 2))
        
        # Initial projection layer
        self.proj = nn.Conv2d(1, base_filters, kernel_size=1)
        
        # Enhanced feature extraction that scales with measurement count
        self.conv1a = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(base_filters)
        self.bn1b = nn.BatchNorm2d(base_filters)
        self.downsample1 = nn.Conv2d(base_filters, base_filters*2, kernel_size=1, stride=2)
        
        self.conv2a = nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1, stride=2)
        self.conv2b = nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(base_filters*2)
        self.bn2b = nn.BatchNorm2d(base_filters*2)
        self.downsample2 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=1, stride=2)
        
        self.conv3a = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1, stride=2)
        self.conv3b = nn.Conv2d(base_filters*4, base_filters*4, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(base_filters*4)
        self.bn3b = nn.BatchNorm2d(base_filters*4)
        
        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened feature size
        self.feature_size = base_filters * 4 * 4 * 4
        
        # Enhanced regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(max(0.05, 0.2 - 0.001 * input_size)),  # Less dropout with more measurements
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to 2D grid
        x = F.pad(x, (0, self.grid_dim**2 - x.size(1)))
        x = x.view(batch_size, 1, self.grid_dim, self.grid_dim)
        
        # Initial projection
        x = self.proj(x)
        
        # First block with residual
        identity = x
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = F.relu(x + identity)
        
        # Second block with residual
        identity = self.downsample1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)
        
        # Third block with residual
        identity = self.downsample2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = F.relu(x + identity)
        
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        return self.regressor(x)

class Transformer(nn.Module):
    def __init__(self, input_size, embed_dim=256):
        super(Transformer, self).__init__()
        # Scale embedding size with input size for better information processing
        self.embed_dim = min(512, max(128, input_size * 4))
        
        # Ensure divisibility with number of heads
        n_heads = min(8, max(4, self.embed_dim // 64))
        self.embed_dim = (self.embed_dim // n_heads) * n_heads
        
        # Store povms for prediction
        self.povms = None
        
        # Enhanced measurement-aware embedding
        self.embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(max(0.05, 0.15 - 0.0005 * input_size))
        )
        
        # More sophisticated positional encoding
        self.pos_encoding = self._create_positional_encoding(5000, self.embed_dim)
        
        # Measurement attention weights - learns importance of each measurement
        self.measurement_attn = nn.Parameter(torch.ones(1, 1000, 1))
        
        # Scale number of layers and feedforward size with input size
        n_layers = min(8, max(2, input_size // 32))
        ff_dim = self.embed_dim * 4
        
        # Fixed: Fixed the parameter order in TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=max(0.05, 0.2 - 0.001 * input_size),  # Fixed: Fixed the syntax error
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(self.embed_dim)
        )
        
        # Enhanced prediction head with measurement-dependent complexity
        hidden_dim = max(128, self.embed_dim // 2)
        
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(max(0.05, 0.1 - 0.0002 * input_size)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Ensure non-negative outputs
        )
        
        # Better initialization that scales with measurement count
        self._init_weights(input_size)

    def _init_weights(self, input_size):
        """Initialize weights with measurement-aware scaling"""
        scale = 1.0 / np.sqrt(input_size)
        
        # Initialize embedding layers
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=scale*2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize transformer differently from output layers
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'output_head' in name:
                    # Output layers need smaller weights
                    nn.init.xavier_uniform_(p, gain=0.01)
                else:
                    # Transformer benefits from Xavier
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
        # Initialize measurement attention with slight randomness
        with torch.no_grad():
            self.measurement_attn.data = torch.ones_like(self.measurement_attn) + \
                                        torch.randn_like(self.measurement_attn) * 0.01
    
    def _create_positional_encoding(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def clear_cache(self):
        """Clear any cached computation results"""
        pass  # No caching in this implementation

    def set_povms(self, povms):
        """Store POVMs for use in prediction"""
        self.povms = povms

    def forward(self, x):
        # Apply measurement-aware embedding
        B, N = x.shape
        x = x.unsqueeze(-1)  # [B, N, 1]
        
        # Apply learned measurement attention - scale with sigmoid to prevent instability
        attn_weights = F.sigmoid(self.measurement_attn[:, :N, :])
        x = x * attn_weights
        
        # Project to embedding dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :N, :]
        
        # Apply transformer directly - no gradient checkpointing to avoid memory issues
        x = self.transformer(x)
        
        # Use pooling weighted by measurement counts for better representation
        weights = torch.softmax(x.sum(dim=-1) / np.sqrt(self.embed_dim), dim=-1).unsqueeze(-1)
        x = torch.sum(x * weights, dim=1)
        
        # Generate final prediction with multi-stage output head
        return self.output_head(x)

class CustomDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CustomLoss(nn.Module):
    def __init__(self, measurement_count=10):
        super().__init__()
        # No need for explicit measurement count dependence
        
    def forward(self, pred, target):
        # Physics-based loss that naturally improves with more measurements
        mse = torch.mean((pred - target)**2)
        
        # Add L1 component for robustness and a relative error component
        rel_error = torch.mean(torch.abs(pred - target)/(target + 1e-6))
        l1_loss = torch.mean(torch.abs(pred - target))
        
        # Combined loss that naturally benefits from more measurements
        return mse + 0.01 * rel_error + 0.05 * l1_loss

def train_model(model, X_train, Y_train, X_val, Y_val, num_measurements, povms=None, batch_size=256, patience=50, max_epochs=2000):
    # Dynamic batch size adjustment to prevent OOM errors
    if isinstance(model, Transformer):
        # Use smaller batch sizes for Transformer models which are memory intensive
        batch_size = min(64, batch_size)
        print(f"Using reduced batch size for Transformer: {batch_size}")
    
    # Initialize minimum batch size to prevent excessive reduction
    min_batch_size = 16
    
    train_dataset = CustomDataset(X_train, Y_train)
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=get_dataloader_workers(),
        prefetch_factor=2  # Reduced for less memory pressure
    )
    
    val_dataset = CustomDataset(X_val, Y_val)
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=get_dataloader_workers()
    )
    
    # Add model.to(device) here to ensure model is on GPU
    model = model.to(device)

    # Important change: Set learning rate and regularization to scale with measurement count
    # This ensures we get proper improvement with more measurements
    if isinstance(model, Transformer):
        # Transformer-specific settings - BETTER SCALING
        lr_scale = 1.0 / np.power(num_measurements, 0.3)  # More aggressive scaling
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=min(2e-3, max(1e-5, 5e-4 * lr_scale)),
            weight_decay=min(0.05, max(0.001, 0.02 * lr_scale)),
            eps=1e-8
        )
        
        # Important change: Better scheduling for more measurements
        warmup_steps = len(train_loader) * min(10, max(1, 200 // num_measurements))
        
        # Use cosine decay after warmup for better convergence
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            # Cosine decay from max_lr to min_lr with measurement-dependent cycles
            cycle_steps = len(train_loader) * max(5, 100 // num_measurements)
            progress = (step - warmup_steps) % cycle_steps / cycle_steps
            return 0.1 + 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Other models - physics-based learning rate scheduling that improves with more measurements
        lr_scale = 1.0 / np.power(num_measurements, 0.2)  # Gentle scaling
        base_lr = min(0.002, max(0.0002, 0.001 * lr_scale))
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=min(0.01, max(0.0001, 0.001 * lr_scale)),
            eps=1e-4
        )
        
        # Important change: Better warmup/decay pattern for more measurements
        def lr_lambda(epoch):
            warmup = min(30, max(5, 100 // num_measurements))
            if epoch < warmup:
                return epoch / warmup
            # Natural cosine decay that depends on measurement count
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (max_epochs - warmup)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Important change: Scale patience with measurement count and model complexity
    if isinstance(model, Transformer):
        # Increased patience for Transformer
        patience = max(80, min(150, 30 + num_measurements // 4))
    else:
        patience = max(50, min(100, 20 + num_measurements // 5))
    
    # Better scaler settings
    scaler = GradScaler(
        init_scale=2**10,
        growth_factor=1.2,
        backoff_factor=0.7,
        growth_interval=200,
        enabled=torch.cuda.is_available()
    )
    
    # Use appropriate criterion based on model type
    criterion = CustomLoss(measurement_count=num_measurements)
    
    # MODIFIED: We'll still keep track of best validation loss for logging purposes,
    # but we won't save the associated model state
    best_val_loss = float('inf')
    best_epoch = -1  # Track the epoch with the best performance
    no_improve = 0
    min_delta = 1e-6  # Minimum improvement threshold
    
    # Add better early stopping with plateau detection
    val_losses = []
    plateau_window = 10  # Look at last 10 epochs for plateau detection
    plateau_threshold = 0.005  # Consider a plateau when improvement is less than this
    early_stop_counter = 0  # Count consecutive plateaus
    
    try:
        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            # Print learning rate every 50 epochs for monitoring
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, current LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Clear GPU cache at the beginning of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            # Training
            model.train()
            train_loss = 0
            batch_count = 0
            
            # Added gradient accumulation steps - adjust based on batch size
            accumulation_steps = max(1, 256 // batch_size)
            optimizer.zero_grad(set_to_none=True)
            
            for i, (batch_X, batch_y) in enumerate(train_loader):
                try:
                    # Move tensors to device and ensure they require gradients
                    batch_X = batch_X.to(device, non_blocking=True).requires_grad_(True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    if torch.cuda.is_available():
                        # Specially handle the Transformer to avoid CUDA errors
                        if isinstance(model, Transformer):
                            outputs = model(batch_X)
                            outputs = outputs.float()
                            batch_y = batch_y.float().view(-1, 1)
                            loss = criterion(outputs, batch_y)
                            loss = loss / accumulation_steps
                            
                            loss.backward()
                            
                            if (i + 1) % accumulation_steps == 0:
                                # Gradient clipping to prevent exploding gradients
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                        else:
                            # Use autocast for non-Transformer models
                            with autocast(device_type='cuda', dtype=torch.float16):
                                outputs = model(batch_X)
                                # Ensure outputs are floating point tensors
                                outputs = outputs.float()
                                batch_y = batch_y.float().view(-1, 1)
                                loss = criterion(outputs, batch_y)
                                loss = loss / accumulation_steps
                            
                            scaler.scale(loss).backward()
                            
                            if (i + 1) % accumulation_steps == 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad(set_to_none=True)
                    else:
                        # CPU path (unchanged)
                        outputs = model(batch_X)
                        outputs = outputs.float()
                        batch_y = batch_y.float().view(-1, 1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        
                        if (i + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none(True))
                    
                    train_loss += loss.item() * accumulation_steps
                    batch_count += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        # Reduce batch size dynamically when OOM occurs
                        if batch_size > min_batch_size:
                            old_batch_size = batch_size
                            batch_size = max(min_batch_size, batch_size // 2)
                            print(f"\nOOM error at batch {i}. Reducing batch size from {old_batch_size} to {batch_size}")
                            
                            # Recreate data loaders with new batch size
                            train_loader = data.DataLoader(
                                train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True if torch.cuda.is_available() else False,
                                num_workers=get_dataloader_workers(),
                                prefetch_factor=2
                            )
                            
                            val_loader = data.DataLoader(
                                val_dataset, 
                                batch_size=batch_size,
                                pin_memory=True if torch.cuda.is_available() else False,
                                num_workers=get_dataloader_workers()
                            )
                            
                            # Adjust accumulation steps based on new batch size
                            accumulation_steps = max(1, 256 // batch_size)
                            
                            # Make sure to clear any lingering gradients
                            optimizer.zero_grad(set_to_none=True)
                            
                            # Break to restart epoch with new batch size
                            break
                        else:
                            # Skip batch if already at minimum size
                            print(f"\nOOM error at batch {i}, skipping batch (already at min batch size)...")
                            if 'loss' in locals():
                                del loss
                            if 'outputs' in locals():
                                del outputs
                            if 'batch_X' in locals():
                                del batch_X
                            if 'batch_y' in locals():
                                del batch_y
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
            
            # Skip validation if we restarted due to OOM
            if 'break' in locals() and locals()['break']:
                # Reset the break flag and continue to next epoch
                del locals()['break']
                continue
                
            # Ensure we have a non-zero number of batches to avoid division by zero
            if batch_count == 0:
                batch_count = 1
            
            # Validation with dynamic batch sizing
            model.eval()
            val_loss = 0
            val_batch_count = 0
            current_val_batch_size = batch_size
            
            with torch.no_grad():
                for val_batch_idx, (batch_X, batch_y) in enumerate(val_loader):
                    try:
                        batch_X = batch_X.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y.view(-1, 1)).item()
                        val_batch_count += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            if current_val_batch_size > min_batch_size:
                                # Reduce validation batch size
                                current_val_batch_size = max(min_batch_size, current_val_batch_size // 2)
                                print(f"\nOOM during validation, reducing batch size to {current_val_batch_size}")
                                
                                # Recreate validation loader with smaller batch size
                                val_loader = data.DataLoader(
                                    val_dataset, 
                                    batch_size=current_val_batch_size,
                                    pin_memory=True if torch.cuda.is_available() else False,
                                    num_workers=get_dataloader_workers()
                                )
                                # Restart validation
                                val_batch_idx = 0
                                val_loss = 0
                                val_batch_count = 0
                                continue
                            else:
                                print("\nOOM during validation even at minimum batch size, skipping batch...")
                                continue
                        else:
                            raise e
            
            # Ensure we have a non-zero number of batches to avoid division by zero
            if val_batch_count == 0:
                val_batch_count = 1
                
            val_loss /= val_batch_count
            
            # Update scheduler using epoch or step based on model type
            if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                if hasattr(scheduler, 'last_epoch'):
                    scheduler.step()
                else:
                    # Handle step-based schedulers
                    for _ in range(batch_count):
                        scheduler.step()
            else:
                scheduler.step()
            
            # Update progress bar
            pbar.set_description(f"Train Loss: {train_loss/batch_count:.6f}, Val Loss: {val_loss:.6f}")
            
            # Store validation loss for plateau detection
            val_losses.append(val_loss)
            
            # Track best validation loss for logging purposes only (not for model saving)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                early_stop_counter = 0  # Reset plateau counter
            else:
                no_improve += 1
                
                # Check for plateau if we have enough history
                if len(val_losses) >= plateau_window:
                    # Calculate recent improvement rate
                    recent_losses = val_losses[-plateau_window:]
                    if max(recent_losses) - min(recent_losses) < plateau_threshold * best_val_loss:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                
                # Stop if either regular patience is exceeded or we have multiple consecutive plateaus
                if no_improve >= patience or early_stop_counter >= 3:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best validation loss was {best_val_loss:.6f} at epoch {best_epoch}")
                    print(f"Final model is from the current epoch {epoch}")
                    # MODIFIED: We don't restore the best model, we keep the current one
                    break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        print(f"Best validation loss was {best_val_loss:.6f} at epoch {best_epoch}")
        print(f"Final model is from the current epoch {epoch}")
    
    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Return the model in its current state, not the one with lowest validation loss
    return model

def predict_in_batches(model, X, povms=None, batch_size=128):  # Reduced default batch size
    """
    Make predictions in smaller batches to avoid memory issues
    """
    if isinstance(model, Transformer) and povms is not None:
        if hasattr(model, 'set_povms'):
            model.set_povms(povms)
    
    dataset = CustomDataset(X, np.zeros(len(X)))
    
    # For prediction, use a smaller batch size to avoid OOM
    adaptive_batch_size = min(batch_size, 64) if isinstance(model, Transformer) else batch_size
    
    # Fix: Change from num_workers(min(2, get_dataloader_workers())) to correct syntax
    loader = data.DataLoader(
        dataset, 
        batch_size=adaptive_batch_size,
        pin_memory=True,
        num_workers=min(2, get_dataloader_workers())  # Fixed syntax error
    )
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_X, _ in loader:
            try:
                batch_X = batch_X.to(device)
                batch_pred = model(batch_X)
                predictions.append(batch_pred.cpu().numpy())
                
                # Free memory immediately after each batch
                del batch_X
                del batch_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Try with a smaller batch
                    if adaptive_batch_size > 1:
                        adaptive_batch_size = max(1, adaptive_batch_size // 2)
                        print(f"OOM in prediction. Reducing batch size to {adaptive_batch_size}")
                        
                        # Recreate data loader with smaller batch size
                        loader = data.DataLoader(
                            dataset, 
                            batch_size=adaptive_batch_size,
                            pin_memory=True,
                            num_workers=1  # Minimum workers for smallest batches
                        )
                        continue
                    else:
                        print("Failed to predict even with batch size 1")
                        return np.zeros(len(X))  # Return zeros as fallback
                else:
                    raise e
    
    # Cleanup after all batches
    torch.cuda.empty_cache()
    
    # Check if we got predictions for all samples
    result = np.vstack(predictions).flatten()
    if len(result) < len(X):
        # Pad with zeros if needed
        result = np.pad(result, (0, len(X) - len(result)))
    elif len(result) > len(X):
        # Truncate if too many
        result = result[:len(X)]
    
    return result

def reconstruct_density_matrix(measurements):
    """Helper function to reconstruct density matrix from measurements"""
    # Move tensor to CPU if it's on GPU
    if torch.is_tensor(measurements):
        measurements = measurements.cpu().numpy()
    dm = np.zeros((16, 16), dtype=np.complex128)
    # Simple reconstruction - you can enhance this
    for i in range(len(measurements)):
        proj = np.eye(16) / 16  # Simple projection
        dm += measurements[i] * proj
    dm = dm / np.trace(dm)
    return dm

def reconstruct_density_matrix_mle(measurements, povms, max_iter=1000, tol=1e-6):
    """
    Maximum likelihood estimation of density matrix from measurements.
    
    Args:
        measurements: Array of measurement outcomes
        povms: List of POVM operators
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        Reconstructed density matrix
    """
    # Initialize with maximally mixed state
    rho = np.eye(16) / 16
    
    # Use only the provided number of measurements/POVMs
    measurements = measurements[:len(povms)]
    
    # Iterative MLE reconstruction
    for _ in range(max_iter):
        R = np.zeros((16, 16), dtype=np.complex128)
        
        # Compute R operator
        for m, povm in zip(measurements, povms):
            # Add numerical stability to probability calculation
            prob = max(np.real(np.trace(rho @ povm)), 1e-10)
            # Safe division with clipping
            factor = np.clip(m / prob, -1e6, 1e6)
            R += factor * povm
        
        # Update density matrix
        new_rho = R @ rho @ R
        new_rho = 0.5 * (new_rho + new_rho.conj().T)  # Ensure Hermiticity
        trace = max(np.real(np.trace(new_rho)), 1e-10)
        new_rho = new_rho / trace
        
        # Check convergence with safe norm calculation
        if np.max(np.abs(new_rho - rho)) < tol:
            break
            
        rho = new_rho
    
    return rho

def setup_signal_handlers():
    """Setup signal handlers to gracefully handle interruptions"""
    import signal
    import multiprocessing as mp
    
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C! Cleaning up...')
        # More careful cleanup of multiprocessing resources
        try:
            # Clean up active children processes
            active_children = mp.active_children()
            for child in active_children:
                try:
                    child.terminate()
                except:
                    pass  # Ignore errors in termination
            
            # Clean up remaining zombies (already terminated processes)
            mp.current_process()._cleanup()
            
            # Clean GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            print("Cleanup complete. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            sys.exit(1)
    
    # Register the signal handler for different signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    return signal_handler  # Return for possible further use

def main():
    # Enable cuDNN auto-tuner for faster runtime when using GPU
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # Add command-line arguments
    parser = argparse.ArgumentParser(description='Entanglement Characterization')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--measurements', type=int, default=None, help='Number of measurements to use')
    parser.add_argument('--sim-size', type=int, default=15000, help='Number of simulation samples')
    parser.add_argument('--save-models', action='store_true', help='Save trained models for future use')
    parser.add_argument('--max-workers', type=int, default=None, 
                       help='Maximum number of worker processes (default: auto)')
    parser.add_argument('--disable-parallel', action='store_true',
                       help='Disable parallel processing for MLE/Bayesian (for debugging)')
    args = parser.parse_args()

    # Initialize variables
    check_gpu_availability()
    
    # Override optimal workers if specified
    if args.max_workers is not None:
        global get_optimal_workers
        original_get_optimal_workers = get_optimal_workers
        def custom_get_optimal_workers():
            print(f"Using user-specified {args.max_workers} workers")
            return args.max_workers
        get_optimal_workers = custom_get_optimal_workers
    
    optimal_workers = get_optimal_workers()
    gpu_count, cuda_cores = get_gpu_info()
    print(f"Available GPU(s): {gpu_count}")
    if gpu_count > 0:
        print(f"Total CUDA cores: ~{cuda_cores}")

    # Set up improved signal handlers for graceful termination
    signal_handler = setup_signal_handlers()
    
    # Force sequential processing if requested
    if args.disable_parallel:
        print("WARNING: Parallel processing disabled. Using sequential computation.")
        
        # Override parallel estimator functions with sequential versions
        global parallel_mle_estimator, parallel_bayesian_estimator
        
        def sequential_mle_estimator(X, povms):
            print("Running sequential MLE estimation...")
            results = []
            for x in tqdm(X, desc="MLE"):
                results.append(_single_mle_estimation(x, povms))
            return results
        
        def sequential_bayesian_estimator(X, povms):
            print("Running sequential Bayesian estimation...")
            results = []
            for x in tqdm(X, desc="Bayesian"):
                results.append(_single_bayesian_estimation(x, povms))
            return results
        
        parallel_mle_estimator = sequential_mle_estimator
        parallel_bayesian_estimator = sequential_bayesian_estimator
    
    # Apply memory-saving settings
    if torch.cuda.is_available():
        # Use a more conservative memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        # Empty cache at start
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Start total runtime timer
    total_start_time = time.time()
    
    results = []
    predictions_data = []
    
    # Use command-line arguments if provided
    num_measurements_list = [10, 20, 50, 100, 250, 400]
    sim_size = args.sim_size
    
    # Dynamic batch size based on GPU memory
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem < 8:
            batch_size = 64
        elif gpu_mem < 12:
            batch_size = 128
        elif gpu_mem < 16:
            batch_size = 192
        else:
            batch_size = 256
        print(f"Auto-selected batch size {batch_size} based on {gpu_mem:.1f} GB GPU memory")
    else:
        batch_size = 64
    
    # Create tracking dictionaries
    all_metrics = {method: {'mse': [], 'rel_error': []} 
                  for method in ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian']}  # Removed 'Generative'

    # Create models directory if saving models
    if args.save_models:
        os.makedirs('saved_models', exist_ok=True)
        print("Will save best models to saved_models directory")

    # Main loop
    for num_measurements in num_measurements_list:
        try:
            print(f"\n{'='*50}")
            print(f"Running with {num_measurements} measurements...")
            
            # Clear memory before each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Generate data and print dimensions
            X, Y, povms = generate_data_with_mixture(sim_size, num_measurements)
            print(f"Full dataset shape - X: {X.shape}, Y: {Y.shape}")
            
            # Use stratified split if possible
            Y_bins = np.digitize(Y, bins=np.linspace(0, 1.5, 10))
            unique, counts = np.unique(Y_bins, return_counts=True)
            if counts.min() < 2:
                print("Warning: Some bins have less than 2 samples. Using non-stratified split.")
                stratify_arg = None
            else:
                stratify_arg = Y_bins
            
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42, stratify=stratify_arg
            )
            X_val, X_test, Y_val, Y_test = train_test_split(
                X_test, Y_test, test_size=0.5, random_state=42
            )
            
            print(f"Training set shape - X: {X_train.shape}, Y: {Y_train.shape}")
            print(f"Validation set shape - X: {X_val.shape}, Y: {Y_val.shape}")
            print(f"Test set shape - X: {X_test.shape}, Y: {Y_test.shape}")
            
            # Normalize all data
            X_train_norm, X_val_norm = normalize_data(X_train, X_val)
            _, X_test_norm = normalize_data(X_train, X_test)

            # Train models with batches
            try:
                start_time = time.time()
                mlp_model = MLP(X_train.shape[1]).to(device)
                mlp_model = train_model(mlp_model, X_train_norm, Y_train, X_val_norm, Y_val, 
                                       num_measurements, batch_size=batch_size)
                mlp_time = time.time() - start_time
                
                # Save MLP model if requested
                if args.save_models:
                    model_path = f'saved_models/mlp_model_{num_measurements}.pt'
                    torch.save(mlp_model.state_dict(), model_path)
                    print(f"MLP model saved to {model_path}")
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                start_time = time.time()
                cnn_model = CNN(X_train.shape[1]).to(device)
                cnn_model = train_model(cnn_model, X_train_norm, Y_train, X_val_norm, Y_val, 
                                        num_measurements, batch_size=batch_size)
                cnn_time = time.time() - start_time
                
                # Save CNN model if requested
                if args.save_models:
                    model_path = f'saved_models/cnn_model_{num_measurements}.pt'
                    torch.save(cnn_model.state_dict(), model_path)
                    print(f"CNN model saved to {model_path}")
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                start_time = time.time()
                transformer_model = Transformer(X_train.shape[1]).to(device)
                transformer_model = train_model(transformer_model, X_train_norm, Y_train, X_val_norm, Y_val, 
                                                num_measurements, povms=povms, batch_size=min(64, batch_size))
                transformer_time = time.time() - start_time

                # Save Transformer model if requested
                if args.save_models:
                    model_path = f'saved_models/transformer_model_{num_measurements}.pt'
                    torch.save(transformer_model.state_dict(), model_path)
                    print(f"Transformer model saved to {model_path}")
                
                # Evaluate models with batches
                mlp_predictions = predict_in_batches(mlp_model, X_test_norm, batch_size)
                cnn_predictions = predict_in_batches(cnn_model, X_test_norm, batch_size)
                transformer_predictions = predict_in_batches(transformer_model, X_test_norm, povms=povms, batch_size=batch_size)
            
                # MLE with significantly increased iterations (default is now 15000)
                start_time = time.time()
                mle_predictions = parallel_mle_estimator(X_test, povms)
                mle_time = time.time() - start_time
                mle_mse = np.mean((np.array(mle_predictions) - Y_test)**2)
    
                # Bayesian with significantly increased iterations (default is now 10000)
                start_time = time.time()
                bayesian_predictions = parallel_bayesian_estimator(X_test, povms)
                bayesian_time = time.time() - start_time
                bayesian_mse = np.mean((np.array(bayesian_predictions) - Y_test)**2)
    
                # Store results - remove Generative
                results.append({
                    "num_measurements": num_measurements,
                    "MLP_MSE": np.mean((mlp_predictions.flatten() - Y_test)**2),
                    "CNN_MSE": np.mean((cnn_predictions.flatten() - Y_test)**2),
                    "Transformer_MSE": np.mean((transformer_predictions.flatten() - Y_test)**2),
                    "MLE_MSE": mle_mse,
                    "Bayesian_MSE": bayesian_mse,
                    "MLP_Time": mlp_time,
                    "CNN_Time": cnn_time,
                    "Transformer_Time": transformer_time,
                    "MLE_Time": mle_time,
                    "Bayesian_Time": bayesian_time,
                    "valid_states": len(Y),
                    "requested_states": sim_size
                })
    
                # Save data as we go to prevent data loss
                with open(f'entanglement_results_{num_measurements}.csv', 'w', newline='') as file:
                    fieldnames = ["num_measurements", "MLP_MSE", "CNN_MSE", "Transformer_MSE", "MLE_MSE", 
                                  "Bayesian_MSE", "MLP_Time", "CNN_Time", "Transformer_Time", 
                                  "MLE_Time", "Bayesian_Time", "valid_states", "requested_states"]
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(results[-1])
            
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Moving to next measurement count.")
                continue
            except Exception as e:
                print(f"Error processing {num_measurements} measurements: {e}")
                traceback.print_exc()
                continue
        
        except KeyboardInterrupt:
            print("\nExecution interrupted by user. Saving results...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            continue

    # Combine results into final csv
    if results:
        try:
            # Write results with rounded values for clarity
            with open('entanglement_results.csv', 'w', newline='') as file:
                fieldnames = ["num_measurements", "MLP_MSE", "CNN_MSE", "Transformer_MSE", "MLE_MSE", 
                             "Bayesian_MSE", "MLP_Time", "CNN_Time", "Transformer_Time", 
                             "MLE_Time", "Bayesian_Time", "valid_states", "requested_states"]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    # Round numeric values for better readability
                    for key in row:
                        if isinstance(row[key], float):
                            row[key] = round(row[key], 6)
                    writer.writerow(row)
            
            print("Results saved to entanglement_results.csv")
        except Exception as e:
            print(f"Error saving final results: {e}")

    # Print total runtime
    total_time = time.time() - total_start_time
    print(f"\nTotal runtime: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    
    # Print GPU memory usage if available
    if torch.cuda.is_available():
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"Max GPU memory reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    
    print("To generate plots, use plot_old_mlebay.py")
    return 0

if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExecution terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)








