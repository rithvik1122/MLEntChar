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
import matplotlib.pyplot as plt
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

def generate_mubs():
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
    rpure = np.random.normal(0, 1, [dim, n]) + 1j * np.random.normal(0, 1, [dim, n])
    rpure = rpure / np.linalg.norm(rpure, axis=0)
    rhon = []
    for i in range(n):
        rhon.append(np.outer(rpure[:, i], rpure[:, i].conj()))
    return rhon

def generate_mixture_of_states(dim=16, mixture_ratio=0.5, rank=2):
    if np.random.rand() < mixture_ratio:
        pure_states = randompure(dim, 1)
        return pure_states[0]
    else:
        return randomHaarState(dim, rank)

def partial_transpose(rho, dims, subsystem=0):
    d1, d2 = dims
    rho_reshaped = rho.reshape(d1, d2, d1, d2)
    if subsystem == 0:
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
    else:
        rho_pt = rho_reshaped.transpose(0, 3, 2, 1)
    rho_pt = rho_pt.reshape(rho.shape)
    return rho_pt

def entanglement_negativity(rho, dims=[4,4]):
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

def generate_data_with_mixture(sim_size, num_measurements, mixture_ratio=0.5, rank=2):
    mubs = generate_mubs()
    povms = construct_povms(mubs)
    
    # Normalize POVMs
    for i in range(len(povms)):
        povms[i] = povms[i] / np.trace(povms[i])
    
    # Select the first 'num_measurements' POVMs directly, ensuring proper dimensionality.
    used_povms = povms[:num_measurements]
    print(f"Using {len(used_povms)} POVMs per state.")  # Debug: confirm measurement dimension
    
    X = []
    Y = []
    
    # More structured sampling of entanglement values
    entanglement_bins = np.linspace(0, 1.5, 6)
    samples_per_bin = sim_size // len(entanglement_bins)
    
    # Dynamically reduce noise for higher measurement counts
    base_noise_level = 0.001
    noise_scale = max(0.001, 0.001 * (100 / max(num_measurements, 10)))
    
    for bin_start, bin_end in zip(entanglement_bins[:-1], entanglement_bins[1:]):
        bin_negativities = []  # Collect negativities for this bin for debugging
        for _ in range(samples_per_bin):
            try:
                # Generate maximally entangled states
                if bin_end > 1.4:
                    psi = np.zeros(16, dtype=np.complex128)
                    for i in range(4):
                        psi[i*4 + i] = 0.5
                    rho = np.outer(psi, psi.conj())
                
                # Generate partially entangled states with better numerics
                elif bin_end > 0.7:
                    ent_param = np.clip(np.random.uniform(bin_start, bin_end), 0, 1)
                    psi = np.zeros(16, dtype=np.complex128)
                    if ent_param > 0.999:  # Handle edge case
                        ent_param = 0.999
                    psi[0] = np.sqrt(ent_param)
                    psi[5] = np.sqrt(1 - ent_param)
                    rho = np.outer(psi, psi.conj())
                
                # Generate mixed states with better conditioning
                else:
                    pure_state = np.random.normal(0, 1, 16) + 1j * np.random.normal(0, 1, 16)
                    pure_state = pure_state / np.linalg.norm(pure_state)
                    pure_rho = np.outer(pure_state, pure_state.conj())
                    mixed_part = np.eye(16) / 16
                    p = np.clip(np.random.uniform(bin_start, bin_end), 0, 1)
                    rho = p * pure_rho + (1-p) * mixed_part
                
                # Ensure rho is Hermitian and positive semi-definite
                rho = 0.5 * (rho + rho.conj().T)
                
                # Add small noise for numerical stability
                noise = np.eye(16) / 16
                rho = (1 - noise_scale) * rho + noise_scale * noise
                
                # Normalize with stable division
                trace = np.trace(rho)
                if abs(trace) < 1e-10:
                    continue
                rho = rho / trace
                
                # Check if matrix is valid before computing measurements
                if not np.all(np.isfinite(rho)):
                    continue
                    
                measurements = []
                # Use used_povms instead of povms[:num_measurements]
                for povm in used_povms:
                    val = max(np.real(np.trace(rho @ povm)), 1e-10)
                    measurements.append(val)
                
                # Only add if we get valid eigenvalues
                eigenvals = np.linalg.eigvalsh(rho)
                if np.all(eigenvals > -1e-10):  # Allow small negative values due to numerical error
                    X.append(measurements)
                    y_val = entanglement_negativity(rho)
                    Y.append(y_val)
                    bin_negativities.append(y_val)
                
            except (np.linalg.LinAlgError, RuntimeWarning) as e:
                continue  # Skip any problematic states
        if bin_negativities:
            print(f"Generated {len(bin_negativities)} states in bin [{bin_start:.2f}, {bin_end:.2f}] "
                  f"with avg negativity: {np.mean(bin_negativities):.4f}")
    
    if len(X) < sim_size:
        print(f"Warning: Could only generate {len(X)} valid states out of {sim_size} requested")
    
    X = np.array(X)
    Y = np.array(Y)
    print(f"Dataset dimensions - X: {X.shape}, Y: {Y.shape}")
    
    return X, Y, povms

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

def _single_mle_estimation(measurements, povms, max_iter=2000):
    # Extract code from the loop in mle_estimator that processes one measurement sample.
    negativities = []
    povms = povms[:len(measurements)]
    
    # Multiple random initializations
    best_overall_likelihood = -np.inf
    best_overall_rho = None
    
    for init_attempt in range(3):  # Try 3 different initial states
        # Initialize with different random states
        if init_attempt == 0:
            rho = np.eye(16) / 16  # Maximally mixed
        elif init_attempt == 1:
            # Random pure state
            psi = np.random.normal(0, 1, 16) + 1j * np.random.normal(0, 1, 16)
            psi = psi / np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())
        else:
            # Random mixed state
            rho = randomHaarState(16, 2)
        
        best_likelihood = -np.inf
        best_rho = None
        no_improve = 0
        step_size = 1.0
        
        # Measurement importance weighting
        total_weight = np.sum(measurements)
        measurement_weights = measurements / (total_weight + 1e-10)
        
        for iter in range(max_iter):
            R = np.zeros((16, 16), dtype=np.complex128)
            likelihood = 0
            
            for i, (m, povm, weight) in enumerate(zip(measurements, povms, measurement_weights)):
                prob = max(np.real(np.trace(rho @ povm)), 1e-10)
                R += weight * (m / prob) * povm
                likelihood += m * np.log(prob)
            
            # Modified update rule with momentum
            momentum = 0.8 + min(0.2, len(measurements) / 1000.0)
            if iter > 0:
                R = momentum * R_prev + (1 - momentum) * R
            R_prev = R.copy()
            
            # Regularized update
            new_rho = (1 - step_size) * rho + step_size * (R @ rho @ R)
            new_rho = 0.5 * (new_rho + new_rho.conj().T)
            new_rho /= np.trace(new_rho)
            
            # Add small amount of noise for stability
            noise = np.eye(16) / 16
            new_rho = 0.99 * new_rho + 0.01 * noise
            
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_rho = new_rho.copy()
                no_improve = 0
                step_size = min(1.0, step_size * 1.1)
            else:
                no_improve += 1
                step_size = max(0.1, step_size * 0.9)
            
            # Adaptive patience based on number of measurements
            patience = 20 + int(np.sqrt(len(measurements)))
            if no_improve > patience:
                break
            
            rho = new_rho
        
        if best_likelihood > best_overall_likelihood:
            best_overall_likelihood = best_likelihood
            best_overall_rho = best_rho
    
    return entanglement_negativity(best_overall_rho)

def _parallel_mle_worker(m, povms, max_iter):
    return _single_mle_estimation(m, povms, max_iter)

def parallel_mle_estimator(X, povms, max_iter=2000):
    # Use more workers for CPU-bound MLE computation
    optimal_workers = get_optimal_workers()
    print(f"Running MLE estimation with {optimal_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        results = list(executor.map(partial(_parallel_mle_worker, povms=povms, max_iter=max_iter), X))
    return results

def _single_bayesian_estimation(measurements, povms, max_iter=1000):
    """
    Simplified Bayesian estimation using maximum entropy principle
    """
    # Initialize with maximally mixed state
    rho = np.eye(16) / 16
    
    # Simple Bayesian update loop
    for _ in range(max_iter):
        # Compute likelihood updates
        R = np.zeros((16, 16), dtype=np.complex128)
        
        for m, povm in zip(measurements, povms[:len(measurements)]):
            prob = max(np.real(np.trace(rho @ povm)), 1e-10)
            R += (m / prob) * povm
        
        # Update state estimate
        new_rho = R @ rho @ R
        new_rho = 0.5 * (new_rho + new_rho.conj().T)  # Ensure Hermiticity
        trace = np.trace(new_rho)
        if trace < 1e-10:  # Numerical stability check
            break
        new_rho = new_rho / trace
        
        # Check convergence
        if np.max(np.abs(new_rho - rho)) < 1e-6:
            break
            
        rho = new_rho
    
    # Add small amount of noise for stability
    rho = 0.99 * rho + 0.01 * (np.eye(16) / 16)
    return entanglement_negativity(rho)

def _parallel_bayesian_worker(m, povms, max_iter):
    return _single_bayesian_estimation(m, povms, max_iter)

def parallel_bayesian_estimator(X, povms, max_iter=1000):
    """
    Parallel implementation of simplified Bayesian estimator
    """
    optimal_workers = get_optimal_workers()
    print(f"Running simplified Bayesian estimation with {optimal_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        results = list(executor.map(partial(_parallel_bayesian_worker, povms=povms, max_iter=max_iter), X))
    return results

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        # Scale network size with input dimension
        hidden_size = max(512, input_size * 8)  # Reduced from 16 to 8
        
        self.input_norm = nn.BatchNorm1d(input_size)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, 1),
            nn.ReLU()  # Added ReLU to ensure non-negative outputs
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                nn.init.uniform_(m.weight, -1/np.sqrt(fan_in), 1/np.sqrt(fan_in))

    def forward(self, x):
        x = self.input_norm(x)
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        # Calculate optimal reshaping dimensions based on input size
        self.grid_dim = int(np.ceil(np.sqrt(input_size)))
        
        # Enhanced feature extraction layers
        self.feature_extractor = nn.Sequential(
            # First conv block - local correlations
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block - measurement patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block - global features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))  # Adaptive pooling for variable input sizes
        )
        
        # Calculate flattened feature size
        self.feature_size = 128 * 2 * 2
        
        # Enhanced regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure non-negative output
        )
        
        # Initialize weights with physical consideration
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Initialize final layer carefully for regression
                if m.out_features == 1:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0.1)
                else:
                    nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to 2D grid while preserving measurement order
        x = F.pad(x, (0, self.grid_dim**2 - x.size(1)))
        x = x.view(batch_size, 1, self.grid_dim, self.grid_dim)
        
        # Extract features with spatial awareness
        features = self.feature_extractor(x)
        features = features.view(batch_size, -1)
        
        # Predict entanglement negativity
        return self.regressor(features)

class Transformer(nn.Module):
    def __init__(self, input_size, embed_dim=256):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoding = self._create_positional_encoding(5000, embed_dim)
        self.povms = None  # Add POVM storage
        self.reconstruction_cache = {}  # Add cache for reconstruction results
        
        # Simplified transformer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Use pre-normalization for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=4,
            norm=nn.LayerNorm(embed_dim)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, 1),
            nn.ReLU()  # Ensure non-negative outputs
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_positional_encoding(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def set_povms(self, povms):
        """Store POVMs for potential use in attention mechanisms"""
        self.povms = povms
        return self

    def clear_cache(self):
        """Clear the reconstruction cache"""
        if hasattr(self, 'reconstruction_cache'):
            self.reconstruction_cache.clear()

    def forward(self, x):
        # Ensure input requires gradient
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
            
        # Reshape and embed input
        B, N = x.shape
        x = x.unsqueeze(-1)  # [B, N, 1]
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :N, :]
        
        # Apply transformer with gradient checkpointing
        if self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.transformer),
                x,
                use_reentrant=False
            )
        else:
            x = self.transformer(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        
        # Final prediction
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
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Scale the loss components better
        mse = torch.mean((pred - target)**2)
        rel_error = torch.mean(torch.abs(pred / (target + 1e-6) - 1))
        l1_loss = torch.mean(torch.abs(pred - target))
        return mse + 0.01 * rel_error + 0.05 * l1_loss  # Reduced weights for stability

def train_model(model, X_train, Y_train, X_val, Y_val, num_measurements, povms=None, batch_size=256, patience=50, max_epochs=2000):  # Reduced batch size
    train_dataset = CustomDataset(X_train, Y_train)
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=get_dataloader_workers(),
        prefetch_factor=4  # Adjusted for faster data loading
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

    # Set base learning rate
    base_lr = 0.001 / np.sqrt(num_measurements)  # Default learning rate

    if isinstance(model, Transformer):
        # Transformer-specific settings
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Override base_lr for transformer
            weight_decay=0.01,
            eps=1e-8
        )
        
        batch_size = min(64, batch_size)
        
        warmup_steps = len(train_loader) * 5
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.95 ** (step - warmup_steps)
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Non-transformer models use base_lr
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=1e-5,
            eps=1e-4
        )
        
        def lr_lambda(epoch):
            warmup = max(num_measurements, 100)
            if epoch < warmup:
                return epoch / warmup
            return 0.995 ** (epoch - warmup)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Adjust patience based on measurement count
    patience = 50 + (num_measurements // 10)
    
    # Better scaler settings
    scaler = GradScaler(
        init_scale=2**10,
        growth_factor=1.2,
        backoff_factor=0.7,
        growth_interval=200,
        enabled=torch.cuda.is_available()
    )
    
    criterion = CustomLoss()
    best_val_loss = float('inf')
    best_model_state = None
    no_improve = 0
    min_delta = 1e-6  # Minimum improvement threshold
    
    try:
        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            # Print learning rate every 50 epochs for monitoring
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, current LR: {scheduler.get_last_lr()[0]:.6f}")
            # Training
            model.train()
            train_loss = 0
            
            # Added gradient accumulation steps
            accumulation_steps = 4
            optimizer.zero_grad(set_to_none=True)
            
            for i, (batch_X, batch_y) in enumerate(train_loader):
                try:
                    # Move tensors to device and ensure they require gradients
                    batch_X = batch_X.to(device, non_blocking=True).requires_grad_(True)  # Enable gradients
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    if torch.cuda.is_available():
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
                        outputs = model(batch_X)
                        outputs = outputs.float()
                        batch_y = batch_y.float().view(-1, 1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        
                        if isinstance(model, Transformer) and (i + 1) % 4 != 0:
                            continue
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                    train_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Try to recover by skipping this batch
                        print(f"\nOOM error at batch {i}, skipping...")
                        if 'loss' in locals():
                            del loss
                        if 'outputs' in locals():
                            del outputs
                        continue
                    else:
                        raise e
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                    try:
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y.view(-1, 1)).item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            print("\nOOM during validation, skipping batch...")
                            continue
                        raise e
            
            val_loss /= len(val_loader)
            scheduler.step()
            
            pbar.set_description(f"Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
            
            # Enhanced early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    model.load_state_dict(best_model_state)
                    break

            if isinstance(model, Transformer):
                # Clear cache periodically without directly accessing reconstruction_cache
                if epoch % 10 == 0:
                    model.clear_cache()
                    torch.cuda.empty_cache()
                
                # Use gradient accumulation for transformer
                effective_batch = batch_size // 4  # Process smaller chunks
                optimizer.zero_grad(set_to_none=True)
                
                for i, (batch_X, batch_y) in enumerate(train_loader):
                    sub_batches = batch_X.chunk(4)  # Split into smaller chunks
                    sub_labels = batch_y.chunk(4)
                    
                    for sub_X, sub_y in zip(sub_batches, sub_labels):
                        sub_X, sub_y = sub_X.to(device), sub_y.to(device)
                        
                        with autocast(device_type='cuda'):  # Added device_type
                            outputs = model(sub_X)
                            loss = criterion(outputs, sub_y.view(-1, 1)) / 4
                        
                        scaler.scale(loss).backward()
                    
                    if (i + 1) % 4 == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving best model...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
    torch.cuda.empty_cache()
    return model

def predict_in_batches(model, X, povms=None, batch_size=1024):
    if isinstance(model, Transformer) and povms is not None:
        model.set_povms(povms)
    
    dataset = CustomDataset(X, np.zeros(len(X)))
    loader = data.DataLoader(
        dataset, 
        batch_size=batch_size,
        pin_memory=True,
        num_workers=get_dataloader_workers()
    )
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            batch_pred = model(batch_X)
            predictions.append(batch_pred.cpu().numpy())
    
    torch.cuda.empty_cache()  # Clear GPU memory
    return np.vstack(predictions).flatten()

class DensityMatrixVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dm_dim, measurement_count=100):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dm_dim = dm_dim
        
        # Adjust hidden size based on measurement count
        multiplier = 8 if measurement_count < 200 else 10
        hidden_size = max(512, input_dim * multiplier)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, dm_dim * dm_dim),
            nn.ReLU()
        )
    
    def encode(self, x):
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        dm_flat = self.decoder(z)
        dm = dm_flat.view(-1, self.dm_dim, self.dm_dim)
        dm = 0.5 * (dm + dm.transpose(-1, -2))
        trace = dm.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        eps = 1e-8
        trace = torch.clamp(trace, min=eps)
        dm = dm / trace.unsqueeze(-1)
        return dm
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_dm = self.decode(z)
        return recon_dm, mu, logvar

def train_generative_model(X, povms, y_true=None, num_epochs=1000, lr=1e-3):
    """Train the generative model using MLE reconstruction approach"""
    # Ensure num_epochs is an integer
    num_epochs = int(num_epochs) if isinstance(num_epochs, (list, tuple)) else num_epochs
    
    input_dim = X.shape[1]
    latent_dim = 64
    dm_dim = 16
    
    model = DensityMatrixVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        dm_dim=dm_dim,
        measurement_count=input_dim
    ).to(device)
    
    # Use adaptive learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Convert data to PyTorch tensors
    X = torch.FloatTensor(X).to(device)
    if y_true is not None:
        y_true = torch.FloatTensor(y_true).to(device)
    
    # Enhanced early stopping configuration
    best_loss = float('inf')
    patience = max(20, X.shape[1] // 5)  # Scale patience with measurement count
    no_improve = 0
    min_delta = 1e-5  # Minimum improvement threshold
    
    # Explicitly move model to GPU and use GPU memory settings
    model = model.to(device)
    if torch.cuda.is_available():
        # Enable memory optimization
        torch.backends.cudnn.benchmark = True
        # Use mixed precision training
        scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_size = min(128, len(X))
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)  # More efficient GPU memory handling
            
            if torch.cuda.is_available():
                with autocast(device_type='cuda'):  # Specify device_type='cuda'
                    recon_dm, mu, logvar = model(batch_X)
                    recon_loss = torch.mean(torch.norm(recon_dm, p='fro', dim=(1,2)))
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    beta = min(1.0, epoch / 100)
                    loss = recon_loss + beta * kl_loss
                
                # Use gradient scaling for mixed precision training
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Existing CPU training code
                recon_dm, mu, logvar = model(batch_X)
                recon_loss = torch.mean(torch.norm(recon_dm, p='fro', dim=(1,2)))
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                beta = min(1.0, epoch / 100)
                loss = recon_loss + beta * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()

        # ...rest of existing code...

        avg_loss = total_loss / (len(X) / batch_size)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')
    
    return model

def evaluate_generative_model(model, X, y_true=None):
    model.eval()
    X = torch.FloatTensor(X).to(device)
    
    negativities = []
    with torch.no_grad():
        for i in range(len(X)):
            x = X[i:i+1]
            recon_dm, _, _ = model(x)
            
            dm = recon_dm[0].cpu().numpy()
            dm = 0.5 * (dm + dm.conj().T)
            dm = dm / np.trace(dm)
            
            neg = entanglement_negativity(dm)
            negativities.append(neg)
    
    negativities = np.array(negativities)
    
    if y_true is not None:
        mse = np.mean((negativities - y_true) ** 2)
        print(f'Test MSE: {mse:.6f}')
    
    return negativities

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

def main():
    # Enable cuDNN auto-tuner for faster runtime when using GPU
    if torch.cuda.is_available():
        cudnn.benchmark = True

    check_gpu_availability()
    optimal_workers = get_optimal_workers()
    gpu_count, cuda_cores = get_gpu_info()
    print(f"Available GPU(s): {gpu_count}")
    if gpu_count > 0:
        print(f"Total CUDA cores: ~{cuda_cores}")

    # Start total runtime timer
    total_start_time = time.time()
    
    results = []
    predictions_data = []  # New list to store prediction details
    num_measurements_list = [10, 20, 50, 100, 250, 400]  # Reduced measurement counts
    sim_size = 15000  # Slightly reduced dataset size for stability

    # Increase batch size for larger measurement counts
    batch_size = 256  # Reduced batch size
    
    # Dynamically adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        # For 20GB GPU, increase batch size; otherwise default to 256
        batch_size = 512 if gpu_mem >= 20 else 256
    else:
        batch_size = 256

    # Add metrics tracking
    all_metrics = {method: {'mse': [], 'rel_error': []} 
                  for method in ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian', 'Generative']}

    for num_measurements in num_measurements_list:
        print(f"\n{'='*50}")
        print(f"Running with {num_measurements} measurements...")
        
        # Clear memory before each run
        torch.cuda.empty_cache()
        
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
        start_time = time.time()
        mlp_model = MLP(X_train.shape[1]).to(device)
        mlp_model = train_model(mlp_model, X_train_norm, Y_train, X_val_norm, Y_val, num_measurements, batch_size=batch_size)
        mlp_time = time.time() - start_time
        
        start_time = time.time()
        cnn_model = CNN(X_train.shape[1]).to(device)
        cnn_model = train_model(cnn_model, X_train_norm, Y_train, X_val_norm, Y_val, num_measurements, batch_size=batch_size)
        cnn_time = time.time() - start_time
        
        start_time = time.time()
        transformer_model = Transformer(X_train.shape[1]).to(device)
        transformer_model = train_model(transformer_model, X_train_norm, Y_train, X_val_norm, Y_val, num_measurements, povms=povms, batch_size=batch_size)
        transformer_time = time.time() - start_time

        # Evaluate models with batches
        mlp_predictions = predict_in_batches(mlp_model, X_test_norm, batch_size)
        cnn_predictions = predict_in_batches(cnn_model, X_test_norm, batch_size)
        transformer_predictions = predict_in_batches(transformer_model, X_test_norm, povms=povms, batch_size=batch_size)

        # MLE
        start_time = time.time()
        mle_predictions = parallel_mle_estimator(X_test, povms)
        mle_time = time.time() - start_time
        mle_mse = np.mean((np.array(mle_predictions) - Y_test)**2)

        # Bayesian
        start_time = time.time()
        bayesian_predictions = parallel_bayesian_estimator(X_test, povms)
        bayesian_time = time.time() - start_time
        bayesian_mse = np.mean((np.array(bayesian_predictions) - Y_test)**2)

        # Add Generative method evaluation
        start_time = time.time()
        gen_model = train_generative_model(
            X=X_train_norm,
            povms=povms,
            y_true=Y_train,
            num_epochs=1000,  # Explicitly pass as integer
            lr=1e-3
        )
        gen_predictions = evaluate_generative_model(gen_model, X_test_norm, Y_test)
        gen_time = time.time() - start_time
        gen_mse = np.mean((np.array(gen_predictions) - Y_test)**2)

        results.append({
            "num_measurements": num_measurements,
            "MLP_MSE": np.mean((mlp_predictions.flatten() - Y_test)**2),
            "CNN_MSE": np.mean((cnn_predictions.flatten() - Y_test)**2),
            "Transformer_MSE": np.mean((transformer_predictions.flatten() - Y_test)**2),
            "MLE_MSE": mle_mse,
            "Bayesian_MSE": bayesian_mse,
            "Generative_MSE": gen_mse,
            "MLP_Time": mlp_time,
            "CNN_Time": cnn_time,
            "Transformer_Time": transformer_time,
            "MLE_Time": mle_time,
            "Bayesian_Time": bayesian_time,
            "Generative_Time": gen_time,
            "valid_states": len(Y),
            "requested_states": sim_size
        })

        # Calculate both MSE and relative error
        for pred, method in [(mlp_predictions, 'MLP'), 
                           (cnn_predictions, 'CNN'),
                           (transformer_predictions, 'Transformer'),
                           (mle_predictions, 'MLE'),
                           (bayesian_predictions, 'Bayesian'),
                           (gen_predictions, 'Generative')]:
            pred = np.array(pred)  # Convert list to numpy array if needed
            mse = np.mean((pred.flatten() - Y_test)**2)
            rel_error = np.mean(np.abs(pred.flatten() - Y_test)/(Y_test + 1e-8))
            all_metrics[method]['mse'].append(mse)
            all_metrics[method]['rel_error'].append(rel_error)

        # After making predictions, store detailed results
        test_size = len(Y_test)
        for i in range(test_size):
            predictions_data.append({
                "num_measurements": num_measurements,
                "true_value": Y_test[i],
                "mlp_pred": mlp_predictions[i],
                "cnn_pred": cnn_predictions[i],
                "transformer_pred": transformer_predictions[i],
                "mle_pred": mle_predictions[i],
                "bayesian_pred": bayesian_predictions[i],
                "generative_pred": gen_predictions[i]
            })

    # Write results with rounded values for clarity
    with open('entanglement_results.csv', 'w', newline='') as file:
        fieldnames = ["num_measurements", "MLP_MSE", "CNN_MSE", "Transformer_MSE", "MLE_MSE", "Bayesian_MSE", "Generative_MSE",
                      "MLP_Time", "CNN_Time", "Transformer_Time", "MLE_Time", "Bayesian_Time", "Generative_Time", "valid_states", "requested_states"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Round numeric values for better readability
            row["MLP_MSE"] = round(row["MLP_MSE"], 6)
            row["CNN_MSE"] = round(row["CNN_MSE"], 6)
            row["Transformer_MSE"] = round(row["Transformer_MSE"], 6)
            row["MLE_MSE"] = round(row["MLE_MSE"], 6)
            row["Bayesian_MSE"] = round(row["Bayesian_MSE"], 6)
            row["Generative_MSE"] = round(row["Generative_MSE"], 6)
            row["MLP_Time"] = round(row["MLP_Time"], 6)
            row["CNN_Time"] = round(row["CNN_Time"], 6)
            row["Transformer_Time"] = round(row["Transformer_Time"], 6)
            row["MLE_Time"] = round(row["MLE_Time"], 6)
            row["Bayesian_Time"] = round(row["Bayesian_Time"], 6)
            row["Generative_Time"] = round(row["Generative_Time"], 6)
            writer.writerow(row)

    with open('predictions_vs_true.csv', 'w', newline='') as file:
        fieldnames = ["num_measurements", "true_value", "mlp_pred", "cnn_pred", "transformer_pred", "mle_pred", "bayesian_pred", "generative_pred"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions_data:
            row["true_value"] = round(row["true_value"], 6)
            row["mlp_pred"] = round(row["mlp_pred"], 6)
            row["cnn_pred"] = round(row["cnn_pred"], 6)
            row["transformer_pred"] = round(row["transformer_pred"], 6)
            row["mle_pred"] = round(row["mle_pred"], 6)
            row["bayesian_pred"] = round(row["bayesian_pred"], 6)
            row["generative_pred"] = round(row["generative_pred"], 6)
            writer.writerow(row)

    measurements = [result["num_measurements"] for result in results]
    mlp_mse = [result["MLP_MSE"] for result in results]
    cnn_mse = [result["CNN_MSE"] for result in results]
    transformer_mse = [result["Transformer_MSE"] for result in results]
    mle_mse = [result["MLE_MSE"] for result in results]
    bayesian_mse = [result["Bayesian_MSE"] for result in results]
    generative_mse = [result["Generative_MSE"] for result in results]

    plt.figure(figsize=(10, 5))
    plt.plot(measurements, mlp_mse, 'o-', label="MLP")
    plt.plot(measurements, cnn_mse, 'o-', label="CNN")
    plt.plot(measurements, transformer_mse, 'o-', label="Transformer")
    plt.plot(measurements, mle_mse, 'o-', label="MLE")
    plt.plot(measurements, bayesian_mse, 'o-', label="Bayesian")
    plt.plot(measurements, generative_mse, 'o-', label="Generative")
    plt.plot(measurements, cnn_time, 'o-', label="CNN")
    plt.xlabel("Number of Measurements")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.legend()
    plt.title("Entanglement Negativity Estimation MSE")
    plt.savefig('mse_results.png', dpi=300)
    plt.show()

    mlp_time = [result["MLP_Time"] for result in results]
    cnn_time = [result["CNN_Time"] for result in results]
    transformer_time = [result["Transformer_Time"] for result in results]
    mle_time = [result["MLE_Time"] for result in results]
    bayesian_time = [result["Bayesian_Time"] for result in results]
    generative_time = [result["Generative_Time"] for result in results]

    plt.figure(figsize=(10, 5))
    plt.plot(measurements, mlp_time, 'o-', label="MLP")
    plt.plot(measurements, cnn_time, 'o-', label="CNN")
    plt.plot(measurements, transformer_time, 'o-', label="Transformer")
    plt.plot(measurements, mle_time, 'o-', label="MLE")
    plt.plot(measurements, bayesian_time, 'o-', label="Bayesian")
    plt.plot(measurements, generative_time, 'o-', label="Generative")
    plt.xlabel("Number of Measurements")
    plt.ylabel("Time (seconds)")
    plt.yscale('log')
    plt.legend()
    plt.title("Computation Time Comparison")
    plt.savefig('time_results.png', dpi=300)
    plt.show()

    # Plot both MSE and relative error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for method in ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian', 'Generative']:
        ax1.plot(measurements, all_metrics[method]['mse'], 'o-', label=method)
        ax2.plot(measurements, all_metrics[method]['rel_error'], 'o-', label=method)
    
    ax1.set_xlabel("Number of Measurements")
    ax1.set_ylabel("MSE")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title("MSE vs Measurements")
    
    ax2.set_xlabel("Number of Measurements")
    ax2.set_ylabel("Relative Error")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.set_title("Relative Error vs Measurements")
    
    plt.tight_layout()
    plt.savefig('error_metrics.png', dpi=300)
    plt.show()
    
    # Print total runtime
    total_time = time.time() - total_start_time
    print(f"\nTotal runtime: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    
    # Print GPU memory usage
    if torch.cuda.is_available():
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"Max GPU memory reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

if __name__ == "__main__":
    main()
