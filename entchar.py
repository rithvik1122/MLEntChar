import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import csv
import time
import matplotlib.pyplot as plt
import torch.utils.data as data
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os  # Added to detect CPU cores
import concurrent.futures  # New import for parallel processing
from functools import partial  # Added import
import torch.backends.cudnn as cudnn  # New import
import torch.nn.functional as F  # Add this import at the top

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
        # Reshape input into 2D structure to better capture measurement correlations
        self.reshape_dim = int(np.sqrt(input_size)) + 1
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        # Add adaptive pooling to handle variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Reshape to 2D grid
        batch_size = x.size(0)
        x = F.pad(x, (0, self.reshape_dim**2 - x.size(1)))
        x = x.view(batch_size, 1, self.reshape_dim, self.reshape_dim)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, input_size, embed_dim=256):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(1, embed_dim)  # Embed each measurement individually
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Reshape to treat each measurement individually
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Global pooling
        return self.fc(x)

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
        rel_error = torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-6))
        l1_loss = torch.mean(torch.abs(pred - target))
        return mse + 0.01 * rel_error + 0.05 * l1_loss  # Reduced weights for stability

def train_model(model, X_train, Y_train, X_val, Y_val, num_measurements, batch_size=256, patience=50, max_epochs=2000):  # Reduced batch size
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
    
    # Adjust learning rate based on number of measurements
    base_lr = 0.001 / np.sqrt(num_measurements)  # Decrease learning rate for more measurements
    
    # Adjust batch size based on number of measurements
    adjusted_batch_size = min(batch_size, max(32, X_train.shape[0] // 20))
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=base_lr,
                                weight_decay=1e-5)
    
    # Further extend warmup proportional to measurements
    def lr_lambda(epoch):
        # Use a warmup period set to the number of measurements or at least 100 epochs
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
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    if torch.cuda.is_available():
                        with autocast(device_type='cuda', dtype=torch.float16):  # Specify dtype
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y.view(-1, 1))
                            loss = loss / accumulation_steps  # Scale loss for accumulation
                        
                        scaler.scale(loss).backward()
                        
                        if (i + 1) % accumulation_steps == 0:
                            scaler.unscale_(optimizer)
                            # Increased max_norm for gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)  # Fixed: Changed from set_to_none(True)
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.view(-1, 1))
                        loss.backward()
                        
                        if isinstance(model, Transformer) and (i + 1) % 4 != 0:
                            continue
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none(True))
                    
                    train_loss += loss.item()
                except KeyboardInterrupt:
                    print("\nTraining interrupted. Saving best model...")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    return model
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y.view(-1, 1)).item()
            
            val_loss /= len(val_loader)
            scheduler.step()
            
            pbar.set_description(f"Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving best model...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
    torch.cuda.empty_cache()
    return model

def predict_in_batches(model, X, batch_size=1024):
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
    num_measurements_list = [10, 20, 50, 100, 200, 400]  # More granular measurements
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
                  for method in ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian']}

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
        transformer_model = train_model(transformer_model, X_train_norm, Y_train, X_val_norm, Y_val, num_measurements, batch_size=batch_size)
        transformer_time = time.time() - start_time

        # Evaluate models with batches
        mlp_predictions = predict_in_batches(mlp_model, X_test_norm, batch_size)
        cnn_predictions = predict_in_batches(cnn_model, X_test_norm, batch_size)
        transformer_predictions = predict_in_batches(transformer_model, X_test_norm, batch_size)

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

        # Calculate both MSE and relative error
        for pred, method in [(mlp_predictions, 'MLP'), 
                           (cnn_predictions, 'CNN'),
                           (transformer_predictions, 'Transformer'),
                           (mle_predictions, 'MLE'),
                           (bayesian_predictions, 'Bayesian')]:
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
                "bayesian_pred": bayesian_predictions[i]
            })

    # Write results with rounded values for clarity
    with open('entanglement_results.csv', 'w', newline='') as file:
        fieldnames = ["num_measurements", "MLP_MSE", "CNN_MSE", "Transformer_MSE", "MLE_MSE", "Bayesian_MSE",
                      "MLP_Time", "CNN_Time", "Transformer_Time", "MLE_Time", "Bayesian_Time", "valid_states", "requested_states"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Round numeric values for better readability
            row["MLP_MSE"] = round(row["MLP_MSE"], 6)
            row["CNN_MSE"] = round(row["CNN_MSE"], 6)
            row["Transformer_MSE"] = round(row["Transformer_MSE"], 6)
            row["MLE_MSE"] = round(row["MLE_MSE"], 6)
            row["Bayesian_MSE"] = round(row["Bayesian_MSE"], 6)
            row["MLP_Time"] = round(row["MLP_Time"], 6)
            row["CNN_Time"] = round(row["CNN_Time"], 6)
            row["Transformer_Time"] = round(row["Transformer_Time"], 6)
            row["MLE_Time"] = round(row["MLE_Time"], 6)
            row["Bayesian_Time"] = round(row["Bayesian_Time"], 6)
            writer.writerow(row)

    with open('predictions_vs_true.csv', 'w', newline='') as file:
        fieldnames = ["num_measurements", "true_value", "mlp_pred", "cnn_pred", "transformer_pred", "mle_pred", "bayesian_pred"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions_data:
            row["true_value"] = round(row["true_value"], 6)
            row["mlp_pred"] = round(row["mlp_pred"], 6)
            row["cnn_pred"] = round(row["cnn_pred"], 6)
            row["transformer_pred"] = round(row["transformer_pred"], 6)
            row["mle_pred"] = round(row["mle_pred"], 6)
            row["bayesian_pred"] = round(row["bayesian_pred"], 6)
            writer.writerow(row)

    measurements = [result["num_measurements"] for result in results]
    mlp_mse = [result["MLP_MSE"] for result in results]
    cnn_mse = [result["CNN_MSE"] for result in results]
    transformer_mse = [result["Transformer_MSE"] for result in results]
    mle_mse = [result["MLE_MSE"] for result in results]
    bayesian_mse = [result["Bayesian_MSE"] for result in results]

    plt.figure(figsize=(10, 5))
    plt.plot(measurements, mlp_mse, label="MLP")
    plt.plot(measurements, cnn_mse, label="CNN")
    plt.plot(measurements, transformer_mse, label="Transformer")
    plt.plot(measurements, mle_mse, label="MLE")
    plt.plot(measurements, bayesian_mse, label="Bayesian")
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

    plt.figure(figsize=(10, 5))
    plt.plot(measurements, mlp_time, label="MLP")
    plt.plot(measurements, cnn_time, label="CNN")
    plt.plot(measurements, transformer_time, label="Transformer")
    plt.plot(measurements, mle_time, label="MLE")
    plt.plot(measurements, bayesian_time, label="Bayesian")
    plt.xlabel("Number of Measurements")
    plt.ylabel("Time (seconds)")
    plt.yscale('log')
    plt.legend()
    plt.title("Computation Time Comparison")
    plt.savefig('time_results.png', dpi=300)
    plt.show()

    # Plot both MSE and relative error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for method in ['MLP', 'CNN', 'Transformer', 'MLE', 'Bayesian']:
        ax1.plot(measurements, all_metrics[method]['mse'], label=method)
        ax2.plot(measurements, all_metrics[method]['rel_error'], label=method)
    
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