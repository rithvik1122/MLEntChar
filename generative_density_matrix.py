import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add device definition and import entanglement_negativity from entchar
import torch
from entchar import entanglement_negativity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: A conditional VAE for density matrix reconstruction

class DensityMatrixVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dm_dim, measurement_count=100):  # Added measurement_count parameter
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dm_dim = dm_dim
        
        # Adjust hidden size based on measurement count
        multiplier = 8 if measurement_count < 200 else 10  # Changed
        hidden_size = max(512, input_dim * multiplier)
        
        # Encoder network: outputs both mu and logvar
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim * 2)  # mu and logvar concatenated
        )
        
        # Decoder network: reconstructs flattened density matrix
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, dm_dim * dm_dim),
            nn.ReLU()  # Ensure non-negative outputs
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
        # Ensure Hermiticity and add epsilon in normalization
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

    def generate_density_matrix(self, measurements):
        """Helper method to generate density matrix from measurements"""
        with torch.no_grad():
            x = torch.FloatTensor(measurements).to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            recon_dm, _, _ = self.forward(x)
            dm = recon_dm[0].cpu().numpy()
            # Ensure Hermiticity and normalization
            dm = 0.5 * (dm + dm.conj().T)
            dm = dm / np.trace(dm)
            return dm

# ...existing code for training and evaluation routines...
# You can condition the VAE on measurements to reconstruct the density matrix.

def train_generative_model(X, povms, y_true, num_epochs=1000, lr=1e-3, batch_size=64, latent_dim=64, dm_dim=16):
    """Train the generative model on measurement data"""
    input_dim = X.shape[1]
    measurement_count = X.shape[1]  # Number of measurements per sample
    
    model = DensityMatrixVAE(input_dim, latent_dim, dm_dim, measurement_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert data to PyTorch tensors
    X = torch.FloatTensor(X).to(device)
    if y_true is not None:
        y_true = torch.FloatTensor(y_true).to(device)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        # Process data in batches
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            
            optimizer.zero_grad()
            recon_dm, mu, logvar = model(batch_X)
            
            # Reconstruction loss (using Frobenius norm)
            recon_loss = torch.mean(torch.norm(recon_dm, p='fro', dim=(1,2)))
            
            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss with KL weight scheduling
            beta = min(1.0, epoch / 100)  # Gradually increase KL weight
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(X):.6f}')
    
    return model

def evaluate_generative_model(model, X, y_true=None):
    """Evaluate the generative model by reconstructing density matrices and computing negativities"""
    model.eval()
    X = torch.FloatTensor(X).to(device)
    
    negativities = []
    with torch.no_grad():
        for i in range(len(X)):
            # Generate density matrix from measurements
            x = X[i:i+1]  # Keep batch dimension
            recon_dm, _, _ = model(x)
            
            # Convert to numpy and ensure proper properties
            dm = recon_dm[0].cpu().numpy()
            dm = 0.5 * (dm + dm.conj().T)  # Ensure Hermiticity
            dm = dm / np.trace(dm)  # Normalize
            
            # Compute entanglement negativity
            neg = entanglement_negativity(dm)
            negativities.append(neg)
    
    negativities = np.array(negativities)
    
    if y_true is not None:
        mse = np.mean((negativities - y_true) ** 2)
        print(f'Test MSE: {mse:.6f}')
    
    return negativities

# ...existing code...
