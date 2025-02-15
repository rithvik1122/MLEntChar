import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ...existing code for training and evaluation routines...
# You can condition the VAE on measurements to reconstruct the density matrix.
