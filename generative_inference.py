import torch
import numpy as np
import argparse
from generative_density_matrix import DensityMatrixVAE  # Import the model architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, measurement_dim, latent_dim, density_dim):
    # Load the generative model and set it to evaluation mode
    model = DensityMatrixVAE(measurement_dim, latent_dim, density_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def generate_density_matrix(model, measurements):
    # Generate the density matrix from a list of measurements
    with torch.no_grad():
        x = torch.FloatTensor(measurements).unsqueeze(0).to(device)  # Shape: [1, measurement_dim]
        recon_dm, _, _ = model(x)
        recon_dm = recon_dm.cpu().numpy()[0]
        # Ensure Hermiticity and normalization
        recon_dm = 0.5 * (recon_dm + recon_dm.conj().T)
        recon_dm = recon_dm / np.trace(recon_dm)
    return recon_dm

def main():
    parser = argparse.ArgumentParser(description="Generate full density matrix from few measurements")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained generative model checkpoint")
    parser.add_argument("--measurements", type=float, nargs="+", required=True, help="List of measurements")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension for the generative model")
    parser.add_argument("--density_dim", type=int, default=16, help="Dimension of the density matrix (e.g., 16 for a 4x4 matrix)")
    args = parser.parse_args()

    measurement_dim = len(args.measurements)
    model = load_model(args.checkpoint, measurement_dim, args.latent_dim, args.density_dim)
    density_matrix = generate_density_matrix(model, args.measurements)
    
    print("Generated Density Matrix:")
    print(density_matrix)

if __name__ == "__main__":
    main()
