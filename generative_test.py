import numpy as np
import torch
import time
# Minimal necessary imports from entchar.py
from entchar import generate_data_with_mixture, train_generative_model, evaluate_generative_model

def main():
    # Use a small dataset for testing
    sim_size = 10000
    num_measurements = 400
    # Generate small synthetic dataset (X, negativity labels Y, density matrices, and POVMs)
    X, Y, density_matrices, povms = generate_data_with_mixture(sim_size, num_measurements)
    
    # Train the generative model on the synthetic measurements and density matrices
    start_time = time.time()
    # Pass Y as true_negativities argument for training
    gen_model = train_generative_model(X, density_matrices, Y, num_epochs=1000, lr=1e-3)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluate the generative model; using the same data for testing
    gen_negativities = evaluate_generative_model(gen_model, X, Y)
    mse_neg = np.mean((np.array(gen_negativities) - np.array(Y))**2)
    print(f"Entanglement Negativity MSE on test data: {mse_neg:.6f}")

if __name__ == "__main__":
    main()
