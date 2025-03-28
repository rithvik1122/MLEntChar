#!/usr/bin/env python3
"""
Utility to identify and list the most important measurements for entanglement characterization
in bi-partite ququart systems. This helps experimentalists to prioritize which measurements
to perform when resources are limited.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import sys

# Import functions from existing files if available
try:
    from entchar_old_mlebay import generate_mubs, construct_povms
except ImportError:
    # Define the functions here in case the import fails
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

def get_measurement_importance(num_measurements=None, output_dir='measurement_analysis'):
    """
    Calculate and rank measurement importance for entanglement characterization.
    Returns the top measurements ranked by information content.
    
    Args:
        num_measurements: Number of top measurements to return (None for all)
        output_dir: Directory to save output files
    
    Returns:
        List of tuples (measurement_index, importance_score)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate MUBs and POVMs
    print("Generating mutually unbiased bases (MUBs)...")
    mubs = generate_mubs()
    
    print("Constructing POVMs from MUBs...")
    povms = construct_povms(mubs)
    
    # Calculate information content (eigenvalue spread) for each POVM
    print("Ranking POVMs by information content...")
    povm_ranks = []
    for i, povm in enumerate(povms):
        try:
            eigenvals = np.linalg.eigvalsh(povm).real
            rank_value = np.max(eigenvals) - np.min(eigenvals)
            povm_ranks.append((i, rank_value))
        except Exception as e:
            print(f"Warning: Could not calculate eigenvalues for POVM {i}: {e}")
            povm_ranks.append((i, 0.0))
    
    # Sort by information content (descending)
    sorted_povm_ranks = sorted(povm_ranks, key=lambda x: x[1], reverse=True)
    
    # Track which MUB basis each measurement comes from
    measurement_to_basis = {}
    measurement_counter = 0
    
    for basis_idx, basis in enumerate(mubs):
        for vec_idx, _ in enumerate(basis):
            for basis2_idx, basis2 in enumerate(mubs):
                for vec2_idx, _ in enumerate(basis2):
                    measurement_to_basis[measurement_counter] = {
                        'subsystem_A_basis': basis_idx,
                        'subsystem_A_vector': vec_idx,
                        'subsystem_B_basis': basis2_idx,
                        'subsystem_B_vector': vec2_idx
                    }
                    measurement_counter += 1
    
    # Limit number of measurements if specified
    if num_measurements is not None:
        sorted_povm_ranks = sorted_povm_ranks[:num_measurements]
    
    # Create detailed measurement information
    measurement_info = []
    for rank_idx, (measurement_idx, score) in enumerate(sorted_povm_ranks):
        if measurement_idx in measurement_to_basis:
            basis_info = measurement_to_basis[measurement_idx]
            info = {
                'importance_rank': rank_idx + 1,
                'measurement_idx': measurement_idx,
                'information_score': score,
                'subsystem_A_basis': basis_info['subsystem_A_basis'],
                'subsystem_A_vector': basis_info['subsystem_A_vector'],
                'subsystem_B_basis': basis_info['subsystem_B_basis'],
                'subsystem_B_vector': basis_info['subsystem_B_vector']
            }
            measurement_info.append(info)
    
    # Create and save DataFrame
    df = pd.DataFrame(measurement_info)
    csv_path = os.path.join(output_dir, 'measurement_importance.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed measurement information to {csv_path}")
    
    # Save importance scores for each POVM
    importance_df = pd.DataFrame({
        'measurement_idx': [idx for idx, _ in povm_ranks],
        'importance_score': [score for _, score in povm_ranks]
    })
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    importance_csv = os.path.join(output_dir, 'importance_scores.csv')
    importance_df.to_csv(importance_csv, index=False)
    
    # Create a visualization of the measurement importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(min(100, len(sorted_povm_ranks))), 
            [score for _, score in sorted_povm_ranks[:100]], 
            alpha=0.7)
    plt.xlabel('Measurement Rank')
    plt.ylabel('Information Content Score')
    plt.title('Information Content of Top 100 Measurements')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'measurement_importance.png'), dpi=300)
    
    # Create a more detailed analysis of which basis combinations are most important
    basis_combination_scores = {}
    for info in measurement_info:
        key = (info['subsystem_A_basis'], info['subsystem_B_basis'])
        if key not in basis_combination_scores:
            basis_combination_scores[key] = []
        basis_combination_scores[key].append(info['information_score'])
    
    # Calculate average importance score for each basis combination
    basis_combo_avg = {key: np.mean(scores) for key, scores in basis_combination_scores.items()}
    basis_combo_count = {key: len(scores) for key, scores in basis_combination_scores.items()}
    
    # Create a heatmap of basis combination importance
    plt.figure(figsize=(10, 8))
    basis_matrix = np.zeros((5, 5))
    for (i, j), avg_score in basis_combo_avg.items():
        basis_matrix[i, j] = avg_score
    
    plt.imshow(basis_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Average Information Content')
    plt.xlabel('Subsystem B Basis Index')
    plt.ylabel('Subsystem A Basis Index')
    plt.title('Importance of Basis Combinations for Entanglement Characterization')
    plt.xticks(range(5), range(5))
    plt.yticks(range(5), range(5))
    
    # Add text annotations
    for i in range(5):
        for j in range(5):
            key = (i, j)
            count = basis_combo_count.get(key, 0)
            avg = basis_combo_avg.get(key, 0)
            plt.text(j, i, f'{avg:.2f}\n({count})', 
                     ha='center', va='center', 
                     color='white' if avg > np.mean(list(basis_combo_avg.values())) else 'black',
                     fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'basis_combination_importance.png'), dpi=300)
    
    # Create a summary text file with practical recommendations
    with open(os.path.join(output_dir, 'experimental_recommendations.txt'), 'w') as f:
        f.write("EXPERIMENTAL RECOMMENDATIONS FOR ENTANGLEMENT MEASUREMENTS\n")
        f.write("======================================================\n\n")
        f.write("This file contains practical recommendations for experimentalists\n")
        f.write("performing entanglement characterization of bi-partite ququart systems.\n\n")
        
        f.write("MOST IMPORTANT BASIS COMBINATIONS:\n")
        f.write("--------------------------------\n")
        # Sort basis combinations by importance
        sorted_basis_combos = sorted(basis_combo_avg.items(), key=lambda x: x[1], reverse=True)
        for idx, ((basis_a, basis_b), score) in enumerate(sorted_basis_combos[:5]):
            f.write(f"{idx+1}. Subsystem A Basis {basis_a} × Subsystem B Basis {basis_b}: ")
            f.write(f"Average information score {score:.4f}\n")
            f.write(f"   Number of top measurements: {basis_combo_count[(basis_a, basis_b)]}\n")
        
        f.write("\nTOP 20 INDIVIDUAL MEASUREMENTS:\n")
        f.write("----------------------------\n")
        for i, info in enumerate(measurement_info[:20]):
            f.write(f"{i+1}. Measurement {info['measurement_idx']}: ")
            f.write(f"Information score {info['information_score']:.4f}\n")
            f.write(f"   Subsystem A: Basis {info['subsystem_A_basis']}, Vector {info['subsystem_A_vector']}\n")
            f.write(f"   Subsystem B: Basis {info['subsystem_B_basis']}, Vector {info['subsystem_B_vector']}\n")
        
        f.write("\nPRACTICAL MEASUREMENT STRATEGY RECOMMENDATIONS:\n")
        f.write("------------------------------------------\n")
        f.write("1. If extremely limited resources (10-20 measurements):\n")
        f.write("   Focus exclusively on the top 10-20 measurements listed above.\n\n")
        
        f.write("2. For moderate resources (50-100 measurements):\n")
        f.write("   - Start with all measurements from the top 2-3 basis combinations\n")
        f.write("   - Add remaining top-ranked individual measurements\n\n")
        
        f.write("3. For comprehensive characterization (200+ measurements):\n")
        f.write("   Include all measurements from the top 5 basis combinations,\n")
        f.write("   then add remaining measurements in order of importance score.\n\n")
        
        f.write("NOTE ON INTERPRETATION:\n")
        f.write("----------------------\n")
        f.write("Basis indices refer to the following measurement bases:\n")
        f.write("0: Computational basis (standard Z measurements)\n")
        f.write("1: Hadamard-like basis (similar to X measurements)\n")
        f.write("2-4: Complex bases with various phase relationships\n\n")
        
        f.write("Vector indices (0-3) refer to specific measurement vectors within each basis.\n")
        f.write("See documentation for mathematical details of each basis vector.\n")
    
    print(f"Created experimental recommendations in {output_dir}/experimental_recommendations.txt")
    
    return sorted_povm_ranks

def main():
    parser = argparse.ArgumentParser(description='Analyze and rank measurement importance for entanglement characterization')
    parser.add_argument('--top', type=int, default=None, help='Number of top measurements to list (default: all)')
    parser.add_argument('--output-dir', type=str, default='measurement_analysis', help='Directory to save outputs')
    args = parser.parse_args()
    
    try:
        print(f"Analyzing importance of measurements for entanglement characterization...")
        ranked_measurements = get_measurement_importance(args.top, args.output_dir)
        print(f"Analysis complete! Results saved to {args.output_dir} directory.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
