#!/usr/bin/env python3
"""
Extract important measurements from trained neural networks.
This script analyzes which measurements were most important for entanglement
estimation by extracting attention weights from MLP and Transformer models.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
import sys

# Import from ML classes from entchar_old_mlebay.py
try:
    from entchar_old_mlebay import MLP, CNN, Transformer, device
except ImportError:
    print("Warning: Could not import neural network models from entchar_old_mlebay.py")
    print("Please ensure this file is in the same directory as entchar_old_mlebay.py")
    sys.exit(1)

def load_model(model_path, num_measurements):
    """
    Load a trained model from a .pt file
    
    Args:
        model_path: Path to the saved model file
        num_measurements: Number of measurements the model was trained on
        
    Returns:
        Loaded model and its type
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Determine model type from filename
    model_type = None
    if 'mlp' in model_path.lower():
        model_type = 'mlp'
        model = MLP(num_measurements)
    elif 'transformer' in model_path.lower():
        model_type = 'transformer'
        model = Transformer(num_measurements)
    elif 'cnn' in model_path.lower():
        model_type = 'cnn'
        model = CNN(num_measurements)
    else:
        raise ValueError(f"Could not determine model type from filename: {model_path}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, model_type

def extract_mlp_attention(model, num_measurements):
    """
    Extract attention weights from an MLP model
    
    Args:
        model: Trained MLP model
        num_measurements: Number of measurements
        
    Returns:
        Numpy array of attention weights
    """
    # Get attention weights from the input_attention module
    with torch.no_grad():
        # Create dummy input
        dummy_input = torch.ones(1, num_measurements).to(device)
        
        # Get attention weights
        attention_weights = model.input_attention(dummy_input)
        
        # Convert to numpy array
        weights = attention_weights.cpu().numpy()[0]
    
    return weights

def extract_transformer_attention(model, num_measurements):
    """
    Extract attention weights from a Transformer model
    
    Args:
        model: Trained Transformer model
        num_measurements: Number of measurements
        
    Returns:
        Numpy array of attention weights and attention maps
    """
    with torch.no_grad():
        # Create dummy input
        dummy_input = torch.ones(1, num_measurements).to(device)
        
        # Get measurement attention weights (from the weighted attention mechanism)
        if hasattr(model, 'measurement_attention'):
            meas_weights = model.measurement_attention(dummy_input).cpu().numpy()[0]
        else:
            meas_weights = torch.sigmoid(model.measurement_weights).cpu().numpy()[0]
        
        # Now get self-attention weights from the transformer encoder
        # We'll need to run a forward pass to capture attention
        attention_maps = []
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            # Attention weights are in output[1]
            if isinstance(output, tuple) and len(output) > 1:
                attention_maps.append(output[1].cpu().detach())
        
        # Register hooks for each attention layer
        hooks = []
        for layer in model.encoder.layers:
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
        
        # Forward pass
        _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process attention maps (average across heads and layers)
        avg_attention = None
        if attention_maps:
            # Shape of attention_maps[i]: [batch_size, num_heads, seq_len, seq_len]
            all_attentions = torch.cat([attn.mean(dim=1) for attn in attention_maps], dim=0)  # [num_layers, seq_len, seq_len]
            avg_attention = all_attentions.mean(dim=0).numpy()  # [seq_len, seq_len]
    
    return meas_weights, avg_attention

def analyze_measurements(model, model_type, num_measurements, output_dir='measurement_analysis'):
    """
    Analyze which measurements are most important based on model attention
    
    Args:
        model: Trained neural network model
        model_type: Type of model ('mlp' or 'transformer')
        num_measurements: Number of measurements
        output_dir: Directory to save output files
        
    Returns:
        DataFrame with measurement importance information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'mlp':
        # Extract attention weights
        attention_weights = extract_mlp_attention(model, num_measurements)
        
        # Create DataFrame
        df = pd.DataFrame({
            'measurement_idx': np.arange(num_measurements),
            'attention_weight': attention_weights,
            'normalized_importance': attention_weights / np.sum(attention_weights) * 100
        })
        
        # Sort by importance
        df = df.sort_values('attention_weight', ascending=False).reset_index(drop=True)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df['attention_weight'], alpha=0.7)
        plt.xlabel('Measurement Index (Sorted by Importance)')
        plt.ylabel('Attention Weight')
        plt.title('MLP Attention Weights for Measurements')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mlp_measurement_importance_{num_measurements}.png'), dpi=300)
        plt.close()
        
        # Create top measurements summary
        top_n = min(20, len(df))
        top_df = df.head(top_n).copy()
        top_df['cumulative_importance'] = top_df['normalized_importance'].cumsum()
        
        # Save to CSV
        df.to_csv(os.path.join(output_dir, f'mlp_measurement_importance_{num_measurements}.csv'), index=False)
        
        # Create a summary text file
        with open(os.path.join(output_dir, f'mlp_measurement_summary_{num_measurements}.txt'), 'w') as f:
            f.write(f"MLP MEASUREMENT IMPORTANCE ANALYSIS ({num_measurements} measurements)\n")
            f.write("======================================================\n\n")
            f.write(f"Top {top_n} most important measurements:\n\n")
            
            for i, row in top_df.iterrows():
                f.write(f"{i+1}. Measurement {int(row['measurement_idx'])}: ")
                f.write(f"Weight {row['attention_weight']:.4f} ")
                f.write(f"(Importance: {row['normalized_importance']:.2f}%, ")
                f.write(f"Cumulative: {row['cumulative_importance']:.2f}%)\n")
            
            f.write(f"\nThe top {top_n} measurements account for {top_df['cumulative_importance'].iloc[-1]:.2f}% of total importance.\n")
        
        print(f"MLP analysis complete. Results saved to {output_dir}")
        return df
        
    elif model_type == 'transformer':
        # Extract attention weights
        meas_weights, attn_maps = extract_transformer_attention(model, num_measurements)
        
        # Create DataFrame for measurement weights
        df = pd.DataFrame({
            'measurement_idx': np.arange(num_measurements),
            'attention_weight': meas_weights,
            'normalized_importance': meas_weights / np.sum(meas_weights) * 100
        })
        
        # Sort by importance
        df = df.sort_values('attention_weight', ascending=False).reset_index(drop=True)
        
        # Create visualization for measurement weights
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df['attention_weight'], alpha=0.7)
        plt.xlabel('Measurement Index (Sorted by Importance)')
        plt.ylabel('Attention Weight')
        plt.title('Transformer Measurement Attention Weights')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'transformer_measurement_importance_{num_measurements}.png'), dpi=300)
        plt.close()
        
        # Create visualization for self-attention maps if available
        if attn_maps is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_maps, cmap='viridis', 
                        xticklabels=20 if num_measurements > 100 else 5,
                        yticklabels=20 if num_measurements > 100 else 5)
            plt.xlabel('Measurement Index (Target)')
            plt.ylabel('Measurement Index (Source)')
            plt.title('Transformer Self-Attention Map')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'transformer_attention_map_{num_measurements}.png'), dpi=300)
            plt.close()
            
            # Save attention map as CSV
            attn_df = pd.DataFrame(attn_maps)
            attn_df.to_csv(os.path.join(output_dir, f'transformer_attention_map_{num_measurements}.csv'), index=False)
            
            # Also calculate the overall importance of each measurement in the attention map
            # by summing attention received by each measurement
            attn_importance = attn_maps.sum(axis=0)
            attn_importance_df = pd.DataFrame({
                'measurement_idx': np.arange(num_measurements),
                'attn_importance': attn_importance,
                'normalized_attn_importance': attn_importance / np.sum(attn_importance) * 100
            }).sort_values('attn_importance', ascending=False).reset_index(drop=True)
            
            # Save to CSV
            attn_importance_df.to_csv(os.path.join(output_dir, f'transformer_attention_importance_{num_measurements}.csv'), index=False)
            
            # Create a visualization for attention importance
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(attn_importance_df)), attn_importance_df['attn_importance'], alpha=0.7)
            plt.xlabel('Measurement Index (Sorted by Attention Received)')
            plt.ylabel('Total Attention Received')
            plt.title('Transformer Self-Attention Importance')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'transformer_attention_importance_{num_measurements}.png'), dpi=300)
            plt.close()
            
            # Combined importance (blend of direct weight and attention received)
            combined_df = df.copy()
            combined_df['attn_importance'] = [attn_importance[int(idx)] for idx in combined_df['measurement_idx']]
            combined_df['combined_score'] = combined_df['normalized_importance'] * combined_df['attn_importance']
            combined_df['combined_score'] = combined_df['combined_score'] / combined_df['combined_score'].sum() * 100
            combined_df = combined_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
            
            # Save combined importance
            combined_df.to_csv(os.path.join(output_dir, f'transformer_combined_importance_{num_measurements}.csv'), index=False)
            
            # Create visualization for combined importance
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(combined_df)), combined_df['combined_score'], alpha=0.7)
            plt.xlabel('Measurement Index (Sorted by Combined Importance)')
            plt.ylabel('Combined Importance Score (%)')
            plt.title('Transformer Combined Measurement Importance')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'transformer_combined_importance_{num_measurements}.png'), dpi=300)
            plt.close()
            
            # Create a summary text file with insights from both attention mechanisms
            with open(os.path.join(output_dir, f'transformer_measurement_summary_{num_measurements}.txt'), 'w') as f:
                f.write(f"TRANSFORMER MEASUREMENT IMPORTANCE ANALYSIS ({num_measurements} measurements)\n")
                f.write("=====================================================================\n\n")
                
                # Direct attention weights
                top_n = min(20, len(df))
                top_df = df.head(top_n).copy()
                top_df['cumulative_importance'] = top_df['normalized_importance'].cumsum()
                
                f.write("PART 1: DIRECT MEASUREMENT ATTENTION WEIGHTS\n")
                f.write("-------------------------------------------\n")
                f.write(f"Top {top_n} most important measurements based on direct attention weights:\n\n")
                
                for i, row in top_df.iterrows():
                    f.write(f"{i+1}. Measurement {int(row['measurement_idx'])}: ")
                    f.write(f"Weight {row['attention_weight']:.4f} ")
                    f.write(f"(Importance: {row['normalized_importance']:.2f}%, ")
                    f.write(f"Cumulative: {row['cumulative_importance']:.2f}%)\n")
                
                f.write(f"\nThe top {top_n} measurements account for {top_df['cumulative_importance'].iloc[-1]:.2f}% of total direct importance.\n")
                
                # Self-attention importance
                f.write("\n\nPART 2: SELF-ATTENTION MEASUREMENT IMPORTANCE\n")
                f.write("--------------------------------------------\n")
                f.write(f"Top {top_n} most important measurements based on attention received:\n\n")
                
                top_attn_df = attn_importance_df.head(top_n).copy()
                top_attn_df['cumulative_importance'] = top_attn_df['normalized_attn_importance'].cumsum()
                
                for i, row in top_attn_df.iterrows():
                    f.write(f"{i+1}. Measurement {int(row['measurement_idx'])}: ")
                    f.write(f"Score {row['attn_importance']:.4f} ")
                    f.write(f"(Importance: {row['normalized_attn_importance']:.2f}%, ")
                    f.write(f"Cumulative: {row['cumulative_importance']:.2f}%)\n")
                
                f.write(f"\nThe top {top_n} measurements account for {top_attn_df['cumulative_importance'].iloc[-1]:.2f}% of total attention received.\n")
                
                # Combined importance
                f.write("\n\nPART 3: COMBINED MEASUREMENT IMPORTANCE\n")
                f.write("--------------------------------------\n")
                f.write(f"Top {top_n} most important measurements based on combined importance:\n\n")
                
                top_combined_df = combined_df.head(top_n).copy()
                top_combined_df['cumulative_importance'] = top_combined_df['combined_score'].cumsum()
                
                for i, row in top_combined_df.iterrows():
                    f.write(f"{i+1}. Measurement {int(row['measurement_idx'])}: ")
                    f.write(f"Score {row['combined_score']:.4f}% ")
                    f.write(f"(Cumulative: {row['cumulative_importance']:.2f}%)\n")
                
                f.write(f"\nThe top {top_n} measurements account for {top_combined_df['cumulative_importance'].iloc[-1]:.2f}% of combined importance.\n")
                
                # Overall insights
                f.write("\n\nINSIGHTS AND RECOMMENDATIONS:\n")
                f.write("----------------------------\n")
                
                # Check for overlapping important measurements
                top_direct = set(top_df['measurement_idx'].astype(int).tolist())
                top_attn = set(top_attn_df['measurement_idx'].astype(int).tolist())
                top_combined = set(top_combined_df['measurement_idx'].astype(int).tolist())
                
                overlap_all = top_direct.intersection(top_attn).intersection(top_combined)
                overlap_any = top_direct.union(top_attn).union(top_combined)
                
                f.write(f"1. {len(overlap_all)} measurements appear in the top {top_n} across all importance metrics.\n")
                if len(overlap_all) > 0:
                    f.write(f"   These crucial measurements are: {', '.join(map(str, sorted(overlap_all)))}\n")
                
                f.write(f"\n2. Most efficient measurement subset recommendation:\n")
                f.write(f"   For maximum information with minimum measurements, prioritize the top 10 from combined importance.\n")
                
                f.write(f"\n3. Distribution of importance:\n")
                direct_90 = np.argmax(top_df['cumulative_importance'].values >= 90) + 1 if any(top_df['cumulative_importance'].values >= 90) else "more than 20"
                attn_90 = np.argmax(top_attn_df['cumulative_importance'].values >= 90) + 1 if any(top_attn_df['cumulative_importance'].values >= 90) else "more than 20"
                combined_90 = np.argmax(top_combined_df['cumulative_importance'].values >= 90) + 1 if any(top_combined_df['cumulative_importance'].values >= 90) else "more than 20"
                
                f.write(f"   - {direct_90} measurements account for 90% of direct attention importance\n")
                f.write(f"   - {attn_90} measurements account for 90% of self-attention importance\n")
                f.write(f"   - {combined_90} measurements account for 90% of combined importance\n")
            
            # Return combined importance as that's most representative
            return combined_df
        else:
            # If no attention maps, just return measurement weights
            df.to_csv(os.path.join(output_dir, f'transformer_measurement_importance_{num_measurements}.csv'), index=False)
            
            # Create a summary text file
            with open(os.path.join(output_dir, f'transformer_measurement_summary_{num_measurements}.txt'), 'w') as f:
                f.write(f"TRANSFORMER MEASUREMENT IMPORTANCE ANALYSIS ({num_measurements} measurements)\n")
                f.write("======================================================\n\n")
                
                top_n = min(20, len(df))
                top_df = df.head(top_n).copy()
                top_df['cumulative_importance'] = top_df['normalized_importance'].cumsum()
                
                f.write(f"Top {top_n} most important measurements:\n\n")
                
                for i, row in top_df.iterrows():
                    f.write(f"{i+1}. Measurement {int(row['measurement_idx'])}: ")
                    f.write(f"Weight {row['attention_weight']:.4f} ")
                    f.write(f"(Importance: {row['normalized_importance']:.2f}%, ")
                    f.write(f"Cumulative: {row['cumulative_importance']:.2f}%)\n")
                
                f.write(f"\nThe top {top_n} measurements account for {top_df['cumulative_importance'].iloc[-1]:.2f}% of total importance.\n")
            
            print(f"Transformer analysis complete. Results saved to {output_dir}")
            return df
    else:
        raise ValueError(f"Model type {model_type} not supported for importance analysis")

def main():
    parser = argparse.ArgumentParser(description='Extract important measurements from trained neural networks')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.pt)')
    parser.add_argument('--measurements', type=int, required=True, help='Number of measurements the model was trained on')
    parser.add_argument('--output-dir', type=str, default='measurement_analysis', help='Directory to save outputs')
    args = parser.parse_args()
    
    try:
        print(f"Loading model from {args.model}...")
        model, model_type = load_model(args.model, args.measurements)
        
        print(f"Extracting important measurements from {model_type.upper()} model...")
        result_df = analyze_measurements(model, model_type, args.measurements, args.output_dir)
        
        print(f"Analysis complete! Results saved to {args.output_dir} directory.")
        print(f"\nTop 5 most important measurements:")
        for i, row in result_df.head(5).iterrows():
            importance_col = 'combined_score' if 'combined_score' in result_df.columns else 'normalized_importance'
            print(f"{i+1}. Measurement {int(row['measurement_idx'])}: {row[importance_col]:.2f}% importance")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
