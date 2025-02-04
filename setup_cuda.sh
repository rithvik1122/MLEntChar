#!/bin/bash

# Source bashrc to get conda and other environment variables
source ~/.bashrc

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Activate conda environment if you're using one
# conda activate your_environment_name  # Uncomment and modify if using conda

# Print configuration
echo "CUDA environment configured:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "PATH includes CUDA: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Python path: $(which python3)"

# Run your Python script with CUDA environment
python3 entchar.py
