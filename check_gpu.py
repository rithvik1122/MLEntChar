import torch
import sys
import subprocess
import os

def check_nvidia_smi():
    try:
        nvidia_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        print("\n=== NVIDIA-SMI Output ===")
        print(nvidia_output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: nvidia-smi not found or failed. GPU might not be present or drivers not installed.")

def check_cuda_installation():
    print("=== CUDA Installation Check ===")
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is NOT available!")
        
def check_pytorch_installation():
    print("\n=== PyTorch Installation ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch debug build? {torch.version.debug}")
    print(f"PyTorch configuration:\n{torch.__config__.show()}")

def check_system_path():
    print("\n=== System PATH ===")
    cuda_path = [p for p in os.environ.get('PATH', '').split(':') if 'cuda' in p.lower()]
    if cuda_path:
        print("CUDA in PATH:", cuda_path)
    else:
        print("No CUDA directories found in PATH")

def check_cuda_environment():
    print("\n=== CUDA Environment Variables ===")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

def main():
    print("=== GPU Diagnostic Report ===")
    print(f"Python version: {sys.version}")
    
    check_pytorch_installation()
    check_cuda_installation()
    check_nvidia_smi()
    check_system_path()
    check_cuda_environment()
    
    print("\n=== Simple GPU Test ===")
    # Try to create a CUDA tensor
    try:
        x = torch.cuda.FloatTensor(1)
        print("Successfully created a CUDA tensor!")
    except Exception as e:
        print(f"Failed to create CUDA tensor: {str(e)}")
        
    # Test CUDA memory
    if torch.cuda.is_available():
        try:
            print("\nTesting CUDA memory allocation...")
            test_tensor = torch.zeros(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("CUDA memory allocation test passed!")
        except Exception as e:
            print(f"CUDA memory allocation test failed: {str(e)}")

if __name__ == "__main__":
    main()
