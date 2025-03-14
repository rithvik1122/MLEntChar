#!/usr/bin/env python3
"""
Debugging utility to analyze memory usage and diagnose multiprocessing issues.
Run this script to get system information and test parallel processing.
"""

import os
import sys
import time
import psutil
import argparse
import numpy as np
import concurrent.futures
from functools import partial
import traceback

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_system_info():
    """Get detailed system information"""
    print("\n=== System Information ===")
    
    # CPU info
    print("\n--- CPU Information ---")
    cpu_count = os.cpu_count()
    print(f"Logical CPUs: {cpu_count}")
    
    try:
        with open('/proc/cpuinfo') as f:
            cpu_info = f.read()
        
        # Extract model name
        for line in cpu_info.split('\n'):
            if 'model name' in line:
                print(f"CPU Model: {line.split(':', 1)[1].strip()}")
                break
    except:
        print("Could not retrieve detailed CPU info")
    
    # Memory info
    print("\n--- Memory Information ---")
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / (1024**3):.2f} GB")
    print(f"Available: {mem.available / (1024**3):.2f} GB")
    print(f"Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    
    # Swap info
    swap = psutil.swap_memory()
    print(f"Swap Total: {swap.total / (1024**3):.2f} GB")
    print(f"Swap Used: {swap.used / (1024**3):.2f} GB ({swap.percent}%)")
    
    # Process info
    print("\n--- Process Information ---")
    process = psutil.Process(os.getpid())
    print(f"Current Process Memory Usage: {process.memory_info().rss / (1024**3):.2f} GB")
    
    # PyTorch info if available
    if HAS_TORCH:
        print("\n--- PyTorch Information ---")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
    
    # Other info
    print("\n--- OS Information ---")
    print(f"Python Version: {sys.version}")
    print(f"Process ID: {os.getpid()}")
    
    # Check process limits (Linux only)
    try:
        import resource
        print("\n--- Resource Limits ---")
        # Max open files
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Max Open Files: {soft} (soft), {hard} (hard)")
        
        # Max processes
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        print(f"Max Processes: {soft} (soft), {hard} (hard)")
        
        # Max memory
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft == -1:
            print("Max Memory: unlimited (soft)")
        else:
            print(f"Max Memory: {soft / (1024**3):.2f} GB (soft)")
            
        if hard == -1:
            print("Max Memory: unlimited (hard)")
        else:
            print(f"Max Memory: {hard / (1024**3):.2f} GB (hard)")
    except:
        print("Could not access resource limits")

def test_worker(sleep_time, array_size, worker_id):
    """Test worker function for parallel processing"""
    # Allocate some memory
    data = np.random.random((array_size, array_size))
    
    # Simulate some work
    time.sleep(sleep_time)
    
    # Do some calculations
    result = np.mean(data) + np.std(data)
    
    # Memory usage for this worker
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024**2)
    
    return f"Worker {worker_id}, Memory: {memory_mb:.2f} MB, Result: {result:.6f}"

def test_multiprocessing(num_workers, array_size=1000, sleep_time=1.0, num_tasks=10):
    """Test multiprocessing with multiple workers"""
    print(f"\nTesting parallel processing with {num_workers} workers")
    print(f"Array size: {array_size}x{array_size} ({array_size**2*8/1024**2:.2f} MB per worker)")
    print(f"Tasks: {num_tasks}")
    
    start_time = time.time()
    results = []
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(test_worker, sleep_time, array_size, i) 
                      for i in range(num_tasks)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  {result}")
                except Exception as e:
                    print(f"  ERROR: {e}")
    except Exception as e:
        print(f"ERROR in multiprocessing: {e}")
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Success rate: {len(results)}/{num_tasks} tasks")
    
    return len(results) == num_tasks

def test_torch_multiprocessing():
    """Test PyTorch's multiprocessing capabilities"""
    if not HAS_TORCH:
        print("\nPyTorch not installed, skipping torch multiprocessing test")
        return
    
    try:
        import torch.multiprocessing as mp
        print("\n=== Testing PyTorch Multiprocessing ===")
        
        def worker_fn(rank):
            # Create a tensor
            x = torch.randn(1000, 1000)
            # Do some computation
            y = x @ x.t()
            # Return sum
            return y.sum().item()
        
        # Test with spawn method
        mp.set_start_method('spawn', force=True)
        ctx = mp.get_context('spawn')
        
        start_time = time.time()
        num_workers = min(4, os.cpu_count())
        print(f"Starting {num_workers} workers with spawn method...")
        
        with ctx.Pool(num_workers) as pool:
            results = pool.map(worker_fn, range(num_workers))
            print(f"Results: {[round(r, 2) for r in results]}")
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"ERROR in torch multiprocessing: {e}")
        traceback.print_exc()

def monitor_memory_usage(duration=10, interval=1):
    """Monitor memory usage over time"""
    print(f"\n=== Monitoring Memory Usage for {duration} seconds ===")
    print("Time  |  Process Memory  |  System Memory")
    print("------|-----------------|---------------")
    
    process = psutil.Process(os.getpid())
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Get process memory
        proc_mem = process.memory_info().rss / (1024**3)
        
        # Get system memory
        sys_mem = psutil.virtual_memory()
        sys_used = sys_mem.used / (1024**3)
        sys_percent = sys_mem.percent
        
        # Print current state
        elapsed = time.time() - start_time
        print(f"{elapsed:5.1f}s |  {proc_mem:6.3f} GB  |  {sys_used:6.3f} GB ({sys_percent:2.0f}%)")
        
        # Wait for next interval
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description='Debug memory and multiprocessing issues')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of workers for multiprocessing test')
    parser.add_argument('--array-size', type=int, default=1000,
                        help='Size of test arrays (NxN)')
    parser.add_argument('--tasks', type=int, default=10,
                        help='Number of tasks to run in parallel')
    parser.add_argument('--monitor', type=int, default=10,
                        help='Duration in seconds to monitor memory')
    parser.add_argument('--skip-mp', action='store_true',
                        help='Skip multiprocessing tests')
    args = parser.parse_args()
    
    # Show basic system info
    get_system_info()
    
    # Test multiprocessing if not skipped
    if not args.skip_mp:
        # Determine number of workers
        if args.workers is None:
            workers = max(2, os.cpu_count() - 1)  # Leave one CPU for system
        else:
            workers = args.workers
        
        # Run multiprocessing test
        test_multiprocessing(
            num_workers=workers, 
            array_size=args.array_size,
            num_tasks=args.tasks
        )
        
        # Test pytorch multiprocessing
        test_torch_multiprocessing()
    
    # Monitor memory usage
    monitor_memory_usage(duration=args.monitor)
    
    print("\nDebug diagnostics complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDiagnostics interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error in diagnostics: {e}")
        traceback.print_exc()
        sys.exit(1)
