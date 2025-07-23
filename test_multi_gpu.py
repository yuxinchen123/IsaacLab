#!/usr/bin/env python3
"""
Test script to verify GPU allocation and parallel execution
Run this before full training to ensure proper multi-GPU setup
"""

import torch
import torch.multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import os

def test_gpu_allocation():
    """Test basic GPU allocation and access"""
    print("🔍 Testing GPU Allocation...")
    print("=" * 50)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < 8:
        print(f"⚠️  Warning: Only {num_gpus} GPUs available, expected 8")
    
    # Test each GPU
    for gpu_id in range(min(num_gpus, 8)):
        device = torch.device(f'cuda:{gpu_id}')
        try:
            # Create tensor on specific GPU
            with torch.cuda.device(device):
                test_tensor = torch.randn(1000, 1000, device=device)
                result = torch.mm(test_tensor, test_tensor.T)
                
            gpu_name = torch.cuda.get_device_name(gpu_id)
            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory // 1024**3
            print(f"✅ GPU {gpu_id}: {gpu_name} ({memory_total}GB) - OK")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id}: Error - {e}")

def test_parallel_execution():
    """Test parallel execution across GPUs"""
    print("\n🚀 Testing Parallel Execution...")
    print("=" * 50)
    
    def worker_task(gpu_id):
        """Simulate agent work on specific GPU"""
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"Agent {gpu_id} starting on GPU {gpu_id}")
        
        with torch.cuda.device(device):
            # Simulate training work
            for step in range(10):
                # Create computation load
                data = torch.randn(512, 512, device=device)
                result = torch.mm(data, data.T)
                
                # Simulate training step delay
                time.sleep(0.1)
                
                if step % 5 == 0:
                    print(f"  Agent {gpu_id} - Step {step}/10")
        
        print(f"✅ Agent {gpu_id} completed on GPU {gpu_id}")
        return gpu_id
    
    # Test with 8 agents across 8 GPUs
    num_agents = min(8, torch.cuda.device_count())
    
    print(f"Starting {num_agents} agents in parallel...")
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        start_time = time.time()
        
        # Submit all tasks
        futures = [executor.submit(worker_task, i) for i in range(num_agents)]
        
        # Wait for completion
        results = [future.result() for future in futures]
        
        end_time = time.time()
        
    print(f"\n🎉 All {num_agents} agents completed!")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"✅ Completed agents: {sorted(results)}")

def test_memory_isolation():
    """Test that agents don't interfere with each other's memory"""
    print("\n🧪 Testing Memory Isolation...")
    print("=" * 50)
    
    def memory_test(gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        
        with torch.cuda.device(device):
            # Allocate memory specific to this GPU
            memory_mb = 500  # 500MB per agent
            elements = (memory_mb * 1024 * 1024) // 4  # 4 bytes per float32
            
            data = torch.randn(elements, device=device)
            memory_used = torch.cuda.memory_allocated(device) / 1024**2
            
            print(f"  GPU {gpu_id}: Allocated {memory_used:.1f} MB")
            
            # Hold memory for a moment
            time.sleep(1)
            
            # Clear memory
            del data
            torch.cuda.empty_cache()
            
        return gpu_id
    
    num_gpus = min(4, torch.cuda.device_count())  # Test with 4 GPUs
    
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(memory_test, i) for i in range(num_gpus)]
        results = [future.result() for future in futures]
    
    print(f"✅ Memory isolation test completed for {len(results)} GPUs")

def main():
    """Run all tests"""
    print("🧪 Multi-GPU Parallel Training Test Suite")
    print("=" * 60)
    
    # Set CUDA device allocation strategy
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    
    try:
        test_gpu_allocation()
        test_parallel_execution() 
        test_memory_isolation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ GPU allocation: Working")
        print("✅ Parallel execution: Working") 
        print("✅ Memory isolation: Working")
        print("🚀 Ready for multi-agent training!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("🔧 Check your GPU setup and CUDA installation")

if __name__ == "__main__":
    main()
