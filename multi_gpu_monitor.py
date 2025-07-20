#!/usr/bin/env python3
"""
Multi-GPU Training Monitor for Isaac Lab
Shows real-time GPU usage, memory, and process information

ESSENTIAL FILE - Keep for future multi-GPU training runs
Use: python multi_gpu_monitor.py
"""

import subprocess
import time
import os
import psutil
from datetime import datetime
import json

class MultiGPUMonitor:
    def __init__(self):
        self.num_gpus = 8
        
    def get_gpu_info(self):
        """Get detailed GPU information"""
        try:
            # Get GPU utilization and memory
            cmd = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [x.strip() for x in line.split(',')]
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4]),
                        'temperature': int(parts[5])
                    })
            return gpu_info
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return []
    
    def get_gpu_processes(self):
        """Get processes running on each GPU"""
        try:
            cmd = ["nvidia-smi", "--query-compute-apps=gpu_name,pid,process_name,used_memory", "--format=csv,noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line and 'python' in line.lower():
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 4:
                        processes.append({
                            'gpu_name': parts[0],
                            'pid': int(parts[1]),
                            'process_name': parts[2],
                            'memory_used': parts[3]
                        })
            return processes
        except Exception as e:
            print(f"Error getting GPU processes: {e}")
            return []
    
    def count_training_processes(self):
        """Count Isaac Lab training processes"""
        count = 0
        gpu_processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if any(keyword in cmdline for keyword in ['train', 'isaac', 'rsl_rl', 'rl_games']):
                        if 'python' in cmdline:
                            count += 1
                            # Try to get GPU assignment
                            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                                gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
                            else:
                                gpu_id = 'auto'
                            
                            gpu_processes[proc.info['pid']] = {
                                'name': proc.info['name'],
                                'gpu_id': gpu_id,
                                'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                            }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return count, gpu_processes
    
    def display_status(self):
        """Display comprehensive GPU and training status"""
        os.system('clear')  # Clear screen
        
        print("🔥 Multi-GPU Training Monitor for Isaac Lab")
        print("=" * 80)
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # GPU Information
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print("🖥️  GPU Status:")
            print("-" * 80)
            print(f"{'GPU':<4} {'Name':<20} {'Util%':<6} {'Mem Used':<10} {'Mem Total':<10} {'Temp°C':<7}")
            print("-" * 80)
            
            total_util = 0
            gpus_in_use = 0
            
            for gpu in gpu_info:
                util_color = "🟢" if gpu['utilization'] > 50 else "🟡" if gpu['utilization'] > 10 else "🔴"
                print(f"{gpu['index']:<4} {gpu['name']:<20} {util_color} {gpu['utilization']:<3}% "
                      f"{gpu['memory_used']:<6}MB {gpu['memory_total']:<6}MB {gpu['temperature']:<4}°C")
                
                total_util += gpu['utilization']
                if gpu['utilization'] > 10:
                    gpus_in_use += 1
            
            avg_util = total_util / len(gpu_info)
            print("-" * 80)
            print(f"📊 Summary: {gpus_in_use}/{len(gpu_info)} GPUs active, Avg Utilization: {avg_util:.1f}%")
        
        print()
        
        # Training Processes
        process_count, training_processes = self.count_training_processes()
        print(f"🚀 Training Processes: {process_count} active")
        
        if training_processes:
            print("-" * 80)
            for pid, info in training_processes.items():
                print(f"PID {pid}: {info['name']} (GPU: {info['gpu_id']})")
                print(f"   Command: {info['cmdline']}")
        
        print()
        
        # GPU Processes
        gpu_processes = self.get_gpu_processes()
        if gpu_processes:
            print("🔧 GPU Processes:")
            print("-" * 80)
            gpu_process_count = {}
            for proc in gpu_processes:
                gpu_name = proc['gpu_name']
                if gpu_name not in gpu_process_count:
                    gpu_process_count[gpu_name] = 0
                gpu_process_count[gpu_name] += 1
                print(f"GPU: {proc['gpu_name']}, PID: {proc['pid']}, Memory: {proc['memory_used']}")
            
            print("-" * 80)
            print(f"📈 Processes per GPU: {dict(gpu_process_count)}")
        
        print()
        print("🔄 Press Ctrl+C to stop monitoring")
        print("💡 For multi-GPU training, you should see:")
        print("   • Multiple processes (1 per GPU)")
        print("   • High utilization across multiple GPUs")
        print("   • Balanced memory usage across GPUs")
    
    def run_monitor(self, interval=3):
        """Run continuous monitoring"""
        try:
            while True:
                self.display_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")

def main():
    monitor = MultiGPUMonitor()
    print("🚀 Starting Multi-GPU Training Monitor...")
    monitor.run_monitor()

if __name__ == "__main__":
    main()
