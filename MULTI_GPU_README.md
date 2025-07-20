# Isaac Lab Multi-GPU Training Setup

## Essential Files

This directory contains the minimal set of files needed for multi-GPU training with Isaac Lab:

### 🚀 `train_multi_gpu.sh`
**Main multi-GPU training launcher**
- Supports 1-8 GPUs with automatic environment distribution
- Interactive menu for different configurations
- Handles 4096-65536 environments efficiently
- Built-in monitoring setup

**Usage:**
```bash
./train_multi_gpu.sh
```

### 📊 `multi_gpu_monitor.py`
**Real-time GPU monitoring utility**
- Shows GPU usage, memory, temperature
- Displays running processes and training progress
- Updates every 2 seconds

**Usage:**
```bash
python multi_gpu_monitor.py
```

## Quick Start

1. **Launch training:**
   ```bash
   ./train_multi_gpu.sh
   ```

2. **Monitor in another terminal:**
   ```bash
   python multi_gpu_monitor.py
   ```

3. **Choose your configuration:**
   - Option 4: All 8 GPUs (32,768 envs) - Recommended
   - Option 5: High-Memory (65,536 envs) - If you have enough VRAM
   - Option 6: Low-Memory (16,384 envs) - For limited VRAM

## Hardware Requirements

- **8x Quadro RTX 6000 (24GB each)** - Your current setup
- **Isaac Lab environment activated**
- **NCCL-capable CUDA installation**

## Training Tips

- Use `--headless` for better performance
- Start with Isaac-Cartpole-v0 for quick tests
- Switch to Isaac-Ant-v0 for full locomotion training
- Monitor GPU memory usage to optimize environment count

---

*All demonstration and test files have been cleaned up. These two files are all you need for future multi-GPU training runs.*
