# Quick Implementation Guide: Video Recording + Wandb for Isaac Lab

## 🚀 Simple Setup (Copy-Paste Commands)

### Step 1: Install Dependencies
```bash
# SSH back to your Isaac Lab server
ssh clairec@pabrtxl1
cd /home/clairec/IsaacLab
source ~/env_isaaclab/bin/activate

# Install required packages
pip install wandb imageio imageio-ffmpeg

# Setup wandb (one-time)
wandb login
# Enter your wandb API key when prompted
```

### Step 2: Run Training with Video Recording
```bash
# Enhanced training command with video recording
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 4096 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 200 \
    --video_interval 100 \
    --wandb_project "isaac-ant-locomotion" \
    --wandb_run_name "ant-ppo-video-$(date +%m%d-%H%M)"
```

### Step 3: Use Our Enhanced Script
```bash
# Run our custom video-enabled training script
python train_ant_video_wandb.py \
    --wandb_project "isaac-ant-locomotion" \
    --wandb_run_name "ant-ppo-with-videos"
```

## 📋 What This Does

### Video Recording:
- ✅ Records 200-frame videos every 100 training steps
- ✅ Saves MP4 files locally in `logs/rsl_rl/ant/videos/`
- ✅ Works in headless mode (no GUI needed)
- ✅ Shows ant agents learning to walk over time

### Wandb Integration:
- ✅ Real-time training metrics dashboard
- ✅ Automatic video upload to wandb
- ✅ Professional experiment tracking
- ✅ Shareable results with your mentor

### Expected Results:
- **Training continues** as before (same +145 reward improvement)
- **Videos saved locally** in logs directory
- **Wandb dashboard** shows progress with embedded videos
- **Professional presentation** of your results

## 🎯 Alternative: Simple Video Recording Only

If wandb setup is complex, just enable video recording:

```bash
# Simple video recording without wandb
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 4096 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 200 \
    --video_interval 50
```

This will:
- ✅ Record videos every 50 steps
- ✅ Save videos to `logs/rsl_rl/ant/videos/`
- ✅ Keep all your successful training setup
- ✅ Allow you to manually upload videos later

## 📱 Accessing Results

### Wandb Dashboard:
- Visit: `https://wandb.ai/your-username/isaac-ant-locomotion`
- View real-time training curves
- Watch embedded training videos
- Share dashboard with mentor

### Local Videos:
```bash
# Check saved videos
ls -la logs/rsl_rl/ant/*/videos/
```

## 🎬 What the Videos Will Show

Your videos will capture:
- **Early training**: Ants falling and struggling
- **Mid training**: Ants starting to coordinate movements  
- **Late training**: Ants walking smoothly and efficiently
- **Progress over time**: Clear improvement in locomotion

Perfect for showing your mentor the learning progression! 🐜➡️🚶‍♂️
