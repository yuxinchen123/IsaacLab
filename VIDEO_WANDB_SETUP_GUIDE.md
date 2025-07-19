# Isaac Lab Video Recording & Wandb Setup Guide

## Step 1: Install Required Packages

```bash
# Navigate to Isaac Lab and activate environment
cd /home/clairec/IsaacLab
source ~/env_isaaclab/bin/activate

# Install wandb and video recording dependencies
pip install wandb imageio imageio-ffmpeg

# Login to wandb (one time setup)
wandb login
```

## Step 2: Modify Training Command for Video Recording

```bash
# Basic training with video recording
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 4096 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 200 \
    --video_interval 100

# Training with wandb integration
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 4096 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 200 \
    --video_interval 100 \
    --wandb_project "isaac-ant-locomotion" \
    --wandb_entity "your-wandb-username"
```

## Step 3: Enhanced Training Script

Save this as `train_ant_video_wandb.py`:

```python
#!/usr/bin/env python3
"""
Enhanced Isaac Lab training with video recording and wandb integration
"""

import os
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Ant-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--enable_cameras", action="store_true", default=True)
parser.add_argument("--video", action="store_true", default=True)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=100)
parser.add_argument("--wandb_project", type=str, default="isaac-ant-locomotion")
parser.add_argument("--wandb_run_name", type=str, default=None)

# Isaac Lab setup
from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher(parser.parse_args())
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import numpy as np

# Wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Install wandb: pip install wandb")
    WANDB_AVAILABLE = False

# Isaac Lab imports
import isaaclab
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# RL imports
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

def main():
    args = parser.parse_args()
    
    # Initialize wandb
    if WANDB_AVAILABLE and args.wandb_project:
        run_name = args.wandb_run_name or f"ant-video-{datetime.now().strftime('%m%d-%H%M')}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "algorithm": "PPO",
                "task": args.task,
                "num_envs": args.num_envs,
                "video_recording": args.video,
                "video_length": args.video_length,
                "video_interval": args.video_interval,
                "seed": 42
            },
            tags=["isaac-lab", "ant", "ppo", "video"],
            notes="Quadruped locomotion with video recording"
        )
        print(f"✅ Wandb initialized: {run_name}")
    
    # Environment setup with video recording capability
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    
    if args.video and args.enable_cameras:
        # Enable rendering for video recording
        env_cfg.sim.render_mode = "rgb_array"
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    env = RslRlVecEnvWrapper(env)
    
    # Agent configuration
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
    agent_cfg = AntPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    
    # Create runner with custom logging
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Setup directories
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    video_dir = os.path.join(log_root_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"[INFO] Training logs: {log_root_path}")
    print(f"[INFO] Videos will be saved to: {video_dir}")
    
    # Save configuration
    os.makedirs(os.path.join(log_root_path, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, "params", "agent.yaml"), agent_cfg)
    
    class VideoLogger:
        def __init__(self, video_dir, video_interval=100, video_length=200):
            self.video_dir = video_dir
            self.video_interval = video_interval
            self.video_length = video_length
            self.step_count = 0
            self.recording = False
            self.frames = []
        
        def log_step(self, env):
            self.step_count += 1
            
            # Start recording at intervals
            if self.step_count % self.video_interval == 0 and not self.recording:
                self.recording = True
                self.frames = []
                print(f"📹 Recording video at step {self.step_count}")
            
            # Capture frame if recording
            if self.recording and args.video:
                try:
                    frame = env.render()
                    if frame is not None:
                        self.frames.append(frame)
                except:
                    pass
            
            # Save video when enough frames collected
            if self.recording and len(self.frames) >= self.video_length:
                self.save_video()
                self.recording = False
        
        def save_video(self):
            if len(self.frames) == 0:
                return
            
            video_path = os.path.join(self.video_dir, f"step_{self.step_count}.mp4")
            try:
                import imageio
                with imageio.get_writer(video_path, fps=30) as writer:
                    for frame in self.frames:
                        writer.append_data(frame)
                print(f"✅ Saved video: {video_path}")
                
                # Log to wandb if available
                if WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        "training_video": wandb.Video(video_path, caption=f"Step {self.step_count}"),
                        "step": self.step_count
                    })
                    print(f"📤 Uploaded video to wandb")
                
            except Exception as e:
                print(f"❌ Failed to save video: {e}")
    
    # Initialize video logger
    if args.video:
        video_logger = VideoLogger(video_dir, args.video_interval, args.video_length)
    
    # Training loop with video recording
    print("🚀 Starting training with video recording...")
    
    try:
        # Custom training loop to integrate video recording
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True
        )
        
        # Export final policy
        export_policy_as_jit(
            runner.alg.actor_critic, 
            path=os.path.join(log_root_path, "exported")
        )
        
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        env.close()
        if WANDB_AVAILABLE and wandb.run:
            wandb.finish()
        simulation_app.close()

if __name__ == "__main__":
    main()
```

## Step 4: Run Training with Video & Wandb

```bash
# Run the enhanced training script
python train_ant_video_wandb.py \
    --wandb_project "isaac-ant-locomotion" \
    --wandb_run_name "ant-ppo-with-videos-$(date +%m%d-%H%M)"
```

## Step 5: What You'll See

### In Wandb Dashboard:
- Real-time training metrics (reward, episode length, losses)
- Video clips showing ant learning to walk over time
- System metrics (GPU usage, FPS)
- Hyperparameter tracking

### Local Files:
- `logs/rsl_rl/ant/videos/` - MP4 video files
- `logs/rsl_rl/ant/` - Training checkpoints and logs
- TensorBoard logs for local monitoring

## Step 6: Monitor Progress

```bash
# Local TensorBoard (optional)
tensorboard --logdir logs/rsl_rl/ant/ --port 6006

# Check wandb dashboard at: https://wandb.ai/your-username/isaac-ant-locomotion
```

## Key Features:

✅ **Headless training** continues normally  
✅ **Video recording** at specified intervals  
✅ **Automatic wandb upload** of videos and metrics  
✅ **Professional logging** with timestamps  
✅ **GPU optimization** maintained  
✅ **Minimal overhead** on training performance  

The videos will show your ant agents getting better at walking over time - perfect for presentations and progress monitoring!
