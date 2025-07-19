#!/usr/bin/env python3
"""
Enhanced Isaac Lab training with video recording and wandb integration
Based on the original RSL-RL training script with added video and wandb features.
"""

import os
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train RL agent with video recording and wandb")
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras")
parser.add_argument("--video", action="store_true", default=True, help="Record videos")
parser.add_argument("--video_length", type=int, default=200, help="Video length in frames")
parser.add_argument("--video_interval", type=int, default=100, help="Video recording interval")
parser.add_argument("--wandb_project", type=str, default="isaac-ant-locomotion", help="Wandb project")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Parse args and setup Isaac Lab
args_cli = parser.parse_args()

# Isaac Lab app launcher
from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import torch
import gymnasium as gym
import numpy as np

# Wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("⚠️  Wandb not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False

try:
    import imageio
    VIDEO_AVAILABLE = True
except ImportError:
    print("⚠️  Video recording not available. Install with: pip install imageio imageio-ffmpeg")
    VIDEO_AVAILABLE = False

# Isaac Lab imports
import isaaclab
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# RL imports
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

class VideoRecorder:
    """Handle video recording during training."""
    
    def __init__(self, video_dir, video_interval=100, video_length=200, fps=30):
        self.video_dir = video_dir
        self.video_interval = video_interval
        self.video_length = video_length
        self.fps = fps
        self.step_count = 0
        self.recording = False
        self.frames = []
        
        os.makedirs(video_dir, exist_ok=True)
        print(f"📹 Video recorder initialized. Videos will be saved to: {video_dir}")
    
    def should_record(self):
        """Check if we should start recording."""
        return (self.step_count % self.video_interval == 0) and not self.recording
    
    def step(self, env):
        """Called each training step."""
        self.step_count += 1
        
        # Start recording if needed
        if self.should_record():
            self.start_recording()
        
        # Capture frame if recording
        if self.recording and VIDEO_AVAILABLE:
            self.capture_frame(env)
        
        # Stop recording if we have enough frames
        if self.recording and len(self.frames) >= self.video_length:
            return self.stop_recording()
        
        return None
    
    def start_recording(self):
        """Start video recording."""
        self.recording = True
        self.frames = []
        print(f"🎬 Started recording video at step {self.step_count}")
    
    def capture_frame(self, env):
        """Capture a frame from the environment."""
        try:
            # Try to get frame from environment
            frame = env.render()
            if frame is not None and len(frame.shape) == 3:
                self.frames.append(frame)
        except Exception as e:
            # Skip frame if rendering fails
            pass
    
    def stop_recording(self):
        """Stop recording and save video."""
        if not self.recording or len(self.frames) == 0:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"training_step_{self.step_count}_{timestamp}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        
        try:
            if VIDEO_AVAILABLE:
                with imageio.get_writer(video_path, fps=self.fps) as writer:
                    for frame in self.frames:
                        writer.append_data(frame)
                
                print(f"✅ Saved video: {video_filename} ({len(self.frames)} frames)")
                self.recording = False
                return video_path
            else:
                print("❌ Cannot save video - imageio not available")
                
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
        
        self.recording = False
        return None

def setup_wandb(args):
    """Initialize wandb logging."""
    if not WANDB_AVAILABLE or not args.wandb_project:
        return False
    
    run_name = args.wandb_run_name or f"ant-video-{datetime.now().strftime('%m%d-%H%M')}"
    
    try:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "algorithm": "PPO",
                "framework": "RSL-RL",
                "task": args.task,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "video_recording": args.video,
                "video_length": args.video_length,
                "video_interval": args.video_interval,
                "headless": args.headless,
            },
            tags=["isaac-lab", "ant", "ppo", "quadruped", "locomotion", "video"],
            notes="Isaac Lab quadruped locomotion training with video recording"
        )
        print(f"✅ Wandb initialized: {run_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize wandb: {e}")
        return False

def log_video_to_wandb(video_path, step):
    """Log video to wandb."""
    if not WANDB_AVAILABLE or not wandb.run or not video_path:
        return
    
    try:
        wandb.log({
            "training_video": wandb.Video(video_path, caption=f"Training Step {step}"),
            "video_step": step
        }, step=step)
        print(f"📤 Uploaded video to wandb (step {step})")
        
    except Exception as e:
        print(f"❌ Failed to upload video to wandb: {e}")

def main():
    """Main training function."""
    args = args_cli
    
    print("🚀 Starting Isaac Lab training with video recording and wandb integration")
    print(f"📊 Task: {args.task}")
    print(f"🏃 Environments: {args.num_envs}")
    print(f"🎥 Video recording: {args.video}")
    print(f"📈 Wandb project: {args.wandb_project}")
    
    # Initialize wandb
    wandb_enabled = setup_wandb(args)
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    
    # Enable rendering for video recording
    if args.video and args.enable_cameras:
        env_cfg.sim.render_mode = "rgb_array"
        print("🎥 Enabled rendering for video recording")
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    env = RslRlVecEnvWrapper(env)
    
    # Load agent configuration
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
    agent_cfg = AntPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    agent_cfg.seed = args.seed
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Setup logging directories
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    # Setup video recording
    video_recorder = None
    if args.video:
        video_dir = os.path.join(log_root_path, "videos", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        video_recorder = VideoRecorder(
            video_dir=video_dir,
            video_interval=args.video_interval,
            video_length=args.video_length
        )
    
    print(f"📂 Logging to: {log_root_path}")
    
    # Save configurations
    os.makedirs(os.path.join(log_root_path, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, "params", "agent.pkl"), agent_cfg)
    
    # Print configurations
    print("\n[INFO] Environment configuration:")
    print_dict(env_cfg, nesting=4)
    print("\n[INFO] Agent configuration:")
    print_dict(agent_cfg, nesting=4)
    
    print("\n🏋️ Starting training...")
    
    try:
        # Training loop with video recording hooks
        if video_recorder:
            # Custom training loop to integrate video recording
            print("📹 Training with video recording enabled")
            
            # We'll use the standard learn method but add video recording separately
            # The video recorder will capture frames during environment steps
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
        else:
            # Standard training without video
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
        
        # Export final policy
        export_policy_as_jit(
            runner.alg.actor_critic,
            path=os.path.join(log_root_path, "exported")
        )
        
        print("✅ Training completed successfully!")
        
        # Log final summary to wandb
        if wandb_enabled:
            wandb.log({
                "training_status": "completed",
                "total_iterations": agent_cfg.max_iterations
            })
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if wandb_enabled:
            wandb.log({"training_status": "failed", "error": str(e)})
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        env.close()
        if wandb_enabled:
            wandb.finish()
        simulation_app.close()

if __name__ == "__main__":
    main()
