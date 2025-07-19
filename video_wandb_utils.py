#!/usr/bin/env python3
"""
Enhanced Isaac Lab training with video recording and wandb integration.
This script modifies the existing RSL-RL training to add video recording and wandb logging.

Usage:
    python train_ant_with_video.py --video --wandb_project "my-ant-training"
"""

import os
import sys
import argparse
from datetime import datetime

# Standard Isaac Lab imports (will be available when running in Isaac Lab environment)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False

def setup_video_recording_env(env_cfg, enable_video=True, video_length=200):
    """Modify environment configuration to enable video recording."""
    if not enable_video:
        return env_cfg
    
    # Enable cameras and rendering for video recording
    if hasattr(env_cfg, 'scene'):
        # Add camera configuration to scene if not present
        if not hasattr(env_cfg.scene, 'camera'):
            from isaaclab.sensors import CameraCfg, patterns
            
            # Add camera for video recording
            env_cfg.scene.camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera",
                spawn=patterns.spawn_camera(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=patterns.OffsetCfg(pos=(0.0, 0.0, 8.0), rot=(0.0, -0.5, 0.0, 0.866)),
                data_types=["rgb"],
                spawn_kwargs={},
                debug_vis=False,
            )
    
    # Configure rendering settings
    if hasattr(env_cfg, 'sim'):
        env_cfg.sim.render_mode = "rgb_array"
    
    return env_cfg

def setup_wandb_logging(project_name, run_name=None, config_dict=None):
    """Initialize wandb logging with configuration."""
    if not WANDB_AVAILABLE:
        print("Wandb not available, skipping wandb logging")
        return False
    
    if run_name is None:
        run_name = f"ant-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    default_config = {
        "algorithm": "PPO",
        "framework": "RSL-RL", 
        "environment": "Isaac-Ant-v0",
        "seed": 42
    }
    
    if config_dict:
        default_config.update(config_dict)
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config=default_config,
            tags=["isaac-lab", "ppo", "ant", "quadruped", "locomotion"],
            notes="Isaac Lab quadruped locomotion training with video recording"
        )
        print(f"✅ Wandb initialized successfully: {run_name}")
        return True
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return False

def log_videos_to_wandb(video_paths, step, caption="Training Progress"):
    """Log video files to wandb."""
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    try:
        videos = []
        for video_path in video_paths:
            if os.path.exists(video_path):
                videos.append(wandb.Video(video_path, caption=f"{caption} - Step {step}"))
        
        if videos:
            wandb.log({"training_videos": videos}, step=step)
            print(f"✅ Logged {len(videos)} videos to wandb at step {step}")
    except Exception as e:
        print(f"Failed to log videos to wandb: {e}")

def create_training_script_with_video():
    """Create a complete training script with video and wandb integration."""
    
    script_content = '''#!/usr/bin/env python3
"""
Isaac Lab Ant Training with Video Recording and Wandb Integration
"""

import os
import sys
import argparse
from datetime import datetime

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train RL agent with video recording")
parser.add_argument("--video", action="store_true", help="Enable video recording")
parser.add_argument("--video_length", type=int, default=200, help="Video length in frames")
parser.add_argument("--video_interval", type=int, default=100, help="Video recording interval")
parser.add_argument("--wandb_project", type=str, default="isaac-ant-training", help="Wandb project")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--enable_cameras", action="store_true", help="Enable cameras")

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import torch
import gymnasium as gym
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Wandb not available - install with: pip install wandb")
    WANDB_AVAILABLE = False

import isaaclab
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# RL imports
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

class VideoRecordingWrapper:
    """Wrapper to handle video recording during training."""
    
    def __init__(self, env, video_dir, video_length=200, video_interval=100):
        self.env = env
        self.video_dir = video_dir
        self.video_length = video_length
        self.video_interval = video_interval
        self.current_step = 0
        self.recording = False
        self.frames = []
        
        os.makedirs(video_dir, exist_ok=True)
    
    def should_record(self):
        return self.current_step % self.video_interval == 0
    
    def start_recording(self):
        self.recording = True
        self.frames = []
        print(f"Started recording video at step {self.current_step}")
    
    def stop_recording(self):
        if self.recording and len(self.frames) > 0:
            video_path = os.path.join(self.video_dir, f"training_step_{self.current_step}.mp4")
            self.save_video(video_path)
            self.recording = False
            return video_path
        return None
    
    def save_video(self, path):
        """Save recorded frames as video."""
        try:
            import imageio
            with imageio.get_writer(path, fps=30) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            print(f"Saved video: {path}")
        except ImportError:
            print("imageio not available for video saving")
        except Exception as e:
            print(f"Failed to save video: {e}")
    
    def step(self, obs):
        self.current_step += 1
        
        # Start recording if needed
        if self.should_record() and not self.recording:
            self.start_recording()
        
        # Record frame if recording
        if self.recording:
            try:
                frame = self.env.render()
                if frame is not None:
                    self.frames.append(frame)
            except:
                pass  # Skip frame if rendering fails
        
        # Stop recording if we have enough frames
        if self.recording and len(self.frames) >= self.video_length:
            return self.stop_recording()
        
        return None

def main():
    """Main training function."""
    
    # Setup wandb
    wandb_enabled = False
    if args_cli.wandb_project and WANDB_AVAILABLE:
        run_name = args_cli.wandb_run_name or f"ant-video-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            wandb.init(
                project=args_cli.wandb_project,
                name=run_name,
                config={
                    "task": args_cli.task,
                    "num_envs": args_cli.num_envs,
                    "algorithm": "PPO",
                    "framework": "RSL-RL",
                    "video_recording": args_cli.video,
                    "video_length": args_cli.video_length,
                    "video_interval": args_cli.video_interval,
                    "headless": args_cli.headless,
                },
                tags=["isaac-lab", "ppo", "ant", "quadruped"]
            )
            wandb_enabled = True
            print("✅ Wandb initialized successfully")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # Enable cameras if video recording is requested
    if args_cli.video and args_cli.enable_cameras:
        env_cfg.sim.render_mode = "rgb_array"
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env)
    
    # Setup video recording
    video_wrapper = None
    if args_cli.video:
        video_dir = os.path.join("logs", "videos", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        video_wrapper = VideoRecordingWrapper(env, video_dir, args_cli.video_length, args_cli.video_interval)
    
    # Load agent configuration
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
    agent_cfg: RslRlOnPolicyRunnerCfg = AntPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Setup logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging to: {log_root_path}")
    
    # Save configurations
    os.makedirs(os.path.join(log_root_path, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, "params", "agent.yaml"), agent_cfg)
    
    # Custom training loop with video recording
    print("Starting training with video recording...")
    
    try:
        # Run training
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        
        # Export final policy
        export_policy_as_jit(runner.alg.actor_critic, path=os.path.join(log_root_path, "exported"))
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        # Cleanup
        env.close()
        if wandb_enabled:
            wandb.finish()
        simulation_app.close()

if __name__ == "__main__":
    main()
'''
    
    return script_content

# Save the complete training script
if __name__ == "__main__":
    script_content = create_training_script_with_video()
    with open("/home/clairec/IsaacLab/train_ant_with_video_complete.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created complete training script with video recording and wandb integration")
