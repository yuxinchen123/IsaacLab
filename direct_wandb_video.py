#!/usr/bin/env python3
"""
Enhanced Isaac Lab training with direct wandb video streaming (no local storage).
This approach records videos in memory and uploads directly to wandb without saving locally.
"""

import os
import tempfile
import argparse
from datetime import datetime
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import imageio
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False

class WandbVideoStreamer:
    """Stream videos directly to wandb without local storage."""
    
    def __init__(self, video_interval=100, video_length=200, fps=30):
        self.video_interval = video_interval
        self.video_length = video_length
        self.fps = fps
        self.step_count = 0
        self.recording = False
        self.frames = []
        
        print(f"📹 Direct wandb video streaming initialized (no local storage)")
    
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
        
        # Upload video when enough frames collected
        if self.recording and len(self.frames) >= self.video_length:
            return self.upload_to_wandb()
        
        return None
    
    def start_recording(self):
        """Start video recording."""
        self.recording = True
        self.frames = []
        print(f"🎬 Started recording video at step {self.step_count} (direct upload)")
    
    def capture_frame(self, env):
        """Capture a frame from the environment."""
        try:
            frame = env.render()
            if frame is not None and len(frame.shape) == 3:
                self.frames.append(frame)
        except Exception as e:
            pass  # Skip frame if rendering fails
    
    def upload_to_wandb(self):
        """Upload video directly to wandb using temporary file."""
        if not self.recording or len(self.frames) == 0:
            return None
        
        if not WANDB_AVAILABLE or not wandb.run:
            print("❌ Wandb not available for video upload")
            self.recording = False
            return None
        
        try:
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write video to temporary file
            with imageio.get_writer(temp_path, fps=self.fps) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            
            # Upload to wandb
            wandb.log({
                "training_video": wandb.Video(temp_path, caption=f"Training Step {self.step_count}"),
                "video_step": self.step_count
            }, step=self.step_count)
            
            # Clean up temporary file immediately
            os.unlink(temp_path)
            
            print(f"✅ Uploaded video to wandb (step {self.step_count}) - no local storage")
            self.recording = False
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload video to wandb: {e}")
            # Clean up temp file if it exists
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
        
        self.recording = False
        return None


class InMemoryVideoStreamer:
    """Stream video data directly as numpy arrays to wandb."""
    
    def __init__(self, video_interval=100, video_length=200, fps=30):
        self.video_interval = video_interval
        self.video_length = video_length
        self.fps = fps
        self.step_count = 0
        self.recording = False
        self.frames = []
        
        print(f"📹 In-memory video streaming initialized (no files at all)")
    
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
        if self.recording:
            self.capture_frame(env)
        
        # Upload video when enough frames collected
        if self.recording and len(self.frames) >= self.video_length:
            return self.stream_to_wandb()
        
        return None
    
    def start_recording(self):
        """Start video recording."""
        self.recording = True
        self.frames = []
        print(f"🎬 Started recording video at step {self.step_count} (in-memory)")
    
    def capture_frame(self, env):
        """Capture a frame from the environment."""
        try:
            frame = env.render()
            if frame is not None and len(frame.shape) == 3:
                self.frames.append(frame)
        except Exception as e:
            pass  # Skip frame if rendering fails
    
    def stream_to_wandb(self):
        """Stream video data directly to wandb as numpy array."""
        if not self.recording or len(self.frames) == 0:
            return None
        
        if not WANDB_AVAILABLE or not wandb.run:
            print("❌ Wandb not available for video upload")
            self.recording = False
            return None
        
        try:
            # Convert frames to numpy array (T, H, W, C)
            video_array = np.array(self.frames)
            
            # Log as wandb video (wandb handles the conversion internally)
            wandb.log({
                "training_video": wandb.Video(
                    video_array, 
                    fps=self.fps, 
                    caption=f"Training Step {self.step_count}"
                ),
                "video_step": self.step_count
            }, step=self.step_count)
            
            print(f"✅ Streamed video to wandb (step {self.step_count}) - pure in-memory")
            self.recording = False
            return True
            
        except Exception as e:
            print(f"❌ Failed to stream video to wandb: {e}")
        
        self.recording = False
        return None


def create_direct_upload_training_script():
    """Create training script with direct wandb upload (no local storage)."""
    
    script_content = '''#!/usr/bin/env python3
"""
Isaac Lab Ant Training with Direct Wandb Video Upload (No Local Storage)
"""

import argparse
from datetime import datetime

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train RL agent with direct wandb video upload")
parser.add_argument("--video", action="store_true", help="Enable video recording")
parser.add_argument("--video_length", type=int, default=200, help="Video length in frames")
parser.add_argument("--video_interval", type=int, default=100, help="Video recording interval")
parser.add_argument("--wandb_project", type=str, default="isaac-ant-locomotion", help="Wandb project")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--use_memory_streaming", action="store_true", 
                    help="Use pure in-memory streaming (no temp files)")

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Standard imports
import gymnasium as gym
import torch

# Isaac Lab imports
import isaaclab
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# RL imports
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

# Import our video streamers
from direct_wandb_video import WandbVideoStreamer, InMemoryVideoStreamer

def setup_wandb(args):
    """Initialize wandb logging."""
    try:
        import wandb
        
        run_name = args.wandb_run_name or f"ant-direct-upload-{datetime.now().strftime('%m%d-%H%M')}"
        
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
                "direct_upload": True,
                "no_local_storage": True,
            },
            tags=["isaac-lab", "ant", "ppo", "direct-upload", "no-local-storage"],
            notes="Isaac Lab training with direct wandb video upload (no local storage)"
        )
        print(f"✅ Wandb initialized: {run_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize wandb: {e}")
        return False

def main():
    """Main training function."""
    
    # Setup wandb
    wandb_enabled = setup_wandb(args_cli)
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env)
    
    # Agent configuration
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
    agent_cfg = AntPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    
    # Create runner with minimal logging (no video directory)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Setup video streamer (choose method)
    video_streamer = None
    if args_cli.video and wandb_enabled:
        if args_cli.use_memory_streaming:
            video_streamer = InMemoryVideoStreamer(
                video_interval=args_cli.video_interval,
                video_length=args_cli.video_length
            )
        else:
            video_streamer = WandbVideoStreamer(
                video_interval=args_cli.video_interval,
                video_length=args_cli.video_length
            )
    
    print("🚀 Starting training with direct wandb video upload...")
    
    try:
        # Training with video streaming
        # Note: This is a simplified example - you'd need to integrate the video_streamer.step() 
        # into the actual training loop of RSL-RL
        
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        env.close()
        if wandb_enabled:
            import wandb
            wandb.finish()
        simulation_app.close()

if __name__ == "__main__":
    main()
'''
    
    with open("train_ant_direct_wandb_upload.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created train_ant_direct_wandb_upload.py")

if __name__ == "__main__":
    # Demo the video streamers
    print("📹 Direct Wandb Video Upload Demo")
    print("\n1. WandbVideoStreamer - Uses temporary files")
    print("2. InMemoryVideoStreamer - Pure in-memory streaming")
    
    create_direct_upload_training_script()
