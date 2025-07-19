#!/usr/bin/env python3
"""
Enhanced Isaac Lab training script with video recording and wandb integration enabled by default.
Based on the original RSL-RL training script with optimized settings for video + wandb.
"""

import argparse
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# Import cli_args from the original location
sys.path.append('/home/clairec/IsaacLab/scripts/reinforcement_learning/rsl_rl')
import cli_args

# Enhanced argument parser with video and wandb defaults
parser = argparse.ArgumentParser(description="Enhanced Isaac Lab training with video recording and wandb integration")

# Basic training arguments
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=1000, help="Training iterations")

# Video recording arguments (enabled by default)
parser.add_argument("--video", action="store_true", default=True, help="Record training videos")
parser.add_argument("--video_length", type=int, default=200, help="Video length in frames")
parser.add_argument("--video_interval", type=int, default=100, help="Steps between video recordings")

# Camera arguments
parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras for video")

# Add original RSL-RL arguments (this will include --logger and --log_project_name)
cli_args.add_rsl_rl_args(parser)

# Add AppLauncher arguments (this will include --headless)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli, hydra_args = parser.parse_known_args()

# Set defaults for wandb if not provided
if args_cli.logger is None:
    args_cli.logger = "wandb"
if args_cli.log_project_name is None:
    args_cli.log_project_name = "isaac-ant-locomotion"

# Auto-enable cameras if video recording is enabled
if args_cli.video:
    args_cli.enable_cameras = True

# Set default run name with timestamp if not provided
if not hasattr(args_cli, 'run_name') or args_cli.run_name is None:
    timestamp = datetime.now().strftime("%m%d-%H%M")
    args_cli.run_name = f"ant-video-{timestamp}"

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import torch
import gymnasium as gym
import os
from datetime import datetime

# Isaac Lab imports
import isaaclab
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# RL imports
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Wandb not available. Install with: pip install wandb")

def setup_enhanced_logging(args, agent_cfg):
    """Setup enhanced logging with wandb integration."""
    
    # Configure wandb if available
    if WANDB_AVAILABLE and args.logger == "wandb":
        # Initialize wandb with enhanced configuration
        wandb_config = {
            "algorithm": "PPO",
            "framework": "RSL-RL",
            "task": args.task,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "video_recording": args.video,
            "video_length": args.video_length,
            "video_interval": args.video_interval,
            "headless": args.headless,
            
            # Agent configuration
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "clip_param": agent_cfg.algorithm.clip_param,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            
            # Network architecture
            "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
            "activation": agent_cfg.policy.activation,
        }
        
        try:
            wandb.init(
                project=args.log_project_name,
                name=args.run_name,
                config=wandb_config,
                tags=["isaac-lab", "ant", "ppo", "quadruped", "locomotion", "video"],
                notes=f"Isaac Lab quadruped locomotion training with video recording. Task: {args.task}, Envs: {args.num_envs}",
                settings=wandb.Settings(start_method="fork")  # Important for multiprocessing
            )
            print(f"✅ Wandb initialized successfully")
            print(f"📊 Project: {args.log_project_name}")
            print(f"🏃 Run: {args.run_name}")
            print(f"🔗 Dashboard: {wandb.run.url}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize wandb: {e}")
            print("🔧 You may need to run: wandb login")
            return False
    
    return False

def main():
    """Main training function with enhanced video and wandb integration."""
    
    print("🚀 Starting Enhanced Isaac Lab Training")
    print("="*60)
    print(f"📊 Task: {args_cli.task}")
    print(f"🏃 Environments: {args_cli.num_envs}")
    print(f"🎥 Video Recording: {args_cli.video}")
    print(f"📈 Logger: {args_cli.logger}")
    print(f"🎯 Project: {args_cli.log_project_name}")
    print(f"🏷️  Run Name: {args_cli.run_name}")
    print("="*60)
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # Enable rendering for video recording
    if args_cli.video:
        env_cfg.sim.render_mode = "rgb_array"
        print("🎥 Enabled rendering for video recording")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env)
    
    # Parse agent configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # Override some settings for better training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations
    
    # Setup enhanced logging
    wandb_enabled = setup_enhanced_logging(args_cli, agent_cfg)
    
    # Create log directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📂 Logging to: {log_dir}")
    
    # Save configurations
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)
    dump_pickle(os.path.join(params_dir, "agent.pkl"), agent_cfg)
    
    # Print configurations
    print("\n📋 Environment Configuration:")
    print_dict(env_cfg, nesting=4)
    print("\n🤖 Agent Configuration:")
    print_dict(agent_cfg, nesting=4)
    
    # Create RSL-RL runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    print("\n🏋️ Starting Training...")
    print(f"⏱️  Max iterations: {agent_cfg.max_iterations}")
    print(f"🎥 Video interval: {args_cli.video_interval} steps")
    print(f"📏 Video length: {args_cli.video_length} frames")
    
    try:
        # Run training with all enhancements
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        
        # Export final policy
        export_policy_as_jit(
            runner.alg.actor_critic,
            path=os.path.join(log_dir, "exported")
        )
        
        print("✅ Training completed successfully!")
        
        # Log completion to wandb
        if wandb_enabled:
            wandb.log({
                "training_status": "completed",
                "final_iteration": agent_cfg.max_iterations,
                "completion_time": datetime.now().isoformat()
            })
            
            # Log final summary
            wandb.summary.update({
                "status": "completed",
                "total_iterations": agent_cfg.max_iterations,
                "task": args_cli.task,
                "num_envs": args_cli.num_envs
            })
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
        if wandb_enabled:
            wandb.log({"training_status": "interrupted"})
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if wandb_enabled:
            wandb.log({"training_status": "failed", "error": str(e)})
        raise
        
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        env.close()
        if wandb_enabled:
            print("📤 Finalizing wandb logs...")
            wandb.finish()
        simulation_app.close()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()
