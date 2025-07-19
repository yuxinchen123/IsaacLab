#!/usr/bin/env python3
"""
Enhanced Isaac Lab training script with video recording and wandb integration.
Based on the original RSL-RL training script with added video and wandb features.
"""

import argparse
import os
import sys
from datetime import datetime

# Add Isaac Lab to path
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Isaac Lab")
parser.add_argument("--video", action="store_true", help="Record training videos")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos")
parser.add_argument("--video_interval", type=int, default=100, help="Interval between video recordings")
parser.add_argument("--wandb_project", type=str, default="isaac-lab-ant", help="Wandb project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Environment task")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--enable_cameras", action="store_true", help="Enable cameras for video recording")

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import wandb
from datetime import datetime

import isaaclab  # noqa: F401
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import RL modules
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

def main():
    """Main training function with video recording and wandb integration."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # Modify environment config for video recording if enabled
    if args_cli.video and args_cli.enable_cameras:
        # Enable cameras in the environment
        if hasattr(env_cfg, 'viewer'):
            env_cfg.viewer.record_video = True
            env_cfg.viewer.video_length = args_cli.video_length
        
        # Enable off-screen rendering for headless video recording
        if hasattr(env_cfg.sim, 'render_mode'):
            env_cfg.sim.render_mode = "rgb_array"
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Initialize wandb
    if args_cli.wandb_project:
        wandb_run_name = args_cli.wandb_run_name or f"ant-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=args_cli.wandb_project,
            name=wandb_run_name,
            config={
                "task": args_cli.task,
                "num_envs": args_cli.num_envs,
                "algorithm": "PPO",
                "framework": "RSL-RL",
                "video_recording": args_cli.video,
                "video_length": args_cli.video_length,
                "video_interval": args_cli.video_interval,
                "headless": args_cli.headless,
                "device": "cuda:0"
            },
            tags=["isaac-lab", "ppo", "ant", "quadruped"]
        )
        print(f"✅ Wandb initialized: {wandb_run_name}")
    
    # Load training configuration
    from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
    agent_cfg: RslRlOnPolicyRunnerCfg = AntPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    
    # Create RSL-RL runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Setup logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, "params", "agent.pkl"), agent_cfg)
    
    # Print configuration
    print("[INFO] Environment configuration:")
    print_dict(env_cfg, nesting=4)
    print("[INFO] Agent configuration:")
    print_dict(agent_cfg, nesting=4)
    
    # Custom training loop with video recording
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
        eval_freq=250,  # Evaluate every 250 iterations
    )
    
    # Save final policy
    export_policy_as_jit(
        runner.alg.actor_critic, 
        path=os.path.join(log_root_path, "exported")
    )
    
    # Close environment and wandb
    env.close()
    if args_cli.wandb_project:
        wandb.finish()
    
    # Close simulation
    simulation_app.close()

if __name__ == "__main__":
    main()
