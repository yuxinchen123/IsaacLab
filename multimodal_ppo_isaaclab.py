#!/usr/bin/env python3
"""
Multimodal PPO Training for Isaac Lab Manipulation Tasks
Combines RGB vision, tactile force grids, and proprioception inputs for end-to-end RL training
Each agent runs in its own environment with separate WandB logging and video recording
"""

import os
import sys
import time
import yaml
import torch
import wandb
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import nullcontext

# Isaac Lab imports
import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher

# Add Isaac Lab tasks to path
sys.path.append("/home/clairec/IsaacLab/source/isaaclab_tasks")

# RSL-RL imports
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

# PyTorch imports
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# Isaac Lab imports
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

class MultimodalObservationEncoder(nn.Module):
    """
    Multimodal encoder for RGB, tactile, and proprioception inputs
    - RGB: CNN encoder (64x64x3) -> embedding
    - Tactile: CNN encoder (6x32x32) -> embedding  
    - Proprioception: MLP with LayerNorm -> embedding
    - Fusion: Concatenate + MLP -> final embedding
    """
    
    def __init__(self, 
                 rgb_shape: Tuple[int, int, int] = (3, 64, 64),
                 tactile_shape: Tuple[int, int, int] = (6, 32, 32),
                 proprio_dim: int = 32,
                 rgb_embedding_dim: int = 256,
                 tactile_embedding_dim: int = 128,
                 proprio_embedding_dim: int = 64,
                 fusion_hidden_dim: int = 512,
                 output_dim: int = 256):
        super().__init__()
        
        self.rgb_shape = rgb_shape
        self.tactile_shape = tactile_shape
        self.proprio_dim = proprio_dim
        
        # RGB CNN Encoder
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(rgb_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, rgb_embedding_dim),
            nn.LayerNorm(rgb_embedding_dim),
            nn.ReLU()
        )
        
        # Tactile CNN Encoder
        self.tactile_encoder = nn.Sequential(
            nn.Conv2d(tactile_shape[0], 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, tactile_embedding_dim),
            nn.LayerNorm(tactile_embedding_dim),
            nn.ReLU()
        )
        
        # Proprioception MLP Encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, proprio_embedding_dim),
            nn.LayerNorm(proprio_embedding_dim),
            nn.ReLU()
        )
        
        # Fusion Network
        total_embedding_dim = rgb_embedding_dim + tactile_embedding_dim + proprio_embedding_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(total_embedding_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal encoder
        
        Args:
            observations: Dict containing 'rgb', 'tactile', 'proprioception' tensors
            
        Returns:
            fused_embedding: Final multimodal embedding
        """
        # Extract individual modalities
        rgb = observations['rgb']
        tactile = observations['tactile'] 
        proprio = observations['proprioception']
        
        # Encode each modality
        rgb_embedding = self.rgb_encoder(rgb)
        tactile_embedding = self.tactile_encoder(tactile)
        proprio_embedding = self.proprio_encoder(proprio)
        
        # Fuse embeddings
        combined = torch.cat([rgb_embedding, tactile_embedding, proprio_embedding], dim=-1)
        fused_embedding = self.fusion_network(combined)
        
        return fused_embedding


class MultimodalActorCritic(ActorCritic):
    """Custom Actor-Critic with multimodal observation encoder"""
    
    def __init__(self, 
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 obs_encoder_cfg: Dict[str, Any],
                 **kwargs):
        
        # Initialize encoder
        self.obs_encoder = MultimodalObservationEncoder(**obs_encoder_cfg)
        
        # Update observation dimensions to use encoder output
        encoded_obs_dim = obs_encoder_cfg.get('output_dim', 256)
        
        super().__init__(
            num_actor_obs=encoded_obs_dim,
            num_critic_obs=encoded_obs_dim, 
            num_actions=num_actions,
            **kwargs
        )
        
    def forward(self, observations, masks=None, hidden_states=None):
        """Forward pass with multimodal encoding"""
        # Encode multimodal observations
        if isinstance(observations, dict):
            encoded_obs = self.obs_encoder(observations)
        else:
            encoded_obs = observations
            
        # Standard ActorCritic forward pass
        return super().forward(encoded_obs, masks, hidden_states)


class MultimodalLiftEnvWrapper:
    """
    Wrapper to add multimodal observations to Isaac Lab Lift environment
    Extends basic lift task with RGB camera and simulated tactile sensing
    """
    
    def __init__(self, base_env_cfg, enable_cameras: bool = True):
        self.base_cfg = base_env_cfg
        self.enable_cameras = enable_cameras
        
        # Add camera configuration
        if enable_cameras:
            self._add_camera_config()
            
        # Add multimodal observations
        self._add_multimodal_observations()
        
    def _add_camera_config(self):
        """Add RGB camera to the environment"""
        # Camera configuration similar to ShadowHandVision
        camera_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.3, -0.3, 0.8), 
                rot=(0.7071, 0.0, 0.7071, 0.0), 
                convention="world"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0)
            ),
            width=64,
            height=64,
        )
        
        # Add to scene
        if not hasattr(self.base_cfg.scene, 'tiled_camera'):
            self.base_cfg.scene.tiled_camera = camera_cfg
            
    def _add_multimodal_observations(self):
        """Add RGB, tactile, and enhanced proprioception observations"""
        
        @configclass
        class MultimodalObservationsCfg(self.base_cfg.observations.__class__):
            """Enhanced observations with multimodal data"""
            
            @configclass
            class PolicyCfg(self.base_cfg.observations.PolicyCfg.__class__):
                """Policy observations with multimodal inputs"""
                
                # RGB camera observations
                rgb_image = ObsTerm(
                    func=self._get_rgb_observation,
                    params={"sensor_cfg": {"name": "tiled_camera", "data_type": "rgb"}}
                )
                
                # Simulated tactile observations (fingertip forces)
                tactile_forces = ObsTerm(
                    func=self._get_tactile_observation,
                    params={"num_fingers": 6, "grid_size": 32}
                )
                
                # Enhanced proprioception
                joint_positions_norm = ObsTerm(func=self._get_normalized_joint_pos)
                joint_velocities_norm = ObsTerm(func=self._get_normalized_joint_vel)
                end_effector_pose = ObsTerm(func=self._get_ee_pose)
                
                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = False  # Keep separate for multimodal processing
                    
            policy: PolicyCfg = PolicyCfg()
            
        # Replace observations
        self.base_cfg.observations = MultimodalObservationsCfg()
        
    @staticmethod
    def _get_rgb_observation(env, sensor_cfg: Dict[str, str]) -> torch.Tensor:
        """Get RGB camera observations"""
        if hasattr(env.scene, 'tiled_camera'):
            camera_data = env.scene.tiled_camera.data
            if 'rgb' in camera_data:
                rgb_data = camera_data['rgb'].clone()
                # Normalize to [0, 1] and rearrange to (B, C, H, W)
                return rgb_data.float() / 255.0
        
        # Fallback: return zero tensor
        return torch.zeros((env.num_envs, 3, 64, 64), device=env.device)
        
    @staticmethod
    def _get_tactile_observation(env, num_fingers: int, grid_size: int) -> torch.Tensor:
        """Simulate tactile force grid observations"""
        # Simple simulation: use contact forces at fingertips
        robot_contact_forces = env.scene.robot.data.net_contact_forces
        
        # Create 6-channel tactile grid (one per finger)
        tactile_grid = torch.zeros((env.num_envs, num_fingers, grid_size, grid_size), device=env.device)
        
        # Simulate force distribution on fingertip grids
        for finger_idx in range(num_fingers):
            if finger_idx < robot_contact_forces.shape[1]:
                force_magnitude = torch.norm(robot_contact_forces[:, finger_idx], dim=-1)
                # Create simple circular pattern centered on grid
                center = grid_size // 2
                for i in range(grid_size):
                    for j in range(grid_size):
                        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                        weight = torch.exp(-dist / 5.0)  # Gaussian falloff
                        tactile_grid[:, finger_idx, i, j] = force_magnitude * weight
                        
        return tactile_grid
        
    @staticmethod  
    def _get_normalized_joint_pos(env) -> torch.Tensor:
        """Get normalized joint positions"""
        joint_pos = env.scene.robot.data.joint_pos
        joint_limits = env.scene.robot.data.default_joint_pos
        
        # Normalize joint angles to [-1, 1]
        normalized = 2.0 * (joint_pos - joint_limits) / (joint_limits.abs() + 1e-6)
        return torch.clamp(normalized, -1.0, 1.0)
        
    @staticmethod
    def _get_normalized_joint_vel(env) -> torch.Tensor:
        """Get normalized joint velocities"""
        joint_vel = env.scene.robot.data.joint_vel
        max_vel = 2.0  # Assume max velocity
        return torch.clamp(joint_vel / max_vel, -1.0, 1.0)
        
    @staticmethod
    def _get_ee_pose(env) -> torch.Tensor:
        """Get end-effector pose (position + orientation)"""
        ee_pos = env.scene.ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = env.scene.ee_frame.data.target_quat_w[..., 0, :]
        return torch.cat([ee_pos, ee_quat], dim=-1)


class MultiAgentPPOTrainer:
    """
    Multi-agent PPO trainer with individual environments and WandB logging
    Each agent runs in isolated environment with separate logging
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.num_agents = cfg['num_agents']
        self.agents = []
        self.envs = []
        self.loggers = []
        
        # Video recording setup
        self.video_record_freq = cfg.get('video_record_freq', 5000)
        self.video_length = cfg.get('video_length', 200)
        
        # Initialize agents
        self._setup_agents()
        
    def _setup_agents(self):
        """Setup individual agents with separate environments"""
        
        for agent_id in range(self.num_agents):
            # Create individual environment
            env_cfg = self._create_agent_env_config(agent_id)
            env = self._create_environment(env_cfg, agent_id)
            
            # Create individual WandB logger
            logger = self._setup_wandb_logger(agent_id)
            
            # Create PPO agent
            agent = self._create_ppo_agent(env, agent_id)
            
            self.envs.append(env)
            self.loggers.append(logger)
            self.agents.append(agent)
            
    def _create_agent_env_config(self, agent_id: int):
        """Create environment configuration for specific agent"""
        from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
        
        # Base configuration
        env_cfg = FrankaCubeLiftEnvCfg()
        
        # Customize for agent
        env_cfg.scene.num_envs = self.cfg['envs_per_agent']
        env_cfg.scene.env_spacing = 2.5
        
        # Apply multimodal wrapper
        wrapper = MultimodalLiftEnvWrapper(env_cfg, enable_cameras=True)
        
        return wrapper.base_cfg
        
    def _create_environment(self, env_cfg, agent_id: int):
        """Create Isaac Lab environment"""
        # Launch Isaac Sim application if first agent
        if agent_id == 0:
            launcher_cfg = {
                "headless": self.cfg.get('headless', True),
                "enable_cameras": True,
                "livestream": 0
            }
            app_launcher = AppLauncher(launcher_cfg)
            simulation_app = app_launcher.app
            
        # Create environment
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        return env
        
    def _create_ppo_agent(self, env, agent_id: int):
        """Create PPO agent for specific environment"""
        
        # Multimodal encoder configuration
        encoder_cfg = {
            'rgb_shape': (3, 64, 64),
            'tactile_shape': (6, 32, 32), 
            'proprio_dim': 32,
            'output_dim': 256
        }
        
        # Actor-Critic configuration
        policy_cfg = {
            'obs_encoder_cfg': encoder_cfg,
            'activation': 'elu',
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            'init_noise_std': 1.0
        }
        
        # Create multimodal actor-critic
        actor_critic = MultimodalActorCritic(
            num_actor_obs=env.observation_space.shape[0],
            num_critic_obs=env.observation_space.shape[0],
            num_actions=env.action_space.shape[0],
            **policy_cfg
        ).to(self.device)
        
        # PPO algorithm configuration
        ppo_cfg = self.cfg['ppo']
        alg_cfg = {
            'value_loss_coef': ppo_cfg.get('value_loss_coef', 1.0),
            'use_clipped_value_loss': ppo_cfg.get('use_clipped_value_loss', True),
            'clip_param': ppo_cfg.get('clip_param', 0.2),
            'entropy_coef': ppo_cfg.get('entropy_coef', 0.01),
            'num_learning_epochs': ppo_cfg.get('num_learning_epochs', 5),
            'num_mini_batches': ppo_cfg.get('num_mini_batches', 4),
            'learning_rate': ppo_cfg.get('learning_rate', 1e-4),
            'schedule': ppo_cfg.get('schedule', 'adaptive'),
            'gamma': ppo_cfg.get('gamma', 0.99),
            'lam': ppo_cfg.get('lam', 0.95),
            'desired_kl': ppo_cfg.get('desired_kl', 0.01),
            'max_grad_norm': ppo_cfg.get('max_grad_norm', 1.0)
        }
        
        # Create PPO algorithm
        ppo_algorithm = PPO(actor_critic, device=self.device, **alg_cfg)
        
        # Create runner
        runner = OnPolicyRunner(env, ppo_algorithm, num_steps_per_env=self.cfg['steps_per_rollout'])
        
        return runner
        
    def _setup_wandb_logger(self, agent_id: int):
        """Setup individual WandB logger for agent"""
        
        if not self.cfg.get('enable_wandb', True):
            return None
            
        # Individual run configuration
        run_config = {
            'project': self.cfg['wandb']['project'],
            'entity': self.cfg['wandb'].get('entity'),
            'name': f"{self.cfg['wandb']['run_name']}_agent_{agent_id}",
            'group': self.cfg['wandb']['run_name'],
            'tags': self.cfg['wandb'].get('tags', []) + [f'agent_{agent_id}'],
            'config': {
                **self.cfg,
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Initialize WandB run
        run = wandb.init(**run_config)
        
        return run
        
    def train(self):
        """Execute multi-agent training"""
        
        max_iterations = self.cfg['max_iterations']
        save_freq = self.cfg.get('save_model_freq', 1000)
        
        print(f"Starting multi-agent training with {self.num_agents} agents...")
        
        for iteration in range(max_iterations):
            
            # Train each agent
            for agent_id, (agent, env, logger) in enumerate(zip(self.agents, self.envs, self.loggers)):
                
                print(f"Iteration {iteration}, Agent {agent_id}")
                
                # Run training step
                agent.learn(num_learning_iterations=1, init_at_random_ep_len=True)
                
                # Log metrics to WandB
                if logger is not None:
                    metrics = self._collect_metrics(agent, env, agent_id)
                    logger.log(metrics, step=iteration)
                    
                    # Record video periodically
                    if iteration % self.video_record_freq == 0:
                        self._record_and_log_video(env, logger, agent_id, iteration)
                        
                # Save model periodically
                if iteration % save_freq == 0:
                    self._save_agent_model(agent, agent_id, iteration)
                    
        print("Training completed!")
        
        # Final cleanup
        for logger in self.loggers:
            if logger is not None:
                logger.finish()
                
    def _collect_metrics(self, agent, env, agent_id: int) -> Dict[str, float]:
        """Collect training metrics for logging"""
        
        metrics = {
            'agent_id': agent_id,
            'episode_length': float(env.episode_length_buf.mean().cpu()),
            'episode_reward': float(env.episode_sums["reward"].mean().cpu()),
            'learning_rate': agent.optimizer.param_groups[0]['lr'],
        }
        
        # Add PPO-specific metrics
        if hasattr(agent, 'storage'):
            if hasattr(agent.storage, 'rewards'):
                metrics['rollout_reward'] = float(agent.storage.rewards.mean().cpu())
            if hasattr(agent.storage, 'values'):
                metrics['rollout_value'] = float(agent.storage.values.mean().cpu())
                
        return metrics
        
    def _record_and_log_video(self, env, logger, agent_id: int, iteration: int):
        """Record environment video and log to WandB"""
        
        if logger is None:
            return
            
        try:
            # Set environment to recording mode
            env.cfg.render_mode = "rgb_array"
            
            # Collect video frames
            frames = []
            obs = env.reset()
            
            for step in range(self.video_length):
                # Random action for demo (replace with policy action in practice)
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                # Capture frame
                if hasattr(env, 'render'):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                        
                if done.any():
                    break
                    
            # Log video to WandB
            if len(frames) > 0:
                video = np.stack(frames, axis=0)
                logger.log({
                    f"agent_{agent_id}/rollout_video": wandb.Video(
                        video, fps=30, format="mp4"
                    )
                }, step=iteration)
                
                print(f"Logged video for Agent {agent_id} at iteration {iteration}")
                
        except Exception as e:
            print(f"Failed to record video for Agent {agent_id}: {e}")
            
    def _save_agent_model(self, agent, agent_id: int, iteration: int):
        """Save agent model checkpoint"""
        
        save_dir = Path(self.cfg['save_dir']) / f"agent_{agent_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_dir / f"model_iteration_{iteration}.pt"
        
        torch.save({
            'iteration': iteration,
            'agent_id': agent_id,
            'actor_critic_state_dict': agent.alg.actor_critic.state_dict(),
            'optimizer_state_dict': agent.alg.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Saved model for Agent {agent_id} at {checkpoint_path}")


def create_training_config() -> Dict[str, Any]:
    """Create default training configuration"""
    
    return {
        # Environment settings
        'num_agents': 8,
        'envs_per_agent': 512,  # 8 * 512 = 4096 total environments
        'device': 'cuda:0',
        'headless': True,
        
        # Training settings
        'max_iterations': 10000,
        'steps_per_rollout': 24,
        'save_model_freq': 1000,
        
        # PPO hyperparameters
        'ppo': {
            'value_loss_coef': 1.0,
            'use_clipped_value_loss': True,
            'clip_param': 0.2,
            'entropy_coef': 0.01,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'learning_rate': 1e-4,
            'schedule': 'adaptive',
            'gamma': 0.99,
            'lam': 0.95,
            'desired_kl': 0.01,
            'max_grad_norm': 1.0
        },
        
        # Logging and video
        'enable_wandb': True,
        'video_record_freq': 5000,
        'video_length': 200,
        
        # WandB configuration
        'wandb': {
            'project': 'isaac-lab-multimodal-ppo',
            'entity': None,  # Set to your WandB entity
            'run_name': f'multimodal_lift_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'tags': ['isaac-lab', 'multimodal', 'ppo', 'manipulation']
        },
        
        # Save directory
        'save_dir': f'/home/clairec/IsaacLab/outputs/multimodal_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }


def create_sweep_config() -> Dict[str, Any]:
    """Create WandB sweep configuration for hyperparameter optimization"""
    
    return {
        'method': 'bayes',
        'metric': {
            'name': 'episode_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'clip_param': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'entropy_coef': {
                'distribution': 'log_uniform_values', 
                'min': 1e-4,
                'max': 1e-1
            },
            'value_loss_coef': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 2.0
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.98,
                'max': 0.999
            },
            'num_learning_epochs': {
                'values': [3, 5, 8, 10]
            },
            'num_mini_batches': {
                'values': [2, 4, 8]
            }
        }
    }


def main():
    """Main training entry point"""
    
    parser = argparse.ArgumentParser(description="Isaac Lab Multimodal PPO Training")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--sweep", action="store_true", help="Run as WandB sweep")
    parser.add_argument("--agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--envs-per-agent", type=int, default=512, help="Environments per agent")
    parser.add_argument("--max-iter", type=int, default=10000, help="Maximum training iterations")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = create_training_config()
        
    # Override with command line arguments
    if args.agents:
        cfg['num_agents'] = args.agents
    if args.envs_per_agent:
        cfg['envs_per_agent'] = args.envs_per_agent
    if args.max_iter:
        cfg['max_iterations'] = args.max_iter
    if args.no_wandb:
        cfg['enable_wandb'] = False
    if args.headless:
        cfg['headless'] = True
        
    # Handle sweep mode
    if args.sweep:
        def train_with_sweep():
            # Initialize sweep run
            with wandb.init() as run:
                # Update config with sweep parameters
                for key, value in run.config.items():
                    if key in cfg['ppo']:
                        cfg['ppo'][key] = value
                        
                # Run training
                trainer = MultiAgentPPOTrainer(cfg)
                trainer.train()
                
        # Create and run sweep
        sweep_config = create_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=cfg['wandb']['project'])
        wandb.agent(sweep_id, train_with_sweep)
        
    else:
        # Standard training
        trainer = MultiAgentPPOTrainer(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
