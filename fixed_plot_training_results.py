#!/usr/bin/env python3
"""
Fixed plotting script for Isaac Lab RL training results.
Uses the actual metric names from TensorBoard logs.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import glob
import os
import argparse

def plot_training_results(log_dir):
    """Plot training results from TensorBoard logs."""
    
    # Find the events file
    events_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not events_files:
        print(f"No TensorBoard events file found in {log_dir}")
        return
    
    events_file = events_files[0]
    print(f"Reading from: {events_file}")
    
    # Load the data
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()
    
    # Available metrics with proper names
    metrics = {
        'Train/mean_reward': 'Episode Reward',
        'Train/mean_episode_length': 'Episode Length',
        'Policy/mean_noise_std': 'Action Noise',
        'Loss/surrogate': 'Policy Loss',
        'Loss/value_function': 'Value Loss',
        'Loss/entropy': 'Entropy Loss',
        'Loss/learning_rate': 'Learning Rate',
        'Perf/total_fps': 'Training FPS'
    }
    
    # Reward components
    reward_components = {
        'Episode_Reward/move_to_target': 'Move to Target',
        'Episode_Reward/alive': 'Alive Bonus',
        'Episode_Reward/upright': 'Stay Upright',
        'Episode_Reward/energy': 'Energy Penalty',
        'Episode_Reward/action_l2': 'Action Smoothness',
        'Episode_Reward/progress': 'Progress Reward'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Isaac Lab Ant Training Results - PPO Algorithm', fontsize=16, fontweight='bold')
    
    # Plot main training metrics
    plot_positions = [
        (0, 0, 'Train/mean_reward'),
        (0, 1, 'Train/mean_episode_length'), 
        (0, 2, 'Policy/mean_noise_std'),
        (1, 0, 'Loss/surrogate'),
        (1, 1, 'Loss/value_function'),
        (1, 2, 'Loss/entropy')
    ]
    
    for row, col, metric in plot_positions:
        ax = axes[row, col]
        if metric in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(metric)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            
            ax.plot(steps, values, linewidth=2, alpha=0.8)
            ax.set_title(metrics.get(metric, metric), fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.grid(True, alpha=0.3)
            
            # Add final value annotation
            if values:
                final_val = values[-1]
                ax.annotate(f'Final: {final_val:.3f}', 
                           xy=(steps[-1], final_val), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No data for\n{metric}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
            ax.set_title(metrics.get(metric, metric), fontweight='bold')
    
    # Plot reward components
    ax_rewards = axes[2, 0]
    reward_data = {}
    for metric, label in reward_components.items():
        if metric in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(metric)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            ax_rewards.plot(steps, values, label=label, linewidth=1.5, alpha=0.8)
            reward_data[label] = values[-1] if values else 0
    
    ax_rewards.set_title('Reward Components', fontweight='bold')
    ax_rewards.set_xlabel('Training Step')
    ax_rewards.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_rewards.grid(True, alpha=0.3)
    
    # Training performance metrics
    ax_perf = axes[2, 1]
    if 'Perf/total_fps' in ea.Tags()['scalars']:
        scalar_events = ea.Scalars('Perf/total_fps')
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        ax_perf.plot(steps, values, 'g-', linewidth=2, alpha=0.8)
        ax_perf.set_title('Training FPS', fontweight='bold')
        ax_perf.set_xlabel('Training Step')
        ax_perf.grid(True, alpha=0.3)
        if values:
            ax_perf.annotate(f'Avg: {np.mean(values):.0f} FPS', 
                           xy=(0.7, 0.9), xycoords='axes fraction',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           fontsize=10, fontweight='bold')
    
    # Summary statistics
    ax_summary = axes[2, 2]
    ax_summary.axis('off')
    
    # Collect summary stats
    summary_text = "Training Summary\n" + "="*20 + "\n\n"
    
    if 'Train/mean_reward' in ea.Tags()['scalars']:
        reward_events = ea.Scalars('Train/mean_reward')
        if reward_events:
            initial_reward = reward_events[0].value
            final_reward = reward_events[-1].value
            improvement = final_reward - initial_reward
            summary_text += f"Reward Improvement:\n"
            summary_text += f"  Initial: {initial_reward:.3f}\n"
            summary_text += f"  Final: {final_reward:.3f}\n"
            summary_text += f"  Gain: +{improvement:.3f}\n\n"
    
    if 'Train/mean_episode_length' in ea.Tags()['scalars']:
        length_events = ea.Scalars('Train/mean_episode_length')
        if length_events:
            initial_length = length_events[0].value
            final_length = length_events[-1].value
            improvement_ratio = final_length / initial_length if initial_length > 0 else 0
            summary_text += f"Episode Length:\n"
            summary_text += f"  Initial: {initial_length:.1f}\n"
            summary_text += f"  Final: {final_length:.1f}\n"
            summary_text += f"  Ratio: {improvement_ratio:.1f}x\n\n"
    
    # Add reward component final values
    if reward_data:
        summary_text += "Final Reward Components:\n"
        for component, value in sorted(reward_data.items(), key=lambda x: x[1], reverse=True):
            summary_text += f"  {component}: {value:.3f}\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(log_dir, 'training_results_fixed.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save a high-level summary plot
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Isaac Lab Ant Training - Key Metrics Overview', fontsize=16, fontweight='bold')
    
    # Main reward curve
    if 'Train/mean_reward' in ea.Tags()['scalars']:
        reward_events = ea.Scalars('Train/mean_reward')
        steps = [e.step for e in reward_events]
        values = [e.value for e in reward_events]
        ax1.plot(steps, values, 'b-', linewidth=3, alpha=0.8)
        ax1.set_title('Episode Reward Progress', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(values) > 10:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            ax1.plot(steps, p(steps), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2e}/step')
            ax1.legend()
    
    # Episode length
    if 'Train/mean_episode_length' in ea.Tags()['scalars']:
        length_events = ea.Scalars('Train/mean_episode_length')
        steps = [e.step for e in length_events]
        values = [e.value for e in length_events]
        ax2.plot(steps, values, 'g-', linewidth=3, alpha=0.8)
        ax2.set_title('Episode Length Progress', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Mean Episode Length')
        ax2.grid(True, alpha=0.3)
    
    # Policy loss
    if 'Loss/surrogate' in ea.Tags()['scalars']:
        loss_events = ea.Scalars('Loss/surrogate')
        steps = [e.step for e in loss_events]
        values = [e.value for e in loss_events]
        ax3.plot(steps, values, 'r-', linewidth=2, alpha=0.8)
        ax3.set_title('Policy Loss', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Surrogate Loss')
        ax3.grid(True, alpha=0.3)
    
    # Action noise (exploration)
    if 'Policy/mean_noise_std' in ea.Tags()['scalars']:
        noise_events = ea.Scalars('Policy/mean_noise_std')
        steps = [e.step for e in noise_events]
        values = [e.value for e in noise_events]
        ax4.plot(steps, values, 'm-', linewidth=2, alpha=0.8)
        ax4.set_title('Action Noise (Exploration)', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Mean Noise Std')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_output = os.path.join(log_dir, 'training_summary_fixed.png')
    plt.savefig(summary_output, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to: {summary_output}")
    
    print("\n" + "="*60)
    print("TRAINING RESULTS ANALYSIS")
    print("="*60)
    
    if 'Train/mean_reward' in ea.Tags()['scalars']:
        reward_events = ea.Scalars('Train/mean_reward')
        print(f"✓ Total training steps: {len(reward_events)}")
        print(f"✓ Final reward: {reward_events[-1].value:.3f}")
        print(f"✓ Initial reward: {reward_events[0].value:.3f}")
        print(f"✓ Improvement: +{reward_events[-1].value - reward_events[0].value:.3f}")
    
    print(f"✓ Plots successfully generated with actual TensorBoard data!")
    print(f"✓ Fixed metric name mapping resolved visualization issues")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Isaac Lab training results')
    parser.add_argument('--log_dir', type=str, 
                       default='/home/clairec/IsaacLab/logs/rsl_rl/ant/2025-07-16_06-53-36',
                       help='Path to the training log directory')
    
    args = parser.parse_args()
    plot_training_results(args.log_dir)
