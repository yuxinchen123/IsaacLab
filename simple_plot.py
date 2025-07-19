#!/usr/bin/env python3
"""
Simple Isaac Lab Training Progress Plotter
Creates clean, focused plots of key training metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import glob
import os

def load_data(log_dir):
    """Load TensorBoard data"""
    events_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not events_files:
        return None
    
    ea = event_accumulator.EventAccumulator(events_files[0])
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in scalar_events],
            'values': [e.value for e in scalar_events]
        }
    return data

def create_training_plots():
    """Create focused training progress plots"""
    log_dir = "/home/clairec/IsaacLab/logs/rsl_rl/ant/2025-07-16_06-53-36"
    data = load_data(log_dir)
    
    if not data:
        print("No data found!")
        return
    
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Isaac Lab Ant Training Progress - Isaac-Ant-v0', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Reward Progress
    if 'Train/mean_reward' in data:
        steps = data['Train/mean_reward']['steps']
        rewards = data['Train/mean_reward']['values']
        ax1.plot(steps, rewards, 'b-', linewidth=2, label='Mean Reward')
        ax1.set_title('Learning Progress: Mean Reward')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if len(rewards) > 1:
            improvement = rewards[-1] - rewards[0]
            ax1.annotate(f'Improvement: {improvement:.2f}', 
                        xy=(0.7, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Plot 2: Episode Length (Survival Time)
    if 'Train/mean_episode_length' in data:
        steps = data['Train/mean_episode_length']['steps']
        lengths = data['Train/mean_episode_length']['values']
        ax2.plot(steps, lengths, 'g-', linewidth=2, label='Episode Length')
        ax2.set_title('Ant Survival: Episode Length')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Average Episode Length (steps)')
        ax2.grid(True, alpha=0.3)
        
        # Add survival improvement
        if len(lengths) > 1:
            improvement = lengths[-1] / lengths[0]
            ax2.annotate(f'Survival improved {improvement:.1f}x', 
                        xy=(0.6, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Plot 3: Key Reward Components
    reward_tags = ['Episode_Reward/progress', 'Episode_Reward/alive', 'Episode_Reward/move_to_target']
    colors = ['purple', 'orange', 'red']
    for i, (tag, color) in enumerate(zip(reward_tags, colors)):
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']
            label = tag.split('/')[-1].replace('_', ' ').title()
            ax3.plot(steps, values, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax3.set_title('Key Reward Components')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Reward Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Efficiency
    if 'Policy/mean_noise_std' in data:
        steps = data['Policy/mean_noise_std']['steps']
        noise = data['Policy/mean_noise_std']['values']
        ax4.plot(steps, noise, 'm-', linewidth=2, label='Action Noise Std')
        ax4.set_title('Learning Efficiency: Action Exploration')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Action Noise Standard Deviation')
        ax4.grid(True, alpha=0.3)
        
        # Show exploration decay
        if len(noise) > 1:
            decay = (noise[0] - noise[-1]) / noise[0] * 100
            ax4.annotate(f'Exploration reduced by {decay:.1f}%', 
                        xy=(0.5, 0.8), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save high-quality plot
    plt.savefig('/home/clairec/IsaacLab/ant_training_progress.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/clairec/IsaacLab/ant_training_progress.pdf', bbox_inches='tight')
    
    print("📊 Training progress plots saved:")
    print("   • ant_training_progress.png (high-res)")
    print("   • ant_training_progress.pdf")
    
    # Print summary statistics
    print("\n🎯 Training Summary:")
    if 'Train/mean_reward' in data:
        rewards = data['Train/mean_reward']['values']
        print(f"   • Reward: {rewards[0]:.3f} → {rewards[-1]:.3f} (Δ{rewards[-1]-rewards[0]:.3f})")
    
    if 'Train/mean_episode_length' in data:
        lengths = data['Train/mean_episode_length']['values']
        print(f"   • Episode Length: {lengths[0]:.1f} → {lengths[-1]:.1f} steps ({lengths[-1]/lengths[0]:.1f}x improvement)")
    
    if 'Policy/mean_noise_std' in data:
        noise = data['Policy/mean_noise_std']['values']
        print(f"   • Action Noise: {noise[0]:.3f} → {noise[-1]:.3f} (exploration reduced)")
    
    return '/home/clairec/IsaacLab/ant_training_progress.png'

if __name__ == "__main__":
    print("Creating Isaac Lab Ant Training Visualization...")
    plot_path = create_training_plots()
    print(f"✅ Plots complete! Check: {plot_path}")
