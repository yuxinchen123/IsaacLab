#!/usr/bin/env python3
"""
Plot Isaac Lab RL Training Results
This script visualizes the training progress from TensorBoard logs
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os
import glob

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard logs"""
    # Find the events file
    events_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not events_files:
        print(f"No TensorBoard events files found in {log_dir}")
        return None
    
    events_file = events_files[0]  # Use the first (and likely only) events file
    print(f"Loading data from: {events_file}")
    
    # Load the event accumulator
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    print("Available scalar tags:", tags['scalars'])
    
    data = {}
    for tag in tags['scalars']:
        try:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': steps, 'values': values}
        except Exception as e:
            print(f"Error loading tag {tag}: {e}")
    
    return data

def plot_training_metrics(data, save_dir):
    """Create comprehensive training plots"""
    if not data:
        print("No data to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # Define plot layout
    plots = [
        ('Episode/Mean Reward', 'Episode Mean Reward', 'Training Steps', 'Reward'),
        ('Episode/Mean Episode Length', 'Episode Mean Length', 'Training Steps', 'Steps'),
        ('Train/Mean Value Loss', 'Value Function Loss', 'Training Steps', 'Loss'),
        ('Train/Mean Surrogate Loss', 'Policy (Surrogate) Loss', 'Training Steps', 'Loss'),
        ('Train/Mean Entropy Loss', 'Entropy Loss', 'Training Steps', 'Loss'),
        ('Train/Mean Action Noise Std', 'Action Noise Standard Deviation', 'Training Steps', 'Std Dev'),
    ]
    
    # Create reward component plots
    reward_components = [
        'Episode_Reward/progress',
        'Episode_Reward/alive', 
        'Episode_Reward/upright',
        'Episode_Reward/move_to_target',
        'Episode_Reward/action_l2',
        'Episode_Reward/energy',
        'Episode_Reward/joint_pos_limits'
    ]
    
    # Plot main metrics
    for i, (tag, title, xlabel, ylabel) in enumerate(plots[:6]):
        ax = plt.subplot(3, 3, i+1)
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']
            plt.plot(steps, values, 'b-', linewidth=2, alpha=0.8)
            plt.title(title, fontsize=12, fontweight='bold')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            
            # Add trend line for main metrics
            if len(steps) > 10:
                z = np.polyfit(steps, values, 1)
                p = np.poly1d(z)
                plt.plot(steps, p(steps), "r--", alpha=0.7, linewidth=1)
        else:
            plt.text(0.5, 0.5, f'No data for\n{tag}', ha='center', va='center', transform=ax.transAxes)
            plt.title(title, fontsize=12, fontweight='bold')
    
    # Plot reward components
    ax = plt.subplot(3, 3, 7)
    reward_data_found = False
    for component in reward_components:
        if component in data:
            steps = data[component]['steps']
            values = data[component]['values']
            plt.plot(steps, values, linewidth=2, alpha=0.8, label=component.split('/')[-1])
            reward_data_found = True
    
    if reward_data_found:
        plt.title('Individual Reward Components', fontsize=12, fontweight='bold')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward Value')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No reward\ncomponent data', ha='center', va='center', transform=ax.transAxes)
        plt.title('Individual Reward Components', fontsize=12, fontweight='bold')
    
    # Plot termination reasons
    ax = plt.subplot(3, 3, 8)
    termination_components = [
        'Episode_Termination/time_out',
        'Episode_Termination/torso_height'
    ]
    
    term_data_found = False
    for component in termination_components:
        if component in data:
            steps = data[component]['steps']
            values = data[component]['values']
            plt.plot(steps, values, linewidth=2, alpha=0.8, label=component.split('/')[-1])
            term_data_found = True
    
    if term_data_found:
        plt.title('Termination Reasons', fontsize=12, fontweight='bold')
        plt.xlabel('Training Steps')
        plt.ylabel('Percentage')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No termination\ndata', ha='center', va='center', transform=ax.transAxes)
        plt.title('Termination Reasons', fontsize=12, fontweight='bold')
    
    # Summary statistics
    ax = plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate some key statistics
    stats_text = "Training Summary:\n\n"
    
    if 'Episode/Mean Reward' in data:
        rewards = data['Episode/Mean Reward']['values']
        if rewards:
            stats_text += f"• Initial Reward: {rewards[0]:.3f}\n"
            stats_text += f"• Final Reward: {rewards[-1]:.3f}\n"
            stats_text += f"• Improvement: {rewards[-1] - rewards[0]:.3f}\n\n"
    
    if 'Episode/Mean Episode Length' in data:
        lengths = data['Episode/Mean Episode Length']['values']
        if lengths:
            stats_text += f"• Initial Episode Length: {lengths[0]:.1f}\n"
            stats_text += f"• Final Episode Length: {lengths[-1]:.1f}\n"
            stats_text += f"• Length Improvement: {lengths[-1] / lengths[0]:.1f}x\n\n"
    
    total_steps = 0
    if any(data.values()):
        total_steps = max([max(d['steps']) for d in data.values() if d['steps']])
        stats_text += f"• Total Training Steps: {total_steps:,}\n"
    
    plt.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path

def main():
    # Set up paths
    log_dir = "/home/clairec/IsaacLab/logs/rsl_rl/ant/2025-07-16_06-53-36"
    save_dir = "/home/clairec/IsaacLab"
    
    print("Isaac Lab RL Training Results Plotter")
    print("=" * 50)
    
    # Load data
    print(f"Loading training data from: {log_dir}")
    data = load_tensorboard_data(log_dir)
    
    if data:
        print(f"Successfully loaded {len(data)} metrics")
        plot_path = plot_training_metrics(data, save_dir)
        print(f"\nTraining visualization complete!")
        print(f"Plots saved to: {plot_path}")
    else:
        print("Failed to load training data")

if __name__ == "__main__":
    main()
