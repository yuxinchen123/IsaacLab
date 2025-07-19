#!/usr/bin/env python3
"""
Generate PDF versions of Isaac Lab training results
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import glob
import os
import shutil

def create_pdf_plots():
    # Load data
    log_dir = '/home/clairec/IsaacLab/logs/rsl_rl/ant/2025-07-16_06-53-36'
    events_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not events_files:
        print("No events file found!")
        return
    
    events_file = events_files[0]
    print(f"Loading data from: {events_file}")
    
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()
    
    # Create PDF
    pdf_file = os.path.join(log_dir, 'isaac_lab_training_results.pdf')
    print(f"Creating PDF: {pdf_file}")
    
    with PdfPages(pdf_file) as pdf:
        # Create the main plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Isaac Lab Ant Training Results - PPO Algorithm', fontsize=18, fontweight='bold')
        
        # Reward progress
        if 'Train/mean_reward' in ea.Tags()['scalars']:
            reward_events = ea.Scalars('Train/mean_reward')
            steps = [e.step for e in reward_events]
            values = [e.value for e in reward_events]
            ax1.plot(steps, values, 'b-', linewidth=4, alpha=0.9)
            ax1.set_title('Episode Reward Progress', fontweight='bold', fontsize=16)
            ax1.set_xlabel('Training Step', fontsize=14)
            ax1.set_ylabel('Mean Reward', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=12)
            
            if values:
                improvement = values[-1] - values[0]
                ax1.text(0.05, 0.95, f'Improvement: +{improvement:.1f}', 
                        transform=ax1.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Episode length
        if 'Train/mean_episode_length' in ea.Tags()['scalars']:
            length_events = ea.Scalars('Train/mean_episode_length')
            steps = [e.step for e in length_events]
            values = [e.value for e in length_events]
            ax2.plot(steps, values, 'g-', linewidth=4, alpha=0.9)
            ax2.set_title('Episode Length Progress', fontweight='bold', fontsize=16)
            ax2.set_xlabel('Training Step', fontsize=14)
            ax2.set_ylabel('Mean Episode Length', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=12)
            
            if values:
                ratio = values[-1] / values[0] if values[0] > 0 else 0
                ax2.text(0.05, 0.95, f'{ratio:.1f}x longer survival', 
                        transform=ax2.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Policy loss
        if 'Loss/surrogate' in ea.Tags()['scalars']:
            loss_events = ea.Scalars('Loss/surrogate')
            steps = [e.step for e in loss_events]
            values = [e.value for e in loss_events]
            ax3.plot(steps, values, 'r-', linewidth=3, alpha=0.8)
            ax3.set_title('Policy Loss (PPO Surrogate)', fontweight='bold', fontsize=16)
            ax3.set_xlabel('Training Step', fontsize=14)
            ax3.set_ylabel('Surrogate Loss', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(labelsize=12)
        
        # Action noise
        if 'Policy/mean_noise_std' in ea.Tags()['scalars']:
            noise_events = ea.Scalars('Policy/mean_noise_std')
            steps = [e.step for e in noise_events]
            values = [e.value for e in noise_events]
            ax4.plot(steps, values, 'm-', linewidth=4, alpha=0.9)
            ax4.set_title('Action Noise (Exploration)', fontweight='bold', fontsize=16)
            ax4.set_xlabel('Training Step', fontsize=14)
            ax4.set_ylabel('Noise Standard Deviation', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=12)
            
            if values:
                reduction = (1 - values[-1]/values[0]) * 100 if values[0] > 0 else 0
                ax4.text(0.05, 0.95, f'{reduction:.1f}% noise reduction', 
                        transform=ax4.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"✅ PDF created: {pdf_file}")
    
    # Copy to main directory
    main_pdf = '/home/clairec/IsaacLab/isaac_lab_training_results.pdf'
    shutil.copy2(pdf_file, main_pdf)
    print(f"✅ PDF copied to: {main_pdf}")
    
    return pdf_file, main_pdf

if __name__ == "__main__":
    create_pdf_plots()
