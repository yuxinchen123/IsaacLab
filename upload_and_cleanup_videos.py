#!/usr/bin/env python3
"""
Auto-upload and cleanup script for Isaac Lab videos.
Watches the video directory and uploads new videos to wandb, then deletes them.
"""

import os
import time
import argparse
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not available - install with: pip install wandb")

class VideoUploadHandler:
    """Handle new video files by uploading to wandb and deleting."""
    
    def __init__(self, delete_after_upload=True):
        self.delete_after_upload = delete_after_upload
        self.uploaded_files = set()
        self.last_check = time.time()
    
    def check_for_new_videos(self, video_dir):
        """Check for new video files and process them."""
        if not os.path.exists(video_dir):
            return
        
        for filename in os.listdir(video_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(video_dir, filename)
                
                # Check if file is new and complete
                if (file_path not in self.uploaded_files and 
                    os.path.getmtime(file_path) > self.last_check - 10):
                    
                    # Wait for file to be fully written
                    time.sleep(2)
                    self.upload_and_cleanup(file_path)
        
        self.last_check = time.time()
    
    def upload_and_cleanup(self, video_path):
        """Upload video to wandb and optionally delete."""
        if not WANDB_AVAILABLE or not wandb.run:
            print(f"⚠️ Wandb not available, skipping upload of {os.path.basename(video_path)}")
            return
        
        try:
            # Extract step number from filename
            filename = os.path.basename(video_path)
            step = self.extract_step_from_filename(filename)
            
            # Upload to wandb
            wandb.log({
                "training_video": wandb.Video(video_path, caption=f"Training Step {step}"),
                "video_step": step
            })
            
            print(f"✅ Uploaded {filename} to wandb")
            self.uploaded_files.add(video_path)
            
            # Delete local file if requested
            if self.delete_after_upload:
                os.remove(video_path)
                print(f"🗑️ Deleted local file: {filename}")
                
        except Exception as e:
            print(f"❌ Failed to upload/delete {os.path.basename(video_path)}: {e}")
    
    def extract_step_from_filename(self, filename):
        """Extract step number from video filename."""
        # Example: rl-video-step-1000.mp4 -> 1000
        try:
            parts = filename.split('-')
            for i, part in enumerate(parts):
                if part == 'step' and i + 1 < len(parts):
                    return int(parts[i + 1].split('.')[0])
        except:
            pass
        return 0

def find_video_directory(watch_dir):
    """Find the video directory in the training logs."""
    for root, dirs, files in os.walk(watch_dir):
        if 'videos' in dirs:
            video_dir = os.path.join(root, 'videos', 'train')
            if os.path.exists(video_dir):
                return video_dir
    return None

def main():
    parser = argparse.ArgumentParser(description="Auto-upload and cleanup Isaac Lab videos")
    parser.add_argument("--watch_dir", type=str, default="logs/rsl_rl/ant/", 
                       help="Directory to watch for videos")
    parser.add_argument("--delete_after_upload", action="store_true", 
                       help="Delete videos after upload")
    parser.add_argument("--wandb_project", type=str, default="isaac-ant-locomotion", 
                       help="Wandb project")
    parser.add_argument("--check_interval", type=int, default=30, 
                       help="Check for new videos every N seconds")
    
    args = parser.parse_args()
    
    # Initialize wandb if not already done
    if WANDB_AVAILABLE and not wandb.run:
        try:
            wandb.init(project=args.wandb_project, job_type="video_uploader")
            print("✅ Wandb initialized for video uploading")
        except Exception as e:
            print(f"❌ Failed to initialize wandb: {e}")
            return
    
    # Find video directory
    video_dir = find_video_directory(args.watch_dir)
    
    if not video_dir:
        print(f"❌ No video directory found in {args.watch_dir}")
        print("   Make sure training is running with --video flag")
        return
    
    print(f"👀 Watching for videos in: {video_dir}")
    print(f"🗑️ Delete after upload: {args.delete_after_upload}")
    print(f"⏰ Check interval: {args.check_interval} seconds")
    
    # Setup video handler
    handler = VideoUploadHandler(delete_after_upload=args.delete_after_upload)
    
    try:
        print("🚀 Started watching for videos... Press Ctrl+C to stop")
        while True:
            handler.check_for_new_videos(video_dir)
            time.sleep(args.check_interval)
            
    except KeyboardInterrupt:
        print("⏹️ Stopped watching")
    
    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()
