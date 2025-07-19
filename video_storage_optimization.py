#!/usr/bin/env python3
"""
Modified Isaac Lab training to minimize local video storage and maximize wandb integration.
Three approaches: minimal local storage, direct upload, and cleanup strategies.
"""

import os
import tempfile
import shutil
from pathlib import Path

# Approach 1: Minimal Local Storage with Immediate Cleanup
def train_with_minimal_storage():
    """
    Strategy: Create videos, upload to wandb immediately, delete local files.
    This keeps only 1-2 videos locally at any time.
    """
    
    training_command = '''
# Modified training that uploads and deletes videos immediately
python scripts/reinforcement_learning/rsl_rl/train.py \\
    --task Isaac-Ant-v0 \\
    --num_envs 4096 \\
    --headless \\
    --video \\
    --video_length 200 \\
    --video_interval 100 \\
    --logger wandb \\
    --wandb_project "isaac-ant-locomotion" \\
    --wandb_name "ant-minimal-storage-$(date +%m%d-%H%M)" &

# Background process to upload and delete videos as they're created
python upload_and_cleanup_videos.py --watch_dir logs/rsl_rl/ant/ --delete_after_upload
'''
    
    return training_command

# Approach 2: Custom Video Recorder with Direct Upload
def create_direct_upload_wrapper():
    """
    Create a wrapper that intercepts video creation and uploads directly to wandb.
    """
    
    wrapper_code = '''
import os
import tempfile
import wandb
from gymnasium.wrappers import RecordVideo

class DirectWandbVideoWrapper(RecordVideo):
    """Video recorder that uploads to wandb and optionally deletes local files."""
    
    def __init__(self, env, video_folder, upload_to_wandb=True, delete_local=True, **kwargs):
        super().__init__(env, video_folder, **kwargs)
        self.upload_to_wandb = upload_to_wandb
        self.delete_local = delete_local
        self.video_count = 0
    
    def close_video_recorder(self):
        """Override to add wandb upload and cleanup."""
        if self.video_recorder:
            # Close the video normally
            super().close_video_recorder()
            
            # Get the video path
            if hasattr(self.video_recorder, 'path'):
                video_path = self.video_recorder.path
                
                # Upload to wandb if enabled
                if self.upload_to_wandb and wandb.run and os.path.exists(video_path):
                    try:
                        wandb.log({
                            "training_video": wandb.Video(video_path, caption=f"Step {self.step_id}"),
                            "video_step": self.step_id
                        })
                        print(f"✅ Uploaded {video_path} to wandb")
                        
                        # Delete local file if requested
                        if self.delete_local:
                            os.remove(video_path)
                            print(f"🗑️ Deleted local file: {video_path}")
                            
                    except Exception as e:
                        print(f"❌ Failed to upload/delete video: {e}")
'''
    
    return wrapper_code

# Approach 3: Auto-cleanup Script
def create_video_cleanup_script():
    """
    Create a script that monitors video directory and uploads/deletes videos automatically.
    """
    
    cleanup_script = '''#!/usr/bin/env python3
"""
Auto-upload and cleanup script for Isaac Lab videos.
Watches the video directory and uploads new videos to wandb, then deletes them.
"""

import os
import time
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class VideoUploadHandler(FileSystemEventHandler):
    """Handle new video files by uploading to wandb and deleting."""
    
    def __init__(self, delete_after_upload=True):
        self.delete_after_upload = delete_after_upload
        self.uploaded_files = set()
    
    def on_created(self, event):
        """Called when a new file is created."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path.endswith('.mp4') and file_path not in self.uploaded_files:
            # Wait a moment for file to be fully written
            time.sleep(2)
            self.upload_and_cleanup(file_path)
    
    def upload_and_cleanup(self, video_path):
        """Upload video to wandb and optionally delete."""
        if not WANDB_AVAILABLE or not wandb.run:
            print(f"⚠️ Wandb not available, skipping upload of {video_path}")
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
            print(f"❌ Failed to upload/delete {video_path}: {e}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Auto-upload and cleanup Isaac Lab videos")
    parser.add_argument("--watch_dir", type=str, required=True, help="Directory to watch for videos")
    parser.add_argument("--delete_after_upload", action="store_true", help="Delete videos after upload")
    parser.add_argument("--wandb_project", type=str, default="isaac-ant-locomotion", help="Wandb project")
    
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
    video_dir = None
    for root, dirs, files in os.walk(args.watch_dir):
        if 'videos' in dirs:
            video_dir = os.path.join(root, 'videos', 'train')
            break
    
    if not video_dir or not os.path.exists(video_dir):
        print(f"❌ No video directory found in {args.watch_dir}")
        return
    
    print(f"👀 Watching for videos in: {video_dir}")
    print(f"🗑️ Delete after upload: {args.delete_after_upload}")
    
    # Setup file watcher
    event_handler = VideoUploadHandler(delete_after_upload=args.delete_after_upload)
    observer = Observer()
    observer.schedule(event_handler, video_dir, recursive=False)
    observer.start()
    
    try:
        print("🚀 Started watching for videos... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("⏹️ Stopped watching")
    
    observer.join()

if __name__ == "__main__":
    main()
'''
    
    with open("upload_and_cleanup_videos.py", "w") as f:
        f.write(cleanup_script)
    
    print("✅ Created upload_and_cleanup_videos.py")

def show_storage_optimization_options():
    """Show all available options for minimizing video storage."""
    
    print("""
🎯 VIDEO STORAGE OPTIMIZATION OPTIONS

=== OPTION 1: Minimal Storage (Recommended) ===
✅ Videos created locally, uploaded immediately, then deleted
✅ Only 1-2 videos stored locally at any time
✅ Full wandb integration
✅ Easy to implement

Command:
python scripts/reinforcement_learning/rsl_rl/train.py \\
    --task Isaac-Ant-v0 \\
    --num_envs 4096 \\
    --headless \\
    --video \\
    --video_length 200 \\
    --video_interval 100 \\
    --logger wandb \\
    --wandb_project "isaac-ant-locomotion" &

# In parallel, run cleanup script:
python upload_and_cleanup_videos.py \\
    --watch_dir logs/rsl_rl/ant/ \\
    --delete_after_upload

=== OPTION 2: Temporary Directory ===
✅ Videos created in /tmp (cleared on reboot)
✅ Upload to wandb from temp location
✅ No permanent local storage

Modify training script to use:
--video_folder /tmp/isaac_videos/

=== OPTION 3: Live Streaming (Advanced) ===
✅ Videos created in memory only
✅ Direct upload to wandb without files
✅ Zero local storage
✅ Requires custom implementation

=== RECOMMENDATION ===
For immediate use: **Option 1 (Minimal Storage)**
- Keep the successful training setup you have
- Add the cleanup script to delete videos after upload
- Reduces storage from 56+ videos to 1-2 videos at any time

Storage comparison:
- Current: ~15MB (56 videos)
- Option 1: ~0.5MB (1-2 videos)
- Option 2: 0MB (temporary)
- Option 3: 0MB (memory only)

""")

if __name__ == "__main__":
    print("📹 Isaac Lab Video Storage Optimization")
    create_video_cleanup_script()
    show_storage_optimization_options()
