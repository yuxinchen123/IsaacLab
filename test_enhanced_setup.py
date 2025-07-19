#!/usr/bin/env python3
"""
Quick test script to verify video recording and wandb integration
"""

import os
import sys

# Add Isaac Lab scripts to path
sys.path.append('/home/clairec/IsaacLab/scripts/reinforcement_learning/rsl_rl')

def test_wandb_setup():
    """Test if wandb is properly configured."""
    try:
        import wandb
        print("✅ Wandb installed")
        
        # Check if logged in
        try:
            wandb.login()
            print("✅ Wandb login successful")
            return True
        except Exception as e:
            print(f"❌ Wandb login failed: {e}")
            print("🔧 Please run: wandb login")
            return False
            
    except ImportError:
        print("❌ Wandb not installed")
        return False

def test_video_dependencies():
    """Test if video recording dependencies are available."""
    try:
        import imageio
        print("✅ Imageio available for video recording")
        return True
    except ImportError:
        print("❌ Imageio not available")
        return False

def test_isaac_lab():
    """Test if Isaac Lab is properly configured."""
    try:
        import isaaclab
        print("✅ Isaac Lab available")
        return True
    except ImportError:
        print("❌ Isaac Lab not available")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Enhanced Isaac Lab Setup")
    print("="*50)
    
    wandb_ok = test_wandb_setup()
    video_ok = test_video_dependencies()
    isaac_ok = test_isaac_lab()
    
    print("="*50)
    if wandb_ok and video_ok and isaac_ok:
        print("✅ ALL TESTS PASSED - Ready for enhanced training!")
        print("\n🚀 To start training with video + wandb:")
        print("python train_enhanced_video_wandb.py")
    else:
        print("⚠️  Some components need setup:")
        if not wandb_ok:
            print("  - Run: wandb login")
        if not video_ok:
            print("  - Run: pip install imageio imageio-ffmpeg")
        if not isaac_ok:
            print("  - Activate Isaac Lab environment")

if __name__ == "__main__":
    main()
