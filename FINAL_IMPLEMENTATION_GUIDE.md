# 🎬 FINAL IMPLEMENTATION: Video Recording + Wandb for Isaac Lab

## 🚀 Step 1: Setup Your Wandb Account

### Option A: Create New Wandb Account (Recommended)
1. **Go to**: https://wandb.ai/signup
2. **Sign up** with your email
3. **Get your API key**: https://wandb.ai/authorize
4. **Copy the API key** (looks like: `1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t`)

### Option B: Use Existing Account
1. **Login to**: https://wandb.ai/
2. **Go to**: https://wandb.ai/authorize
3. **Copy your API key**

### Option C: Share Your API Key With Me
If you want me to configure it for you, you can share your wandb API key by:
- Pasting it in the chat: `My wandb API key is: 1a2b3c4d...`
- Or tell me your wandb username and I'll help you set it up

## 🛠️ Step 2: Configure Wandb (Copy-Paste Commands)

```bash
# SSH back to Isaac Lab server
ssh clairec@pabrtxl1
cd /home/clairec/IsaacLab
source ~/env_isaaclab/bin/activate

# Setup wandb with your API key
wandb login
# Paste your API key when prompted

# Or set it directly (replace with your actual key)
export WANDB_API_KEY="your_api_key_here"
wandb login --relogin
```

## 🎯 Step 3: Run Enhanced Training (Ready to Use!)

```bash
# Simple command - everything is pre-configured
python train_enhanced_video_wandb.py

# Or customize the project/run name
python train_enhanced_video_wandb.py \
    --log_project_name "my-isaac-ant-project" \
    --run_name "ant-video-experiment-1"

# Or use original script with wandb enabled
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 4096 \
    --headless \
    --video \
    --video_length 200 \
    --video_interval 100 \
    --logger wandb \
    --log_project_name "isaac-ant-locomotion"
```

## 📊 Step 4: What You'll Get

### Immediate Results:
- ✅ **Training starts** with same successful PPO setup
- ✅ **Videos recorded** every 100 training steps  
- ✅ **Real-time dashboard** at wandb.ai
- ✅ **Automatic uploads** of videos and metrics

### Wandb Dashboard Features:
- 📈 **Live training curves** (reward, episode length, losses)
- 🎥 **Training videos** showing ant learning progression
- 🔍 **Hyperparameter tracking** 
- 📱 **Shareable links** for your mentor
- 💾 **Experiment comparison** across runs

### Local Files:
- 📁 `logs/rsl_rl/ant/*/videos/` - MP4 video files
- 📊 TensorBoard logs (backup)
- 🤖 Model checkpoints

## 🎬 Video Content

Your videos will show:
1. **Early training** (0-200 steps): Ants falling, struggling to coordinate
2. **Learning phase** (200-600 steps): Gradual improvement in balance and movement
3. **Mastery phase** (600+ steps): Smooth, efficient quadruped locomotion

Perfect for presentations! 🐜➡️🚶‍♂️

## 🔗 Accessing Your Results

### Wandb Dashboard:
- **URL**: https://wandb.ai/your-username/isaac-ant-locomotion
- **Shareable**: Send link directly to your mentor
- **Mobile-friendly**: View progress from anywhere

### Share with Mentor:
```bash
# After training starts, you'll get a link like:
# 🔗 Dashboard: https://wandb.ai/your-username/isaac-ant-locomotion/runs/ant-video-0716-1430
```

## 🎯 Quick Test (30 seconds)

To test everything works:
```bash
# Quick 5-iteration test
python train_enhanced_video_wandb.py --max_iterations 5 --video_interval 2
```

This will:
- ✅ Test wandb connection
- ✅ Record a quick video
- ✅ Upload to dashboard
- ✅ Confirm everything works

## 🆘 Troubleshooting

### If wandb login fails:
```bash
# Check your API key
wandb status

# Re-login with specific key
wandb login --relogin
```

### If videos don't upload:
```bash
# Check imageio installation
python -c "import imageio; print('Video ready!')"
```

### If training fails:
```bash
# Fallback to standard training
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-v0 --num_envs 4096 --headless
```

---

## 📋 Implementation Status: ✅ READY TO RUN!

Everything is implemented and ready. Just need your wandb API key to complete the setup! 

**What's your preferred option for wandb setup?**
- Share your API key with me?
- Create new account yourself?
- Use existing account?
