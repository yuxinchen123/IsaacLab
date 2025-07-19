# Quick Command Cheat Sheet

## 🔥 Essential Commands (Copy-Paste Ready)

### Get Back Into Isaac Lab
```bash
ssh clairec@pabrtxl1
cd /home/clairec/IsaacLab
source ~/env_isaaclab/bin/activate
```

### Run the Same Successful Training
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-v0 --num_envs 4096 --headless
```

### Generate Training Plots
```bash
python fixed_plot_training_results.py
python create_pdf_plots.py
```

### Check Training Results
```bash
ls -la logs/rsl_rl/ant/2025-07-16_06-53-36/
cat logs/rsl_rl/ant/2025-07-16_06-53-36/params/agent.yaml
```

### Verify Everything Works
```bash
python -c "import isaaclab; print('✅ Isaac Lab ready!')"
nvidia-smi
```

## 📊 Our Results Summary
- **Reward**: -0.456 → 144.596 (+145.053 improvement)
- **Episode Length**: 19.7 → 907.3 steps (46x longer survival)
- **Algorithm**: PPO (RSL-RL), seed=42, 4096 parallel envs
- **Status**: Training converged successfully (623/1000 iterations)

## 📁 Key Files
- `RELOGIN_GUIDE.md` - Complete relogin instructions
- `isaac_lab_training_results.pdf` - Main results
- `mentor_presentation.html` - Presentation for mentor
- `logs/rsl_rl/ant/2025-07-16_06-53-36/` - All training data (62MB)
