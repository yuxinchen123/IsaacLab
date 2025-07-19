#!/bin/bash
# Enhanced Isaac Lab Training Script with Video + Wandb
# This script uses the existing Isaac Lab training with optimized settings

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Enhanced Isaac Lab Training with Video + Wandb${NC}"
echo "======================================================"

# Check if in Isaac Lab directory
if [ ! -f "scripts/reinforcement_learning/rsl_rl/train.py" ]; then
    echo -e "${RED}❌ Please run this script from Isaac Lab directory${NC}"
    exit 1
fi

# Activate environment
echo -e "${YELLOW}🔧 Activating Isaac Lab environment...${NC}"
source ~/env_isaaclab/bin/activate

# Check wandb login status
echo -e "${YELLOW}📊 Checking wandb status...${NC}"
if ! python -c "import wandb; wandb.login()" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Wandb not logged in. Please log in:${NC}"
    echo "Enter your wandb API key (get it from https://wandb.ai/authorize):"
    wandb login
fi

# Set default values
TASK=${1:-"Isaac-Ant-v0"}
NUM_ENVS=${2:-4096}
PROJECT_NAME=${3:-"isaac-ant-locomotion"}
RUN_NAME=${4:-"ant-video-$(date +%m%d-%H%M)"}

echo -e "${GREEN}🎯 Training Configuration:${NC}"
echo "  Task: $TASK"
echo "  Environments: $NUM_ENVS"
echo "  Wandb Project: $PROJECT_NAME"
echo "  Run Name: $RUN_NAME"
echo "  Video Recording: Enabled"
echo "  Headless Mode: Enabled"

echo -e "${BLUE}🏋️ Starting enhanced training...${NC}"

# Run training with video recording and wandb
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --enable_cameras \
    --video \
    --video_length 200 \
    --video_interval 100 \
    --logger wandb \
    --log_project_name "$PROJECT_NAME" \
    --run_name "$RUN_NAME" \
    --seed 42

echo -e "${GREEN}✅ Training completed!${NC}"
echo -e "${BLUE}📊 Check your results at: https://wandb.ai/your-username/$PROJECT_NAME${NC}"
