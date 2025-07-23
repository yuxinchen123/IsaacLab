#!/bin/bash
"""
Hyperparameter sweep script - WandB Bayesian optimization
Automatically searches optimal hyperparameters with early termination
"""

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ISAACLAB_DIR="/home/clairec/IsaacLab"
PYTHON_EXE="$ISAACLAB_DIR/isaaclab.sh -p"

echo "=========================================="
echo "Isaac Factory PegInsert - Hyperparameter Sweep"
echo "=========================================="
echo "Mode: WandB Bayesian optimization"
echo "Search space: Learning rate, clip_param, entropy_coef, etc."
echo "Early termination: Hyperband"
echo "Agents: 4 (reduced for sweep efficiency)"
echo "Environments per agent: 128"
echo "=========================================="

# Check WandB login
if ! wandb status | grep -q "Logged in"; then
    echo "Please login to WandB first: wandb login"
    exit 1
fi

# Change to IsaacLab directory
cd "$ISAACLAB_DIR"

# Execute hyperparameter sweep
echo "Starting WandB sweep..."
echo "This will create a sweep and run multiple agents automatically."
echo "You can monitor progress at: https://wandb.ai/"

exec $PYTHON_EXE multimodal_ppo_training.py \
    --sweep \
    --use_wandb \
    --agents 4 \
    --envs-per-agent 128 \
    --max-iter 5000 \
    --headless
