#!/bin/bash

# Training script for adversarial RL models - Supports cartpole and quadrotor_2D
# Usage: ./train_rl_model_unified.sh [system] [task] [algo]
# Example: ./train_rl_model_unified.sh quadrotor_2D track ppo

# Default values
SYS=${1:-'quadrotor_2D'}    # Options: 'cartpole' or 'quadrotor_2D'
TASK=${2:-'track'}      # Options: 'track' or 'stab'
ALGO=${3:-'ppo'}        # Options: 'ppo' or 'sac'
SAFETY_FILTER='nl_mpsc'

# Determine system name for task registration
if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

echo "Training RL model with:"
echo "  System: $SYS"
echo "  Task: $TASK"
echo "  Algorithm: $ALGO"
echo "  Safety Filter: $SAFETY_FILTER"
echo ""

# Train the unsafe controller/agent.
python3 train_rl.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
    --output_dir ./models/rl_models/${SYS}/${TASK}/${ALGO} \
    --seed 42 \
    --kv_overrides \
        task_config.init_state=None \
        sf_config.soften_constraints=True
