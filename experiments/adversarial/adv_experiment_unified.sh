#!/bin/bash

# Adversarial Experiment Script - Supports cartpole and quadrotor_2D
# Usage: ./adv_experiment_unified.sh [system] [task] [algo]
# Example: ./adv_experiment_unified.sh quadrotor_2D track ppo

# Default values
SYS=${1:-'cartpole'}    # Options: 'cartpole' or 'quadrotor_2D'
TASK=${2:-'track'}      # Options: 'track' or 'stab'
ALGO=${3:-'ppo'}        # Options: 'ppo' or 'sac'
SAFETY_FILTER='nl_mpsc'
MPSC_COST='precomputed_cost'  # Options: 'one_step_cost' or 'precomputed_cost'
MPSC_COST_HORIZON=30

# Determine system name for task registration
if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

echo "Running adversarial experiment with:"
echo "  System: $SYS"
echo "  Task: $TASK"
echo "  Algorithm: $ALGO"
echo "  Safety Filter: $SAFETY_FILTER"
echo ""

# Model-predictive safety certification of an unsafe controller.
python3 ./adv_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
    --kv_overrides \
        task_config.randomized_init=False \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON}
