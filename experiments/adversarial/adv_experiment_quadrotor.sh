#!/bin/bash

# Quadrotor 2D Adversarial Experiment Script
# This script runs the MPSC safety filter with an adversarial RL agent on the 2D quadrotor

SYS='quadrotor_2D'
TASK='track'  # Options: 'track' or 'stab'
ALGO='ppo'    # Options: 'ppo' or 'sac'
SAFETY_FILTER='nl_mpsc'
MPSC_COST='precomputed_cost'  # Options: 'one_step_cost' or 'precomputed_cost'
MPSC_COST_HORIZON=20

# System name for the task registration
SYS_NAME='quadrotor'

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
