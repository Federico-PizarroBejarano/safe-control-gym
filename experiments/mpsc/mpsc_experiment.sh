#!/bin/bash

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

ALGO='ppo'

SAFETY_FILTER='nl_mpsc'
# MPSC_COST='one_step_cost'
MPSC_COST='precomputed_cost'
MPSC_COST_HORIZON=5
DECAY_FACTOR=0.0

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Model predictive safety certification of an unsafe controller.
python3 ./mpsc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
    --kv_overrides \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
    --model=${1}
