#!/bin/bash

SYS='quadrotor_3D_attitude'
SAFETY_FILTER='nl_mpsc'
MPSC_COST='one_step_cost'
MPSC_COST_HORIZON=1
DECAY_FACTOR=1

python3 ./subsys_experiment.py \
    --task quadrotor \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}.yaml \
        ./config_overrides/${SAFETY_FILTER}_${SYS}.yaml \
    --kv_overrides \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.max_w=0.002 \
        sf_config.slack_cost=1000.0
