#!/bin/bash

NUM_DRONES=4
MPSC_COST='one_step_cost'
MPSC_COST_HORIZON=1
DECAY_FACTOR=1

python3 ./subsys_experiment.py \
    --task quadrotor \
    --safety_filter nl_mpsc \
    --overrides \
        ./config_overrides/quadrotor_3D_attitude.yaml \
        ./config_overrides/nl_mpsc_quadrotor_3D_attitude.yaml \
    --kv_overrides \
        num_drones=${NUM_DRONES} \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.max_w=0.002 \
        sf_config.slack_cost=1000.0
