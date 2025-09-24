#!/bin/bash

# MPSC_COST='one_step_cost'
MPSC_COST='precomputed_cost'
MPSC_COST_HORIZON=10
DECAY_FACTOR=0.85

# TRAJ_TYPE='no_collision'
# TRAJ_TYPE='mild_collision'
# TRAJ_TYPE='medium_collision'
TRAJ_TYPE='severe_collision'

SF_TYPE='none'
# SF_TYPE='naive'
# SF_TYPE='safe_teleop_basic'
# SF_TYPE='safe_teleop_advanced'
# SF_TYPE='safe_swarm_basic'
# SF_TYPE='safe_swarm_advanced'
# SF_TYPE='ours'

if [ "$1" ]; then
  TRAJ_TYPE=$1
fi
if [ "$2" ]; then
  SF_TYPE=$2
fi

python3 ./subsys_experiment.py \
    --algo lqr \
    --task quadrotor \
    --safety_filter nl_mpsc \
    --overrides \
        ./config_overrides/quadrotor_3D_attitude.yaml \
        ./config_overrides/nl_mpsc.yaml \
        ./config_overrides/lqr.yaml \
    --kv_overrides \
        traj_type=${TRAJ_TYPE} \
        sf_type=${SF_TYPE} \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.slack_cost=1000.0
