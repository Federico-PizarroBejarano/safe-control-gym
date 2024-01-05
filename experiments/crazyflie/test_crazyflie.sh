#!/bin/bash

TASK='track'

MPSC='nl_mpsc'
# MPSC='linear_mpsc'

MPSC_COST='one_step_cost'
# MPSC_COST='constant_cost'
# MPSC_COST='regularized_cost'
# MPSC_COST='lqr_cost'
# MPSC_COST='precomputed_cost'
# MPSC_COST='learned_cost'

MPSC_COST_HORIZON=10

# python3 ./train_rl.py \
python3 ./crazyflie_experiment.py \
    --task quadrotor \
    --algo ppo \
    --safety_filter ${MPSC} \
    --overrides \
        ./config_overrides/crazyflie_${TASK}.yaml \
        ./config_overrides/ppo_crazyflie.yaml \
        ./config_overrides/nl_mpsc.yaml \
    --output_dir ./models/rl_models/ppo/ \
    --kv_overrides \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        # task_config.init_state=None
