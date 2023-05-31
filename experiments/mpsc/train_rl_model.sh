#!/bin/bash

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stab'
# TASK='track'

ALGO='ppo'
# ALGO='sac'

# SAFETY_FILTER='linear_mpsc'
SAFETY_FILTER='nl_mpsc'

MPSC_COST='one_step_cost'
# MPSC_COST='constant_cost'
# MPSC_COST='regularized_cost'
# MPSC_COST='lqr_cost'
# MPSC_COST='precomputed_cost'
# MPSC_COST='learned_cost'

MPSC_COST_HORIZON=2
DECAY_FACTOR=0.95

TAG='TEST'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
python3 train_rl.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
    --output_dir ./unsafe_rl_temp_data/${TAG}/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.soften_constraints=True \
