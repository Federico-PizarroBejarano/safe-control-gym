#!/bin/bash

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stab'
# TASK='track'

# SAFETY_FILTER='linear_mpsc'
SAFETY_FILTER='nl_mpsc'

ALGO='ppo'
# ALGO='sac'

TAG='early_stop_2'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

rm -rf ./unsafe_rl_temp_data/${TAG}/

# Train the unsafe controller/agent.
python3 train_rl.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
    --output_dir ./unsafe_rl_temp_data/${TAG}/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None
