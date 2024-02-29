#!/bin/bash

ALGO='ppo'
# ALGO='sac'

if [ "$2" ]; then
  ALGO=$2
fi

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

if [ "$1" = 'mpsf' ]; then
    FILTER='True'
else
    FILTER='False'
fi

if [ "$3" = True ]; then
    CONSTR_PEN_TAG='_cpen'
else
    CONSTR_PEN_TAG=''
fi

if [ "$4" = False ]; then
    SF_PEN_TAG=''
else
    SF_PEN_TAG="_$4"
fi

TAG="$1${CONSTR_PEN_TAG}${SF_PEN_TAG}_dm_t1"
echo $TAG $SYS $ALGO $TASK

python3 ./train_rl.py \
    --task quadrotor \
    --algo ${ALGO} \
    --safety_filter ${MPSC} \
    --overrides \
        ./config_overrides/crazyflie_${TASK}.yaml \
        ./config_overrides/${ALGO}_crazyflie.yaml \
        ./config_overrides/nl_mpsc.yaml \
    --output_dir ./models/rl_models/${ALGO}/${TAG} \
    --kv_overrides \
        sf_config.cost_function=one_step_cost \
        algo_config.filter_train_actions=$FILTER \
        algo_config.penalize_sf_diff=$FILTER \
        algo_config.use_safe_reset=$FILTER \
        algo_config.sf_penalty=$4 \
        task_config.use_constraint_penalty=$3

python3 ./crazyflie_experiment.py \
    --task quadrotor \
    --algo ${ALGO} \
    --safety_filter ${MPSC} \
    --overrides \
        ./config_overrides/crazyflie_${TASK}.yaml \
        ./config_overrides/${ALGO}_crazyflie.yaml \
        ./config_overrides/nl_mpsc.yaml \
    --output_dir ./models/rl_models/${ALGO}/${TAG} \
    --kv_overrides \
        sf_config.cost_function=precomputed_cost \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        algo_config.filter_train_actions=$FILTER \
        algo_config.penalize_sf_diff=$FILTER \
        algo_config.use_safe_reset=$FILTER \
        algo_config.sf_penalty=$4 \
        task_config.use_constraint_penalty=$3
