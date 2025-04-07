#!/bin/bash
#SBATCH --job-name=safe_control_gym_train_rl_%j
#SBATCH --output=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.out
#SBATCH --error=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.err
#SBATCH --partition=cpu
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --qos=normal

SYS='cartpole'
TASK='track'
ALGO='ppo'
SAFETY_FILTER='nl_mpsc'

MPSC_COST='one_step_cost'

MPSC_COST_HORIZON=1
DECAY_FACTOR=1.0
SF_PENALTY=1.0

TAG="grad_${SYS}_${TASK}_${ALGO}_uncert"
echo $TAG $SYS $ALGO $TASK

if [ "$SYS" = 'cartpole' ]; then
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
    --output_dir ./models/rl_models/${TAG}/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.soften_constraints=True \
        algo_config.filter_train_actions=False \
        algo_config.use_safe_reset=False \
        algo_config.penalize_sf_diff=False \
        algo_config.sf_penalty=$SF_PENALTY
