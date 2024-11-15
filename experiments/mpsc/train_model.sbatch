#!/bin/bash
#SBATCH --job-name=safe_control_gym_train_rl_%j
#SBATCH --output=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.out
#SBATCH --error=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.err
#SBATCH --partition=cpu
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --qos=normal

SYS='quadrotor_3D'
TASK='track'
ALGO='ppo'
SAFETY_FILTER='nl_mpsc'

MPSC_COST='precomputed_cost'

MPSC_COST_HORIZON=$2
DECAY_FACTOR=$3
SF_PENALTY=1.0

TAG="curriculum_$1_$2_$3"
echo $TAG $SYS $ALGO $TASK

# Train the unsafe controller/agent.
python3 train_rl.py \
    --algo ${ALGO} \
    --task 'quadrotor' \
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
        sf_config.max_decay_factor=${DECAY_FACTOR} \
        sf_config.soften_constraints=True \
        algo_config.filter_train_actions=${FILTER} \
        algo_config.use_safe_reset=${FILTER} \
        algo_config.penalize_sf_diff=${FILTER} \
        algo_config.sf_penalty=SF_PENALTY

./mpsc_experiment.sh $TAG $SYS $TASK $ALGO
