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

SYS='cartpole'
TASK='stab'
ALGO='ppo'
SAFETY_FILTER='nl_mpsc'

# Train the unsafe controller/agent.
python3 train_rl.py \
    --algo ${ALGO} \
    --task ${SYS} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
    --output_dir ./models/rl_models/test3/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        sf_config.cost_function=one_step_cost \
        sf_config.soften_constraints=True \
        algo_config.filter_train_actions=True \
        algo_config.use_safe_reset=False \
        algo_config.penalize_sf_diff=True
