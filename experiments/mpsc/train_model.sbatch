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

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='cpo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

if [ "$4" ]; then
  SYS=$4
fi
if [ "$5" ]; then
  TASK=$5
fi
if [ "$6" ]; then
  ALGO=$6
fi

# SAFETY_FILTER='linear_mpsc'
SAFETY_FILTER='nl_mpsc'

if [ "$1" = 'm_mpsf' ]; then
    MPSC_COST='precomputed_cost'
    FILTER='True'
elif [ "$1" = 'mpsf' ]; then
    MPSC_COST='one_step_cost'
    FILTER='True'
else
    MPSC_COST='one_step_cost'
    FILTER='False'
fi

MPSC_COST_HORIZON=2
DECAY_FACTOR=0.85

if [ "$2" = 'True' ]; then
    SAFE_RESET_TAG='_sr'
else
    SAFE_RESET_TAG=''
fi

if [ "$3" = 'True' ]; then
    PENALIZE_SF_TAG='_pen'
else
    PENALIZE_SF_TAG=''
fi

if [ "$7" = True ]; then
    CONSTR_PEN='True'
    CONSTR_PEN_TAG='_cpen'
else
    CONSTR_PEN='False'
    CONSTR_PEN_TAG=''
fi

if [ "$8" = False ]; then
    SF_PEN_TAG=''
else
    SF_PEN_TAG="_$8"
fi

if [ -z "$9" ]; then
    SEED=1337
else
    SEED=$9
fi

TAG="$1${SAFE_RESET_TAG}${PENALIZE_SF_TAG}${CONSTR_PEN_TAG}${SF_PEN_TAG}"
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
    --output_dir ./models/rl_models/${SYS}/${TASK}/${ALGO}/${TAG}/seed_${SEED}/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        task_config.use_constraint_penalty=${CONSTR_PEN} \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.soften_constraints=True \
        algo_config.filter_train_actions=${FILTER} \
        algo_config.use_safe_reset=$2 \
        algo_config.penalize_sf_diff=$3 \
        algo_config.sf_penalty=$8 \
        task_config.seed=${SEED} \
        algo_config.seed=${SEED} \
        sf_config.seed=${SEED} \

./mpsc_experiment.sh $TAG $SYS $TASK $ALGO $SEED
# python plotting_results.py $SYS $TASK $ALGO
