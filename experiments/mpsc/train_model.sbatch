#!/bin/bash
#SBATCH --job-name=safe_control_gym_train_rl_%j
#SBATCH --output=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.out
#SBATCH --error=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cpu,gpu,wsgpu
#SBATCH --ntasks=1
#SBATCH -t 0-12:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --qos=normal

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stab'
# TASK='track'

ALGO='ppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

if [ "$5" ]; then
  SYS=$5
fi
if [ "$6" ]; then
  TASK=$6
fi
if [ "$7" ]; then
  ALGO=$7
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
    EARLY_STOP_TAG='_es'
else
    EARLY_STOP_TAG=''
fi

if [ "$4" = 'True' ] && [ "$1" != 'none' ]; then
    PENALIZE_SF_TAG='_pen'
else
    PENALIZE_SF_TAG=''
fi

TAG="$1${SAFE_RESET_TAG}${EARLY_STOP_TAG}${PENALIZE_SF_TAG}"
echo $TAG $SYS $ALGO $TASK

if [ "$SYS" = 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

if [ "$ALGO" == 'safe_explorer_ppo' ]; then
    # Pretrain the unsafe controller/agent.
    python3 train_rl.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}_pretrain.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./models/rl_models/${SYS}/${TASK}/${ALGO}_pretrain/ \
        --seed 2 \
        --kv_overrides \
            task_config.init_state=None
fi

# Train the unsafe controller/agent.
python3 train_rl.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}_linear.yaml \
    --output_dir ./models/rl_models/${SYS}/${TASK}/${ALGO}/${TAG}/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.soften_constraints=True \
        algo_config.filter_train_actions=${FILTER} \
        algo_config.use_safe_reset=$2 \
        task_config.done_on_violation=$3 \
        algo_config.penalize_sf_diff=$4 \
        algo_config.pretrained=./models/rl_models/${SYS}/${TASK}/${ALGO}_pretrain/ \

./mpsc_experiment.sh $TAG $SYS $TASK $ALGO
