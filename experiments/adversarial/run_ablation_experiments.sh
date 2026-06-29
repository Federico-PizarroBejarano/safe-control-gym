#!/bin/bash

# Usage: ./run_ablation_experiments.sh [system] [task] [algo]
# Example: ./run_ablation_experiments.sh quadrotor_2D track ppo

SYS=${1:-'cartpole'}
TASK=${2:-'track'}
ALGO=${3:-'ppo'}
SAFETY_FILTER='nl_mpsc'
OUTPUT_DIR='./ablation'

# Determine system name for task registration
if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

echo "Running ablation experiments for:"
echo "  System: $SYS"
echo "  Task: $TASK"
echo "  Algorithm: $ALGO"
echo ""

# Ablation experiment configurations
# Format: "experiment_name:kv_override1:kv_override2:..."
EXPERIMENTS=(
    # "correction_only:algo_config.adversarial_reward.use_theta_reward=False:algo_config.adversarial_reward.use_velocity_reward=False:algo_config.adversarial_reward.use_oscillation_reward=False:algo_config.adversarial_reward.use_stability_penalty=False"
    "state_only:algo_config.adversarial_reward.use_correction_reward=False:algo_config.adversarial_reward.use_correction_ratio=False:algo_config.adversarial_reward.use_correction_bonus=False:algo_config.adversarial_reward.use_no_correction_penalty=False"
    "no_correction_mag:algo_config.adversarial_reward.use_correction_reward=False"
    # "no_correction_ratio:algo_config.adversarial_reward.use_correction_ratio=False"
    "no_correction_bonus:algo_config.adversarial_reward.use_correction_bonus=False"
    # "no_correction_penalty:algo_config.adversarial_reward.use_no_correction_penalty=False"
    # "no_theta:algo_config.adversarial_reward.use_theta_reward=False"
    "no_velocity:algo_config.adversarial_reward.use_velocity_reward=False"
    # "no_oscillation:algo_config.adversarial_reward.use_oscillation_reward=False"
    "no_stability_penalty:algo_config.adversarial_reward.use_stability_penalty=False"
    # "no_cart_penalty:algo_config.adversarial_reward.use_cart_penalty=False"  # cartpole only
    "w_correction_10:algo_config.adversarial_reward.w_correction=10.0"
    # "w_correction_50:algo_config.adversarial_reward.w_correction=50.0"
    # "temp_5:algo_config.adversarial_reward.temperature=5.0"
    "temp_30:algo_config.adversarial_reward.temperature=30.0"
    "temp_500:algo_config.adversarial_reward.temperature=500.0"
    # "no_safe_reset:algo_config.use_safe_reset=False"
)

# Add quadrotor-specific ablations if running quadrotor
if [ "$SYS" == 'quadrotor_2D' ]; then
    EXPERIMENTS+=(
        "no_altitude_penalty:algo_config.adversarial_reward.use_altitude_penalty=False"
        "no_position_reward:algo_config.adversarial_reward.use_position_reward=False"
    )
fi

for SEED in 2; do
    for EXP in "${EXPERIMENTS[@]}"; do
        # Parse experiment name and overrides
        IFS=':' read -ra PARTS <<< "$EXP"
        EXP_NAME="${PARTS[0]}"

        # Build kv_overrides string
        KV_OVERRIDES=""
        for ((i=1; i<${#PARTS[@]}; i++)); do
            KV_OVERRIDES="$KV_OVERRIDES ${PARTS[$i]}"
        done

        echo "Training: ${EXP_NAME} (seed=${SEED})"

        python3 ./train_rl.py \
            --algo ${ALGO} \
            --task ${SYS_NAME} \
            --safety_filter ${SAFETY_FILTER} \
            --overrides \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
            --output_dir ${OUTPUT_DIR}/${SYS}/${EXP_NAME}/seed_${SEED} \
            --seed ${SEED} \
            --kv_overrides ${KV_OVERRIDES}
    done
done
