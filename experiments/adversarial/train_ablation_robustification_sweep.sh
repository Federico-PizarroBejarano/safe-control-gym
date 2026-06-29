#!/bin/bash

# Train models across both ablation variants AND robustification parameters
# This sweeps: reward function components × MPC horizons × max_w constraint tightening
#
# Usage: ./train_ablation_robustification_sweep.sh [seed]
# Example: ./train_ablation_robustification_sweep.sh 42

SEED=${1:-42}
SYS='cartpole'
SYS_NAME='cartpole'
TASK='track'
ALGO='ppo'
SAFETY_FILTER='nl_mpsc'
OUTPUT_DIR='./ablation_robustification'

echo "========================================================================"
echo "ABLATION × ROBUSTIFICATION SWEEP"
echo "  Reward variants × MPC parameters"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# Ablation variants
# Format: "name:override1:override2:..."
ABLATIONS=(
    "baseline"
    "state_only:algo_config.adversarial_reward.use_correction_reward=False:algo_config.adversarial_reward.use_correction_ratio=False:algo_config.adversarial_reward.use_correction_bonus=False:algo_config.adversarial_reward.use_no_correction_penalty=False"
    "no_correction_mag:algo_config.adversarial_reward.use_correction_reward=False"
    "no_correction_bonus:algo_config.adversarial_reward.use_correction_bonus=False"
    "no_velocity:algo_config.adversarial_reward.use_velocity_reward=False"
    "no_stability_penalty:algo_config.adversarial_reward.use_stability_penalty=False"
    "w_correction_10:algo_config.adversarial_reward.w_correction=10.0"
    "temp_30:algo_config.adversarial_reward.temperature=30.0"
)

# Robustification parameters
HORIZONS=(20 30 40)
MAX_W_VALUES=(0.001 0.002 0.005)

# Calculate total experiments
TOTAL=$((${#ABLATIONS[@]} * ${#HORIZONS[@]} * ${#MAX_W_VALUES[@]}))
echo "Total experiments: $TOTAL"
echo ""

COUNT=0

for ABLATION in "${ABLATIONS[@]}"; do
    # Parse ablation name and overrides
    IFS=':' read -ra ABLATION_PARTS <<< "$ABLATION"
    ABLATION_NAME="${ABLATION_PARTS[0]}"

    # Build ablation overrides
    ABLATION_OVERRIDES=""
    for ((i=1; i<${#ABLATION_PARTS[@]}; i++)); do
        ABLATION_OVERRIDES="$ABLATION_OVERRIDES ${ABLATION_PARTS[$i]}"
    done

    for HORIZON in "${HORIZONS[@]}"; do
        for MAX_W in "${MAX_W_VALUES[@]}"; do
            COUNT=$((COUNT + 1))

            # Create experiment name
            EXP_NAME="${ABLATION_NAME}_h${HORIZON}_w${MAX_W}"
            EXP_DIR="${OUTPUT_DIR}/seed_${SEED}/${EXP_NAME}"

            echo "[${COUNT}/${TOTAL}] Training: $EXP_NAME"

            # Build safety filter overrides
            SF_OVERRIDES="sf_config.horizon=${HORIZON} sf_config.max_w=${MAX_W}"

            # Run training
            python3 ./train_rl.py \
                --algo ${ALGO} \
                --task ${SYS_NAME} \
                --safety_filter ${SAFETY_FILTER} \
                --overrides \
                    ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                    ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                    ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
                --output_dir ${EXP_DIR} \
                --seed ${SEED} \
                --kv_overrides ${ABLATION_OVERRIDES} ${SF_OVERRIDES}

            if [ $? -ne 0 ]; then
                echo "  [FAILED] Training failed for $EXP_NAME"
            fi
        done
    done
done

echo ""
echo "========================================================================"
echo "TRAINING COMPLETE"
echo "Models saved to: $OUTPUT_DIR/seed_${SEED}/"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python3 eval_ablation_robustification.py --seed $SEED"
echo "  2. Visualize: python3 plot_ablation_robustification.py --seed $SEED"
echo "========================================================================"
