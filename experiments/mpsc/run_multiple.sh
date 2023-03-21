#!/bin/bash

MPSC_COST_HORIZON=2

for SYS in cartpole quadrotor_2D quadrotor_3D; do
    for TASK in stab track; do
        for ALGO in lqr pid ppo sac; do
            for SAFETY_FILTER in nl_mpsc; do
                for MPSC_COST in one_step_cost regularized_cost precomputed_cost; do
                    echo STARTING TEST $SYS $TASK $ALGO $SAFETY_FILTER $MPSC_COST

                    if [ "$SYS" == 'cartpole' ] && [ "$ALGO" == 'pid' ]; then
                        echo SKIPPING - PID only implemented for quadrotor
                        continue
                    fi

                    if [ "$SYS" == 'cartpole' ]; then
                        SYS_NAME=$SYS
                    else
                        SYS_NAME='quadrotor'
                    fi

                    # Model-predictive safety certification of an unsafe controller.
                    python3 ./mpsc_experiment.py \
                        --task ${SYS_NAME} \
                        --algo ${ALGO} \
                        --safety_filter ${SAFETY_FILTER} \
                        --overrides \
                            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                            ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
                        --kv_overrides \
                            sf_config.cost_function=${MPSC_COST} \
                            sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} || exit 1
                done
            done
        done
    done
done
