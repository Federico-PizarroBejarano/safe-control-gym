#!/bin/bash
for SYS in quadrotor_3D; do
    for ALGO in ppo; do
        for TASK in track; do
            for SF in none mpsf; do
                for SAFE_RESET in True False; do
                    for EARLY_STOP in True False; do
                        for PENALIZE_SF in True False; do
                            if [ "$SF" = 'none' ] && [ "$SAFE_RESET" = 'False' ] && [ "$EARLY_STOP" = 'False' ] && [ "$PENALIZE_SF" = 'False' ]; then
                                sbatch train_model.sbatch $SF $SAFE_RESET $EARLY_STOP $PENALIZE_SF $SYS $TASK $ALGO True

                                if [ "$ALGO" = 'ppo' ]; then
                                    sbatch train_model.sbatch $SF $SAFE_RESET $EARLY_STOP $PENALIZE_SF $SYS $TASK safe_explorer_ppo False
                                fi
                            fi

                            sbatch train_model.sbatch $SF $SAFE_RESET $EARLY_STOP $PENALIZE_SF $SYS $TASK $ALGO False
                        done
                    done
                done
            done
        done
    done
done
