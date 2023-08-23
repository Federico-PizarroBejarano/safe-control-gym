#!/bin/bash
for SYS in cartpole; do
    for ALGO in ppo sac; do
        for TASK in stab track; do
            for SF in none mpsf; do
                for SAFE_RESET in True False; do
                    for EARLY_STOP in True False; do
                        for PENALIZE_SF in True False; do
                            if [ "$SF" = 'none' ] && [ "$PENALIZE_SF" = 'True' ]; then
                                continue
                            fi

                            sbatch train_model.sbatch $SF $SAFE_RESET $EARLY_STOP $PENALIZE_SF $SYS $TASK $ALGO
                        done
                    done
                done
            done
        done
    done
done
