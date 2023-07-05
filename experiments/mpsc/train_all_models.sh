#!/bin/bash

for SF in none mpsf m_mpsf; do
    for SAFE_RESET in True False; do
        for EARLY_STOP in True False; do
            for PENALIZE_SF in True False; do
                if [ "$SF" = 'none' ] && [ "$PENALIZE_SF" = 'True' ]; then
                    continue
                fi

                # These cases dont work and slow down training
                if [ "$SF" != 'none' ] && [ "$SAFE_RESET" = 'True' ] && [ "$EARLY_STOP" = 'True' ] && [ "$PENALIZE_SF" = 'True' ]; then
                    continue
                fi

                if [ "$SF" = 'mpsf' ] && [ "$SAFE_RESET" = 'False' ] && [ "$EARLY_STOP" = 'True' ] && [ "$PENALIZE_SF" = 'True' ]; then
                    continue
                fi

                sbatch train_model.sbatch $SF $SAFE_RESET $EARLY_STOP $PENALIZE_SF
            done
        done
    done
done
