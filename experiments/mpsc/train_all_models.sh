#!/bin/bash
for SYS in quadrotor_3D; do
    for ALGO in sac ppo; do
        for TASK in track; do
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False #mpsf_sr_pen
            sbatch train_model.sbatch none False False $SYS $TASK $ALGO False #none
            sbatch train_model.sbatch none False False $SYS $TASK $ALGO True #none_cpen
            if [ "$ALGO" == 'ppo' ]; then
                sbatch train_model.sbatch none False False $SYS $TASK safe_explorer_ppo False #safe-ppo
                sbatch train_model.sbatch none False False $SYS $TASK cpo False #cpo
            fi
        done
    done
done
