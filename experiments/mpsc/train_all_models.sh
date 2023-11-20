#!/bin/bash
for SYS in quadrotor_3D; do
    for ALGO in ppo; do
        for TASK in track; do
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 0 #mpsf_sr_pen_0
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 1 #mpsf_sr_pen_1
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 1 #mpsf_sr_pen_3
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 10 #mpsf_sr_pen_10
            sbatchk train_model.sbatch mpsf True True $SYS $TASK $ALGO False 25 #mpsf_sr_pen_25
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 75 #mpsf_sr_pen_75
            sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 200 #mpsf_sr_pen_200
            sbatch train_model.sbatch none False False $SYS $TASK $ALGO False False #none
            sbatch train_model.sbatch none False False $SYS $TASK $ALGO True False #none_cpen
            if [ "$ALGO" == 'ppo' ]; then
                sbatch train_model.sbatch none False False $SYS $TASK safe_explorer_ppo False #safe-ppo
                sbatch train_model.sbatch none False False $SYS $TASK cpo False #cpo
            fi
        done
    done
done
