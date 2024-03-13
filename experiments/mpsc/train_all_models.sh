#!/bin/bash
for SYS in quadrotor_3D; do
    for ALGO in ppo; do
        for TASK in track; do
            for SEED in 42 62 821 99 4077; do # 1102 1014 14 960406 2031; do
                sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 1 $SEED #mpsf_sr_pen_1
                sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 10 $SEED #mpsf_sr_pen_10
                sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 100 $SEED #mpsf_sr_pen_100
                sbatch train_model.sbatch mpsf True True $SYS $TASK $ALGO False 1000 $SEED #mpsf_sr_pen_1000
                sbatch train_model.sbatch none False False $SYS $TASK $ALGO False False $SEED #none
                sbatch train_model.sbatch none False False $SYS $TASK $ALGO True False $SEED #none_cpen
            done
        done
    done
done
