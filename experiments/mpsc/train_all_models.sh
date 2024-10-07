#!/bin/bash
for SYS in quadrotor_3D; do
    for ALGO in ppo; do
        for TASK in track; do
            for SEED in 42 62 821 99 4077; do
                # MPSF Ablation
                ./train_model.sbatch none False False $SYS $TASK $ALGO False False $SEED #none
                ./train_model.sbatch none False True  $SYS $TASK $ALGO False 1     $SEED #none_pen_1
                ./train_model.sbatch none True  False $SYS $TASK $ALGO False False $SEED #none_sr
                ./train_model.sbatch none True  True  $SYS $TASK $ALGO False 1     $SEED #none_sr_pen_1
                ./train_model.sbatch mpsf False False $SYS $TASK $ALGO False False $SEED #mpsf
                ./train_model.sbatch mpsf False True  $SYS $TASK $ALGO False 1     $SEED #mpsf_pen_1
                ./train_model.sbatch mpsf True  False $SYS $TASK $ALGO False False $SEED #mpsf_sr
                ./train_model.sbatch mpsf True  True  $SYS $TASK $ALGO False 1     $SEED #mpsf_sr_pen_1

                # Constr Pen
                ./train_model.sbatch none False False $SYS $TASK $ALGO True  0.01  $SEED #none_cpen_0.01
                ./train_model.sbatch none False False $SYS $TASK $ALGO True  0.1   $SEED #none_cpen_0.1
                ./train_model.sbatch none False False $SYS $TASK $ALGO True  1     $SEED #none_cpen_1
            done
        done
    done
done
