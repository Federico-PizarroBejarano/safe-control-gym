#!/bin/bash
sbatch train_model.sbatch True 1 1
for MPSC_COST_HORIZON in 2 5 10 20; do
    for DECAY_FACTOR in 0.25 0.5 0.75 1; do
        sbatch train_model.sbatch True  $MPSC_COST_HORIZON $DECAY_FACTOR
        sbatch train_model.sbatch False $MPSC_COST_HORIZON $DECAY_FACTOR
    done
done
