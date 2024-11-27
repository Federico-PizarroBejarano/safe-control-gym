#!/bin/bash
sbatch train_model.sbatch False 1 1 False
for MPSC_COST_HORIZON in 2 5 10 20; do
    for DECAY_FACTOR in 0.25 0.5 0.75 1; do
        # Ignore precomputed differences
        sbatch train_model.sbatch False $MPSC_COST_HORIZON $DECAY_FACTOR False
        sbatch train_model.sbatch True  $MPSC_COST_HORIZON $DECAY_FACTOR False

        # Preserve random state
        sbatch train_model.sbatch False $MPSC_COST_HORIZON $DECAY_FACTOR True
        sbatch train_model.sbatch True  $MPSC_COST_HORIZON $DECAY_FACTOR True
    done
done
