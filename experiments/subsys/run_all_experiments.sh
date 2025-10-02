#!/bin/bash
for TRAJ_TYPE in severe_collision; do
    for SF_TYPE in none_lqr none_mpc none naive safe_teleop_basic safe_teleop_advanced safe_swarm_basic safe_swarm_advanced ours; do
        ./subsys_experiment.sh $TRAJ_TYPE $SF_TYPE
    done
done
