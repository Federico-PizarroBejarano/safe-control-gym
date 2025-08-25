#!/bin/bash
for TRAJ_TYPE in no_collision mild_collision medium_collision severe_collision; do
    for SF_TYPE in none naive safe_teleop_basic safe_teleop_advanced safe_swarm_basic safe_swarm_advanced ours; do
        ./subsys_experiment.sh $TRAJ_TYPE $SF_TYPE
    done
done
