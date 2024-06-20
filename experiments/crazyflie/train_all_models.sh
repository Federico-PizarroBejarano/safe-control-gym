#!/bin/bash
for ALGO in ppo; do
    for SEED in 42 62 821 99 4077; do # 1102 1014 14 960406 2031; do
        ./test_crazyflie.sh mpsf $ALGO False 0.1 $SEED #mpsf_sr_pen_0.1
        ./test_crazyflie.sh mpsf $ALGO False 1 $SEED #mpsf_sr_pen_1
        ./test_crazyflie.sh mpsf $ALGO False 10 $SEED #mpsf_sr_pen_10
        ./test_crazyflie.sh none $ALGO False False $SEED #none
        ./test_crazyflie.sh none $ALGO True False $SEED #none_cpen
    done
done
