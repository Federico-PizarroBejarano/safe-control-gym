#!/bin/bash
for ALGO in ppo; do
    ./test_crazyflie.sh mpsf $ALGO False 0.1 #mpsf_sr_pen_0.1
    ./test_crazyflie.sh mpsf $ALGO False 1 #mpsf_sr_pen_1
    ./test_crazyflie.sh mpsf $ALGO False 10 #mpsf_sr_pen_10
    # ./test_crazyflie.sh none $ALGO False False #none
    ./test_crazyflie.sh none $ALGO True False #none_cpen
done
