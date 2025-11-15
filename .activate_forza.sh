#!/bin/bash
TARGET_PATH="/home/misys/forza_ws/race_stack/base_system/f110_simulator/f1tenth_gym/gym   
/home/misys/forza_ws/race_stack/planner/global_planner/global_planner/global_racetrajectory_optimization"
echo "[INFO] Setting f110_gym to use forza_ws"
echo "$TARGET_PATH" > ~/.local/lib/python3.10/site-packages/easy-install.pth
