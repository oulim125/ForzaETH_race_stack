#!/bin/bash

USERNAME=$(whoami)
USERNAME=$(whoami)
USERNAME=$(whoami)
USERNAME=$(whoami)
# Setup permissions
# USER="$(id -u -n)"
sudo chown -R $USERNAME /home/misys/forza_ws/

# Install dependencies
rosdep update &&
    rosdep install --from-paths /home/misys/forza_forza_ws --ignore-src -y

# Setup race_stack
bash /home/$(whoami)/forza_ws/race_stack/src/race_stack/.install_utils/f110_sim_setup.sh || echo "Failed to setup f110_sim"
bash /home/$(whoami)/forza_ws/race_stack/src/race_stack/.install_utils/gb_opt_setup.sh || echo "Failed to setup gb_opt"

# Apply Joystick patch
sudo chmod 666 /dev/input/js0
sudo chmod 666 /dev/input/event*

# setup f1tenth_gym
cd /home/$(whoami)/forza_ws/race_stack &&
    colcon build --packages-up-to f110_gym --base-paths /home/$(whoami)/forza_ws/race_stack \
        --cmake-args "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On" \
        -Wall -Wextra -Wpedantic --cmake-clean-cache
