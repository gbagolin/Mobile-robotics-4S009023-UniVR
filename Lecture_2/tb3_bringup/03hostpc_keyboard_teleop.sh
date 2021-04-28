#!/bin/bash

source ./config.sh

echo "Starting teleop node..."

source $HOME/catkin_ws/devel/setup.bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
