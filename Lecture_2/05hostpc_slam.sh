#!/bin/bash

source ./config.sh

echo "Starting SLAM node..."

source $HOME/catkin_ws/devel/setup.bash
roslaunch turtlebot3_slam turtlebot3_slam.launch

NOW=$(date +%Y%m%d_%H%M%S)

echo "Saving map to map_$NOW..."
rosrun map_server map_saver -f "map_$NOW"

