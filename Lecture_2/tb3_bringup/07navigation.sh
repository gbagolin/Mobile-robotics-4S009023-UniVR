#!/bin/bash

source ./config.sh

echo "Starting teleop node..."

source $HOME/catkin_ws/devel/setup.bash
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml
