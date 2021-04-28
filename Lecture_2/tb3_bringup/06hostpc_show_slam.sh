#!/bin/bash

source ./00config.sh

echo "Starting SLAM rviz node..."

source $HOME/catkin_ws/devel/setup.bash
rosrun rviz rviz -d `rospack find turtlebot3_slam`/rviz/turtlebot3_slam.rviz