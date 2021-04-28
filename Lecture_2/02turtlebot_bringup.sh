#!/bin/bash

source ./config.sh

./00sendconfiguration.sh

# Execute node
echo "Starting ros on turtlebot"
ssh ${RASPBERRY_USERNAME}@${RASPBERRY_IP} "bash -c 'chmod +x raspberry_config.sh; source ./raspberry_config.sh; roslaunch turtlebot3_bringup turtlebot3_robot.launch'"

