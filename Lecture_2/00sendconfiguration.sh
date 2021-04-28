#!/bin/bash

source ./config.sh

# Create ros script
echo -e "
export RASPBERRY_IP=\"${RASPBERRY_IP}\"
export MASTER_IP=\"${LOCAL_IP}\"

export TURTLEBOT3_MODEL=waffle
export ROS_HOSTNAME=${RASPBERRY_IP}
export ROS_MASTER_URI=http://${LOCAL_IP}:11311

source catkin_ws/devel/setup.bash
" > raspberry_config.sh

# Send to raspberry
echo "Sending configuration file..."
scp raspberry_config.sh ${RASPBERRY_USERNAME}@${RASPBERRY_IP}:
rm raspberry_config.sh

