#!/bin/bash

# ------------------------------------------------------------------------------
# Write here local pc ip address and raspberry ip address.
export RASPBERRY_IP="157.27.193.127"
export LOCAL_IP="157.27.198.72"
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Write here ubuntu raspberry username
export RASPBERRY_USERNAME="ubuntu"
export RASPBERRY_PASSWORD="maestro123"
# ------------------------------------------------------------------------------

export TURTLEBOT3_MODEL=waffle_pi
export ROS_HOSTNAME=${LOCAL_IP}
export ROS_MASTER_URI=http://${LOCAL_IP}:11311

