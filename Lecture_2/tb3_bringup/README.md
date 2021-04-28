# TurtleBot3 Bringup
Scripts for the bringup of the TurtleBot3.

## Requirements
* A wireless network
* A configured robot as described [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)
* Ubuntu on raspberry must connect automatically to a wireless network 

## Dynamic IP
With a dynamic IP address (as in the university's network) check the raspberry IP using a monitor (or a serial terminal like picocom/putty). Use ifconfig command to get the ip

## Usage
1. Connect host pc to the same wireless network of the TurtleBot3
2. Edit 00config.sh with the correct ip addresses and ubuntu raspberry username
3. Turn on TurtleBot3 and wait ~45secs for ubuntu to start
4. Run 01hostpc_bringup.sh in a terminal of the host pc
5. In another terminal run 02turtlebot_bringup.sh (typing raspberry password twice)

Now turtlebot is ready and working (lidar sensor, if connected should start rotating).

## Teleoperation
Keyboard teleoperation is available by running 03hostpc_keyboard_teleop.sh

## SLAM
With the system running, it is possible to run SLAM node to create map by running 05hostpc_slam.sh
When script is closed, map is saved in current directory.
To see the built map real-time, launch 06hostpc_show_slam.sh with slam running.

## Shutdown system
To poweroff the TurtleBot3, close the scripts with CTRL+C in inverse order.
Before shutting down it is important to poweroff the raspberry operating system by running poweroff.sh and waiting ~30secs before removing power!