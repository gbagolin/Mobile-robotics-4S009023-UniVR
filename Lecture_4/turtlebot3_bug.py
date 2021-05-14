#! /usr/bin/env python2.7

# We may have to launch the script with python2 <script_name>, depending on the python version that we used to build ROS packages
# This will solve "ImportError: dynamic module does not define module export function (PyInit__tf2)"

import math
import time
from math import sqrt, pow, pi, atan2

import rospy
import tf
import numpy as np
from geometry_msgs.msg import Twist, Point, Quaternion
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

class Bug():

    def __init__(self):
        self.goal_x, self.goal_y, self.goal_reached = 2.2, 0.0, 0.5     # i.e., to the other end of the wall

        self.lin_vel, self.ang_vel = 0.15, 0.13   # ang_vel is in rad/s, so we rotate 5 deg/s
        self.safe_stop_dist = 0.4 + 0.05    # stop distance + lidar error 

        self.r = rospy.Rate(10)

        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)     # subscribe as publisher to cmd_vel for velocity commands
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
         
        self.bug()
        

    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rot = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return Point(*trans), np.rad2deg(rot[2])

    def get_scan(self):
        scan_topic = rospy.wait_for_message('scan', LaserScan)    # wait for a scan msg to retrieve recent sensor's reading
       
        # back_right, right, front, left, back_left
        #scan_filter = [scan.ranges[0], scan.ranges[1], scan.ranges[5], scan.ranges[-2], scan.ranges[-1]]
        scan = [scan_topic.ranges[i] for i in range(len(scan_topic.ranges))]
        for i in range(len(scan)):       # cast limit values (0, Inf) to usable floats
            if scan[i] == float('Inf'):
                scan[i] = 3.5
            elif math.isnan(scan[i]):
                scan[i] = 0
        
        return scan

    def get_goal_info(self, tb3_pos):
        distance = sqrt(pow(self.goal_x - tb3_pos.x, 2) + pow(self.goal_y - tb3_pos.y, 2))  # compute distance wrt goal
        heading = atan2(self.goal_y - tb3_pos.y, self.goal_x- tb3_pos.x)    # compute heading to the goal in rad
        
        return distance, np.rad2deg(heading)     # return heading in deg
        
    def bug(self):
        pass
        # Implement here the bug algorithm
        # move toward goal
        # if hit obs:
            # turn left
            # move around obs untile path to goal is clear

    def move(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0.2
        move_cmd.angular.z = 0.15
        self.cmd_pub.publish(move_cmd)

def main():
    rospy.init_node('turtlebot3_bug')
    try:
        bug = Bug()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
