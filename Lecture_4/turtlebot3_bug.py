#! /usr/bin/env python2.7

# We may have to launch the script with python2 <script_name>, depending on the python version that we used to build ROS packages
# This will solve "ImportError: dynamic module does not define module export function (PyInit__tf2)"

import math
from threading import Thread
import time
from math import sqrt, pow, pi, atan2

import rospy
import tf
import numpy as np
from geometry_msgs.msg import Twist, Point, Quaternion
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

class Bug():

    TRESHOLD = 0.5
    THRESHOLD_GOAL = 0.1

    def __init__(self):
        rospy.init_node('turtlebot3_bug')
        rospy.on_shutdown(self.stop_turtlebot)
        self.goal_x, self.goal_y, self.goal_reached = 2.2, 0.0, 0.5     # i.e., to the other end of the wall

        self.lin_vel, self.ang_vel = 0.15, 0.13   # ang_vel is in rad/s, so we rotate 5 deg/s
        self.safe_stop_dist = 0.4 + 0.05    # stop distance + lidar error 

        self.r = rospy.Rate(1000)

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
        
    def stop_turtlebot(self): 
        self.cmd_pub.publish(Twist())

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

    def move_forward(self): 
        print("Move forward")
        move_cmd = Twist()
        move_cmd.linear.x = 0.2
        self.cmd_pub.publish(move_cmd)

    def stop(self): 
        self.cmd_pub.publish(Twist())
    
    def rotate_left(self, angle): 
        _ ,turtle_angle = self.get_odom()

        while(turtle_angle < angle): 
            move_cmd = Twist()
            move_cmd.angular.z = 0.2
            self.cmd_pub.publish(move_cmd)
            _ ,turtle_angle = self.get_odom()

        #i stop the robot
        self.cmd_pub.publish(Twist())

    def rotate_right (self, angle): 
        _ ,turtle_angle = self.get_odom()

        while(turtle_angle > angle): 
            move_cmd = Twist()
            move_cmd.angular.z = -0.2
            self.cmd_pub.publish(move_cmd)
            _ ,turtle_angle = self.get_odom()

        self.cmd_pub.publish(Twist())

    def rotate_towards_goal(self): 
        print("Rotating towards the goal") 
        pose ,turtle_angle = self.get_odom()

        _, goal_angle = self.get_goal_info(pose)
        
        #rotate right, example: turtle_angle = 0, goal_angle = -90
        theta = goal_angle
        epsilon = abs(goal_angle - turtle_angle)
        if epsilon > 0.1: 
            if theta > 0: 
                self.rotate_left(theta)
            else: 
                self.rotate_right(theta)

    def can_move_forward(self): 
            scan = self.get_scan()
            if scan[3] > self.TRESHOLD and scan[4] > self.TRESHOLD and scan[5] > self.TRESHOLD and scan[6] > self.TRESHOLD: 
            # if scan[5] > self.TRESHOLD: 
                return True 
            return False 

    def is_right_clear(self): 
            scan = self.get_scan()
            if scan[0] > self.TRESHOLD and scan[1] > self.TRESHOLD and scan[2] > self.TRESHOLD:
                return True 
            return False 

    def is_left_clear(self): 
            scan = self.get_scan()
            if scan[7] > self.TRESHOLD and scan[8] > self.TRESHOLD and scan[9] > self.TRESHOLD and scan[10] > self.TRESHOLD:
                return True 
            return False 

    def rotate_left_till_forward_clear(self): 
        print("Rotating left till forward is clear")
        if not self.can_move_forward(): 
            move_cmd = Twist()
            move_cmd.angular.z = 0.2
            self.cmd_pub.publish(move_cmd)

        while not self.can_move_forward(): 
            pass 
        
        self.stop()

    def move_forward_till_obstacle_clean(self): 
        print("Move forward till clear from obstacle")
        while not self.is_right_clear(): 
            self.stop()
            while not self.can_move_forward():
                self.rotate_left_till_forward_clear()
            
            while self.can_move_forward() and not self.is_right_clear():
                print("Move forward to avoid obstacle to the right")
                self.move_forward()

        self.stop()

    def bug0(self):
        turtle_pos, turtle_angle = self.get_odom()    
        distance_to_goal, angle_to_goal= self.get_goal_info(turtle_pos)    

        print("Goal pos: ", self.goal_x, self.goal_y)
        print("Turtle pos: ", turtle_pos)
        print("Turtle angle: ", turtle_angle)
        print("Angle to goal: ", angle_to_goal)
        print("Distance to goal: ", distance_to_goal)

        while(distance_to_goal > self.THRESHOLD_GOAL): 
            distance_to_goal, angle_to_goal = self.get_goal_info(turtle_pos)
            print("Laser scan: ",self.get_scan())
            print("Distance to goal: ", distance_to_goal)
            print("Goal pos: ", self.goal_x, self.goal_y)
            print("Turtle pos: ", turtle_pos)
            print("Angle to goal: ", angle_to_goal)
            
            self.rotate_towards_goal()

            while(self.can_move_forward() and distance_to_goal > self.THRESHOLD_GOAL): 
                turtle_pos, turtle_angle = self.get_odom() 
                distance_to_goal, angle_to_goal = self.get_goal_info(turtle_pos)
                self.rotate_towards_goal()
                self.move_forward()
                pass 

            self.stop()

            if(distance_to_goal <= self.THRESHOLD_GOAL):
                break 

            self.rotate_left_till_forward_clear() 
            self.move_forward_till_obstacle_clean()


    def bug(self):
        # while(True): 
        #     pos, angle = self.get_odom()
        #     print("Goal info: ", self.get_goal_info(pos))
        #     print("Turtle angle pose info: ", self.get_odom())
        #     self.r.sleep()
        # print("Laser scan: ", self.get_scan())
        self.bug0()
        self.stop()


def main():
    bug = Bug()

if __name__ == '__main__':
    main()
