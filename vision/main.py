#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import cv_bridge

def callback(data):

    bridge = cv_bridge.CvBridge()

    # Load the image from the message
    try:
        cv_image =  bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    except cv_bridge.CvBridgeError as e:
        print(e)

    # Get dimensions and locate the target point
    height, width, channels = cv_image.shape

    # TODO add license plate detection code here
    
    # Generate a move command based on the robots position relative to the line
    # TODO Replace this with driving control code
    move = Twist()
    move.linear.x = 0.15

    # Publish the move command
    pub = rospy.Publisher('/cmd_vel', Twist, 
        queue_size=1)

    pub.publish(move)

    # Show the Camera Feed
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)
    
def driver():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    image_topic = "/rrbot/camera1/image_raw/"

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, callback)
   
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


driver()

rate = rospy.Rate(2)
while not rospy.is_shutdown():
    if __name__ == '__main__':
        driver()
        rate.sleep()