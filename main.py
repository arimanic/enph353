#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import cv_bridge

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

import time


from constants import *
import vision
from classifier import Classifier
from collections import Counter

def callback(data):

    bridge = cv_bridge.CvBridge()

    # Load the image from the message
    try:
        cv_image =  bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    except cv_bridge.CvBridgeError as e:
        print(e)

    # Find the plates in the image feed
    plate_pair = vision.findPlates(cv_image)

    # Classify each character
    if len(plate_pair) != 0:
        for pair in plate_pair:
            spot = vision.readSpot(pair["spot"], binaryClassifier, numberClassifier)
            plate = vision.readPlate(pair["plate"], letterClassifier, numberClassifier)

            if spot not in results:
                results[spot] = Counter()

            # Put the plate count in the results
            results[spot][plate] += 1
    
    # Publish the move command
    pub = rospy.Publisher('/cmd_vel', Twist, 
        queue_size=1)

    #pub.publish(move)


    if int(time.time() * 10) % 100 == 0:
        print("Plates Found:")
        print({k : v.most_common(1)[0][0] for k,v in results.items()})

    
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


letterClassifier = Classifier("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
letterClassifier.loadWeights(LETTER_WEIGHTS_PATH)

numberClassifier = Classifier("0123456789")
numberClassifier.loadWeights(NUMBER_WEIGHTS_PATH)

binaryClassifier = Classifier('01')
binaryClassifier.loadWeights(BINARY_WEIGHTS_PATH)

results = dict()

driver()

rate = rospy.Rate(2)
while not rospy.is_shutdown():
    if __name__ == '__main__':
        driver()
        rate.sleep()