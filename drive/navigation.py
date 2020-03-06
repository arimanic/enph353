#!/usr/bin/env python3

# Copyright (c) 2018 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import anki_vector
from anki_vector.util import degrees
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

def main():
    turn = 0
    args = anki_vector.util.parse_command_args()
    with anki_vector.AsyncRobot(args.serial) as robot:
        robot.behavior.set_head_angle(degrees(-7.0))
        robot.behavior.set_lift_height(1)
        robot.camera.init_camera_feed()

        while True:
            img_pil = robot.camera.latest_image.raw_image
            img_live = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2GRAY)

            ret, frame = cv2.threshold(img_live, 240, 255, cv2.THRESH_BINARY)
            cv2.imshow('blob', frame)
            cv2.waitKey(1)

            x,y = frame.shape
            # print("x: " + str(x) + "   y: " + str(y))
            grid = np.indices((1,y))

            edge_right = np.sum((255-frame[x-1,:])*grid[1])/np.sum((255-frame[x-1,:]))
            print("edge: " + str(edge_right))
            
            if(edge_right > y/2 - 38 and turn == 0 ):
                print("turning small L")
                robot.motors.set_wheel_motors(36, 40)

            elif(edge_right < y/2 - 43 and turn == 0):
                print("turning small R")
                robot.motors.set_wheel_motors(40, 36)

            else:
                robot.motors.set_wheel_motors(40, 38)

            #arrived at the first turn
            if(edge_right == 319.5 and turn == 0):
                print("Lets TURN")
                robot.motors.set_wheel_motors(30,30)
                time.sleep(2)
                robot.motors.set_wheel_motors(30,-30)
                time.sleep(3)
                turn = 1

            if(turn == 1):
                robot.motors.set_head_motor(0,0)    


if __name__ == "__main__":
    main()

