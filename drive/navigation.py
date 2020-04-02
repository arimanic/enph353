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
        # robot.viewer.show()

        robot.motors.set_wheel_motors(50, -50)
        time.sleep(1)
        robot.motors.set_wheel_motors(60,60)
        time.sleep(1)
        robot.motors.set_wheel_motors(-60,60)
        time.sleep(1)
        robot.motors.set_wheel_motors(40,40)
        time.sleep(1)

        while True:
            img_pil = robot.camera.latest_image.raw_image
            img_live = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2GRAY)
            ret, frame = cv2.threshold(img_live, 240, 255, cv2.THRESH_BINARY)

            x,y = frame.shape
            # print("x: " + str(x) + "   y: " + str(y))
            grid = np.indices((1,y-300))
            grid = grid + 300

            edge_right = np.sum((frame[x-1,300:y])*grid[1])/np.sum((frame[x-1,300:y]))
            # print("edge: " + str(edge_right))
            
            if(edge_right < 495):
                print("turning small L")
                robot.motors.set_wheel_motors(36, 40)

            elif(edge_right > 510):
                print("turning small R")
                robot.motors.set_wheel_motors(40, 36)

            else:
                robot.motors.set_wheel_motors(40, 38)

            # print(turn)
            # # TURN HARD RIGHT
            # if(np.isnan(edge_right) and turn < 2):
            #     # print("HARD TURN RIGHT")
            #     robot.motors.set_wheel_motors(60,60)
            #     time.sleep(1)
            #     robot.motors.set_wheel_motors(43,-43)
            #     time.sleep(2)
            #     robot.motors.set_wheel_motors(0,0)
            #     time.sleep(1)
            #     turn += 1   

            # # TURN HARD LEFT
            # elif(np.isnan(edge_right) and turn >= 2):
            # #    print("HARD TURN LEFT")
            #     robot.motors.set_wheel_motors(0,0)
            #     time.sleep(10)
            #     robot.motors.set_wheel_motors(60,60)
            #     time.sleep(1)
            #     robot.motors.set_wheel_motors(-40,40)
            #     time.sleep(2)
            #     turn += 1   


if __name__ == "__main__":
    main()

