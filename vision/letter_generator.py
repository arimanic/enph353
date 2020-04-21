#!/usr/bin/env python

import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw

from constants import *

path = os.path.dirname(os.path.realpath(__file__)) + "/"

NUMBER_OF_PLATES = 10000
path = NUMBER_DATA_PATH

for i in range(0, NUMBER_OF_PLATES):

    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += (random.choice(string.ascii_uppercase))

    # Pick two random numbers
    num = randint(0, 99)
    plate_num = "{:02d}".format(num)

    all_rand = plate_num
    one_rand = all_rand[np.random.randint(0,2)]

    # Write plate to image
    blank_plate = np.full((INPUT_HEIGHT,INPUT_WIDTH, 3), 255, dtype=np.uint8)
   # blank_plate = cv2.imread(path+'blank_letter.png')


    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)

    x_offset = np.random.randint(-2,2)
    y_offset = np.random.randint(-2,2)

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", np.random.randint(56,60))
    draw.text((x_offset, y_offset),one_rand, (0,0,0), font=monospace)

    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)

    # Randomly pixelate the image
    scale = np.random.randint(1, 3)
    height,width, _ = blank_plate.shape

    w,h = (int(width/scale), int(height/scale))

    # Resize input to "pixelated" size
    temp = cv2.resize(blank_plate, (h,w), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    blank_plate = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


    # Apply a random amount of blur to the images
    for i in range(np.random.randint(0, high=1)):
        blank_plate = cv2.GaussianBlur(blank_plate, (5,5), 1)

    # Write license plate to file
    cv2.imwrite(os.path.join(path, 
                                "{}_{}.png".format(one_rand, np.random.randint(0,1000000))),
                blank_plate)

   # cv2.imshow("new",blank_plate)
   # cv2.waitKey()
