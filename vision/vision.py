# Main license plate detection file
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from transform import four_point_transform
from scipy.signal import convolve2d, find_peaks

import os

from constants import *
from classifier import Classifier

def findPlates(cv_bgr_img):
    
    ## Image Segmentation

    # Convert from BGR to HSV
    cv_hsv_img = cv2.cvtColor(cv_bgr_img, cv2.COLOR_BGR2HSV)
    cv_rgb_img = cv2.cvtColor(cv_hsv_img, cv2.COLOR_HSV2RGB)

    # Correct value of HSV
    cv_hsv_img[:,:,2] = 255
    
    # Define ranges to filter
    light_blue = (0,200,255)
    dark_blue = (40,230,255)

    light_green = (40,0,255)
    dark_green = (60,255,255)

    light_yellow = (70,200,255)
    dark_yellow = (90,255,255)

    # Make a mask to pick out each color range
    blue_mask = cv2.inRange(cv_hsv_img, light_blue, dark_blue)
    green_mask = cv2.inRange(cv_hsv_img, light_green, dark_green)
    yellow_mask = cv2.inRange(cv_hsv_img, light_yellow, dark_yellow)

    # combine all of the masks into a single mask
    mask = blue_mask + green_mask + yellow_mask

    # Blur the mask to reduce jagged edges and make contouring more reliable
    mask = cv2.GaussianBlur(mask, (5,5), 1)

    # Use the mask to find car contours
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Search for child contours. Group pairings that have the same parent
    plate_pairs = list()    
    if hierarchy is not None:
        for parent in hierarchy[0]:
            plate_pair = list()
            if parent[2] != -1: # Look for contours with children
                # First plate is the direct child of the parent
                child_idx = parent[2]
                child = hierarchy[0, child_idx]

                # Get its bounding box
                bounds = checkSize(child_idx, contours)# TODO optimize this to use bounding box function. This is unnecessary 

                # Crop the image out if the bounding box is big enough
                if bounds[0]:
                    plate_pair.append(cropBounds(cv_rgb_img, bounds[1]))

                # Iterate through all of the other children in this hierarchy level
                while child[0] != -1:
                    child_idx = child[0] # Get the index of the next child
                    child = hierarchy[0, child_idx] # Get that child from the hierarchy

                    bounds = checkSize(child_idx, contours)

                    if bounds[0]:
                        plate_pair.append(cropBounds(cv_rgb_img, bounds[1]))

                # Correct spot/plate pairings will only have 2 results. Append good results to the list of found plates
                if len(plate_pair) == 2:
                    plate_pairs.append(plate_pair)
                
     
    ## Image persepctive transform
    corrected_pairs = list()

    # Shift the perspective on each plate so that it appears straight on
    for pair in plate_pairs:
        corrected_pair = list()
        
        # Correct the perspective of the plates
        plate1 = normalPerspective(pair[0])
        plate2 = normalPerspective(pair[1]) 

        # Some obliquely angled images will have their perspective incorrectly
        # transformed because the corner of the car will be mistake for the corner
        # of the license plate. These images will have a very "square" aspect ratio.
        # Ignore them because we cant pull any letters out of them
        if aspectRatio(plate1) < 1.4 or aspectRatio(plate2) < 1.4:
            continue

        # Threshold the images to improve contrast
        plate1 = cv2.cvtColor(plate1, cv2.COLOR_RGB2GRAY)
        plate2 = cv2.cvtColor(plate2, cv2.COLOR_RGB2GRAY) 

        _, plate1 = cv2.threshold(plate1, 70, 255, cv2.THRESH_BINARY)
        _, plate2 = cv2.threshold(plate2, 70, 255, cv2.THRESH_BINARY)



        corrected_pair.append(plate1)
        corrected_pair.append(plate2)
        corrected_pairs.append(corrected_pair)


    ## Letter Detection "chunking"
    chunked_pairs = list()
    for pair in corrected_pairs:

        # Segment each plate image into individual characters
        letters1 = getChars(pair[0])
        letters2 = getChars(pair[1])

        # The spot label has 3 characters. Plate has 4 characters. Sort by length
        if len(letters1) > len(letters2):
            unlabeled_pair = {
                "spot": letters2,
                "plate": letters1
            }
        else:
            unlabeled_pair = {
                "spot": letters1,
                "plate": letters2
            }

        # Sometimes there are errors in the chunking. If the lengths dont match
        # up to what they should be ignore the set of images
        if len(unlabeled_pair["spot"]) != 3 or len(unlabeled_pair["plate"]) != 4: #TODO change 2 to 3
            continue
        else:
            # Save good image sets
            chunked_pairs.append(unlabeled_pair)

    for pair in corrected_pairs:
        cv2.imshow("plate1", pair[0])
        cv2.imshow("plate2", pair[1])

    # for pair in chunked_pairs:
    #     cv2.imwrite("/home/fizzer/enph353/full.png", cv_rgb_img)
    #     for key, value in pair.items():
    #         i=0
    #         for img in value:
    #             i = i+1
    #             cv2.imshow(key + "{}".format(i), img*255)
    #             cv2.imwrite("/home/fizzer/enph353/{}_crop{}.png".format(key,i), img*255)

    cv2.imshow("Image window", cv_rgb_img) 
        
    cv2.waitKey(3)

    return chunked_pairs



def checkSize(idx, contours):
    child_contour = contours[idx]

    # Crop the bounding box of the first plate
    child_bound = cv2.boundingRect(child_contour)
    x0 = child_bound[0]
    y0 = child_bound[1]
    width = child_bound[2]
    height = child_bound[3]

    #print(width/height)

    if float(width)/float(height) > 1.1 and width*height > MIN_IMAGE_SIZE:
        return (True, child_bound)
    else:
        return (False, None)

# Returns the image crop defined by bounds plus a buffer area
def cropBounds(img, bounds):
    buffer = 5
    x0 = bounds[0] - buffer
    y0 = bounds[1] - buffer
    width = bounds[2] + 2*buffer
    height = bounds[3] + 2*buffer

    return img[y0:y0+height, x0:x0+width]

# Gives a straight on view of the license plate
def normalPerspective(img):
    # Convert image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Up V to max to get rid of the letters
    hsv_img[:,:,2] = 255

    # Save image as grayscale
    gray = cv2.cvtColor(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2GRAY)
    corners = cv2.cornerHarris(gray, 2,3,0.04)

    # Crop the corners into 4 quarters and find the max in each section
    height, width = corners.shape
    mid_height = int(round(height/2))
    mid_width = int(round(width/2))

    top_left =  corners[0:mid_height, 0:mid_width]
    bot_left =  corners[mid_height:height, 0:mid_width]
    top_right = corners[0:mid_height, mid_width:width]
    bot_right = corners[mid_height:height, mid_width:width]

    # Find the coordinate of the maximum in each quadrant
    top_left_idx = np.unravel_index(top_left.argmax(), top_left.shape)
    bot_left_idx = np.unravel_index(bot_left.argmax(), bot_left.shape) + np.array([mid_height,0])
    top_right_idx = np.unravel_index(top_right.argmax(), top_right.shape) + np.array([0,mid_width])
    bot_right_idx = np.unravel_index(bot_right.argmax(), bot_right.shape) + np.array([mid_height,mid_width])

    # Transform takes (x,y) coordinate format. Currently have (y,x) format so flip each corner
    top_left_idx = np.flip(top_left_idx)
    top_right_idx = np.flip(top_right_idx)
    bot_left_idx = np.flip(bot_left_idx)
    bot_right_idx = np.flip(bot_right_idx)

    return four_point_transform(img, np.array([top_left_idx, top_right_idx, bot_left_idx, bot_right_idx]))

# Returns the width/height aspect ratio of the image
def aspectRatio(img):
    height = img.shape[0]
    width = img.shape[1]

    return float(width)/float(height)

# Takes a plate image and returns a list of cropped characters on the plate
def getChars(img):

    height, width = img.shape

    img = removeBars(img)

    # This operation is a fast way of getting the convolution response of a vertical line
    response = np.sum(img, axis=0)/ height

    # Find the peaks of the convolution response and use them to crop letters.
    peaks = find_peaks(response, 240, plateau_size=1)

    crops = list()
    for i in range(len(peaks[0]) - 1): 
        left = peaks[1]['right_edges'][i] - 1
        right = peaks[1]['left_edges'][i+1] + 1
        crops.append(img[:,left:right])

    # Resize the images
    resized_crops = list()
    for img in crops:
        img = img/255
        img = cv2.resize(img, (INPUT_WIDTH,INPUT_HEIGHT))
        resized_crops.append(img.reshape(INPUT_HEIGHT, INPUT_WIDTH,1))

    return np.array(resized_crops)

def getModel():
    json_file = open('/home/fizzer/enph353/vision/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    conv_model = model_from_json(loaded_model_json)

    # Create a callback to save a checkpoint file after each epoch
    checkpoint_path = "/home/fizzer/enph353/vision/training/cp.ckpt"
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_path)

    conv_model.load_weights(checkpoint_path)    

    return conv_model

def removeBars(img):

    height, width = img.shape

    # This operation is a fast way of getting the convolution response of a horizontal line
    response = np.sum(img, axis=1)/ width

    # Mark every row with a low response
    to_remove = response < 100

    # Change all marked rows to be all white
    img[to_remove,:] = 255

    return img

def readPlate(plate_imgs, letter_cnn, number_cnn):
    letters = plate_imgs[0:2]
    numbers = plate_imgs[2:4]
    return letter_cnn.predict(letters) + number_cnn.predict(numbers)

def readSpot(spot_imgs, binary_cnn, number_cnn):
    return "P" + binary_cnn.predict(spot_imgs[1:2]) + number_cnn.predict(spot_imgs[2:3])


#img = cv2.imread('gy2.png')
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#findPlates(img)

