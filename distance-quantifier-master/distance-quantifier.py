################################################################################

# Example : calculates distances to nearby objects and displays this
#  - performs YOLO (v3) object detection from a dataset of image files
#  - performs stereo disparity and calculates a disparity map from two rectified stereo images
#  - calculates the disparity of each object using this map
#  - uses this information to calculate distance of the object and display

# Author: Calum Johnston, calum.p.johnston@durham.ac.uk
# Credit to: Toby Breckon, toby.breckon@durham.ac.uk

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Implements the You Only Look Once (YOLO) object detection architecture decribed in full in:
# Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
# https://pjreddie.com/media/files/papers/YOLOv3.pdf

# To use first download the following files:
    
# https://pjreddie.com/media/files/yolov3.weights
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true

################################################################################

import cv2
import numpy as np
import os
import dense_disparity_detection as dis
import yolo_detection as yolo
from collections import Counter

################################################################################
# === DRAWING + DISTANCE CALCULATION FUNCTIONS === #
################################################################################

#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# box: image parameters for object detection
# colour: to draw detection rectangle in
def drawPred(image, class_name, confidence, box, colour):
    # Get box coordinates and distance
    left = box[0]; top = box[1]
    width = box[2]; height = box[3]
    distance = box[4]
    right = left + width
    bottom = top + height

    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2fm (%.2f)' % (class_name, distance, confidence)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    
#####################################################################
# Gets the distance of an object  in an image based on the area around it
# disparity_scaled: scaled version of the disparity map
# box: image parameters for object detection
# radius: radius for amount of pixels to be calculated from the centre
# recurse: boolean variable to determine whether we have called the function from within the function
def getBoxDistance(disparity_scaled, box):
    # Get information about box
    # NOTE: left must be at least 128
    left = max(box[0], 128); top = box[1]
    width = box[2]; height = box[3]

    # setup camera variables 
    f = camera_focal_length_px
    B = stereo_camera_baseline_m     

    # get the centre coordinates of the object
    centre_point_X = int(left+(width/2))
    centre_point_Y = int(top+(height/2))

    # setup variables:
    # centre_to_edge = determines size of the gaussian kernel used (total x/y dimension = 2*centre_to_edge+1)
    # nonZeroCount = used to ignore any 0 disparity values
    # sigma = used as sigma calculation in generating kernel
    # arr = used to store disparity values for the central pixels of the object (same size as kernel)
    # centre_to_edge = determines size of the gaussian kernel used (total x/y dimension = 2*centre_to_edge+1)
    centre_to_edge = 1
    sigma = 1
    nonZeroCount = 0
    arr = []
    
    # IMPORTANT PART - For smaller boxes surrounding objects we typically need to analysis a larger
    # percentage of it's pixels to get a more accurate distance, and vice versa for larger boxes.
    # The size is determined by cente_to_edge, so we edit that here
    # centre_to_edge = determines size of the gaussian kernel used (total x/y dimension = 2*centre_to_edge+1)
    if(min(width, height) < 30):
        centre_to_edge = int(min(width, height) / 2)
    elif(min(width, height) < 40):
        centre_to_edge = int(min(width, height) / 3)
    elif(min(width, height) < 50):
        centre_to_edge = int(min(width, height) / 4)
    elif(min(width, height) < 75):
        centre_to_edge = int(min(width, height) / 8)
    elif(min(width, height) < 100):
        centre_to_edge = int(min(width, height) / 16)
    else:
        centre_to_edge = int(min(width, height) / 32)

    # get the gaussian kernel for defined dimensions
    kernel = gkern((centre_to_edge * 2) + 1, sigma)

    # get the disparity values for the central block of pixels and create a 2D list of them
    for x in range(centre_point_X - centre_to_edge, centre_point_X + centre_to_edge + 1):
        newLst = []
        for y in range(centre_point_Y - centre_to_edge, centre_point_Y + centre_to_edge + 1):
            if(x > 0 and x < disparity_scaled.shape[1] and y > 0 and y < disparity_scaled.shape[0]):
                newLst.append(disparity_scaled[y, x])
                if(disparity_scaled[y,x] > 0):
                    nonZeroCount += 1
        arr.append(newLst)

    # calculate the average distance of the object
    # - Convolute the kernel with the arr
    # - Sum the elements of the produced matrix
    # - Divide by the number of non-zero disparity values
    if(nonZeroCount > 0):
        aver_Dis = sum(np.convolve(np.ndarray.flatten(kernel), np.ndarray.flatten(np.array(arr)))) / nonZeroCount
        if(aver_Dis > 0):
            averageDistance = (f * B) / aver_Dis
            return averageDistance

    # If no disparity values > 0 were found, simply return 0 (it will hence not be drawn)
    return 0


#####################################################################
# Returns a gaussian kernel to the image of size n by n
# size = the distance from the centre of the kernel to the edge
# box = the sigma value for calculating the kernel
def gkern(size=3,sigma=1):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)
    






################################################################################
# === Initialisation of the required variables === #
################################################################################

################################################################################
# Initialisation of dataset

# where is the data?
master_path_to_dataset = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\z-resources\TTBB-durham-02-10-17-sub10";
master_path_to_yolo_resources = os.path.dirname(os.path.realpath(__file__)) + "\yolo resources"; 
directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images"; 

# resolve full directory location of data set for left / right images
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left));

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683_L for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = "1506942604.475373"; 


################################################################################
# Initialisation of image parameters 

# fixed camera parameters for this stereo setup (from calibration)
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;


################################################################################
# Initialisation of YOLO parameters

# dataset defining what objects to include
dataset = ["car", "bus", "truck", "person", "bicycle", "motorbike"]







################################################################################
# === Main Program - Calculations === #
################################################################################

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)
    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # check the file is a PNG file (left) and check a correspondoning right image actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        ################################################################################
        # Setup of windows, images, timers, etc
        ################################################################################
        # start a timer and convert to ms. (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # read left and right images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)


        ################################################################################
        # PRE-PROCESSING
        ################################################################################
        # perform edge enhancement on the image using a generated kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        im = cv2.filter2D(imgL, -1, kernel)
        
        
        ################################################################################
        # YOLO Object Detection 
        ################################################################################
        # Gets the information about objects
        classIDs, classes, confidences, boxes = yolo.yolo(imgL)
        classIDs1, classes1, confidences1, boxes1 = yolo.yolo(im)

        # Combine the two detections to produce total objects
        addBox = True
        count = 0;
        for box1 in boxes1:
            addBox = True
            for box in boxes:
                if (box1[0] - 10 <= box[0] <= box1[0] + 10) and (box1[1] - 10 <= box[1] <= box1[1] + 10) and (box1[2] - 10 <= box[2] <= box1[2] + 10) and (box1[3] - 10 <= box[3] <= box1[3] + 10):
                    addBox = False
                    break;
            if(addBox == True):
                classIDs.append(classIDs1[count])
                confidences.append(confidences1[count])
                boxes.append(box1)
            count += 1 
            

        ################################################################################
        # Stereo Disparity & Distance Calculation
        ################################################################################
        # Gets the disparity map for the left and right image
        # NOTE: Pre-processing of images done in here
        disparity = dis.disparity(imgL, imgR)


        ################################################################################
        # Resulting Distance Calculations + Drawing 
        ################################################################################
        # variable that stores the minimum distance drawn each time
        min_distance = 1000
        
        # for each box (representing one object) get it's distance
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            if(classes[classIDs[detected_object]] in dataset):  # check it's an object we want to track
                distance = getBoxDistance(disparity, box)
                if not(box[1] < 480 < box[1] + box[3] and box[0] < 512 < box[0] + box[2]) and (distance != 0):
                    box.append(distance)
                    drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], box, (255, 178, 50))
                    # update minimum distance
                    if(distance < min_distance):
                        min_distance = distance

        #################################################################################
        # Output of image
        #################################################################################
        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        #Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        #label = 'Inference time: %.2f ms' % (stop_t)
        #cv2.putText(imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # print file names and minimum distance to standard output
        print(filename_left)
        print(filename_right , " : nearest detected scene object (%.2fm)" % (min_distance))

        # display image showing object detection w/distances
        cv2.imshow("Object Detection v1",imgL)
    
        # wait for user input till next image
        cv2.waitKey()
