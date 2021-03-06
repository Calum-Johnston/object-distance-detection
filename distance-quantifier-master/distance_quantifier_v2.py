################################################################################

# Example : calculates distances to nearby objects and displays this
#  - performs YOLO (v3) object detection from a dataset of image files
#  - for each object performs FLANN matching using ORB
#  - the disparity at each feature point of the object is calculated
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
import sparse_disparity_detection as dis
import yolo_detection as yolo

################################################################################
# Specify directories containing images & YOLO files here
master_path_to_dataset = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\z-resources\TTBB-durham-02-10-17-sub10";
master_path_to_yolo_resources = os.path.dirname(os.path.realpath(__file__)) + "\yolo resources"; 







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
# Gets the distance of an object in an image based on the area around it
# box: image parameters for object detection
# imgL: left image in which the object will be cropped from
# imgR: right image to compare the object against for feature detection
def getBoxDistance(box, object_imgL, imgR):
    # Get information about box
    left = box[0]; top = box[1]
    width = box[2]; height = box[3]

    # parameters to determine whether the object area has been increased
    ext = 0

    # if box is too small features won't be detected properly, so we increase it's size 
    if(height < 50 or width < 50):
        top -= 50; height += 100
        left -= 50; width += 100
        ext = 50

    # Get details of camera, to be used to calculate distance
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    # Ensure box isn't out of bounds of the image
    top = max(0, top)
    left = max(0, left)


    # Crop left image to isolate object
    # Crop right image to only incorporate possible matching features
    # (Images have already been rectified, so discard other y values)
    cropImgL = imgL[top:top+height, left:left+width]
    cropImgR = imgR[top:top+height, 0:imgR.shape[1]]

    # Gets the distance of the object using the disparity of only that object
    distance = dis.disparity(cropImgL, cropImgR, f, B, 20, left, ext)

    return distance
    






################################################################################
# === Initialisation of the required variables === #
################################################################################

################################################################################
# Initialisation of dataset

# where is the data?
directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images"; 

# resolve full directory location of data set for left / right images
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left));

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683_L for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = "1506942474.483193"


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
        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # read left and right images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # copy imgL (while it's unused) - to be used for drawing boxes on later
        result_imgL = imgL.copy()

        
        ################################################################################
        # Pre-processing of images (to be used later)
        ################################################################################
        # DISPARITY CALCULATIONS
        # convert to grayscale 
        # N.B. need to do for both as both are 3-channel images
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');
        
        # setup the adaptive histogram equalisation object - CLAHE object
        # CLAHE = Constrast Limited Adaptive Histogram Equalisation
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
        
        # Perform histogram equalisation on each image
        histo_imgL = clahe.apply(grayL)
        histo_imgR = clahe.apply(grayR)

        # OBJECT DETECTION CALCULATIONS
        # Perform edge enhancement on image by sharpening
        kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
        enhanced_imgL = cv2.filter2D(imgL, -1, kernel)


        ################################################################################
        # YOLO Object Detection 
        ################################################################################
        # Gets the information about objects
        classIDs, classes, confidences, boxes = yolo.yolo(enhanced_imgL)


        ################################################################################
        # Resulting Distance Calculations + Drawing 
        ################################################################################
        # variable that stores the minimum distance drawn each time 
        min_distance = 0
    
        # for each box (representing one object) get it's distance
        # - we calculate average distance based on images as they normally are, then on histogram equalised versions
        # - this is then averaged to give the distance (from testing seemed to give more accurate distances)
        for detected_object in range(0, len(boxes)):
            if(classes[classIDs[detected_object]] in dataset): # check it's an object we want to track
                box = boxes[detected_object]
                distance_normal = getBoxDistance(box, imgL, imgR)
                distance_histo = getBoxDistance(box, histo_imgL, histo_imgR)
                if(distance_normal == 0 and distance_histo != 0):
                    box.append(distance_histo)
                elif(distance_histo == 0 and distance_normal != 0):
                    box.append(distance_normal)
                elif(distance_histo != 0 and distance_normal != 0):
                    distance_total = (distance_normal + distance_histo) / 2
                    box.append(distance_total)
                else:
                    box.append(0)

            

        # draw each box onto the image - as long as they have some distanc
        # - as long as they have some distance (box[4])
        # - as long as they're not the car from which we are gathering distance
        # - - above done by checking whether centre of car is in box
        # done in seperate loop as previous to ensure no features of the boxes are matched
        for detected_object in range(0, len(boxes)):
            if(min_distance == 0): min_distance = 1000
            box = boxes[detected_object]
            if(classes[classIDs[detected_object]] in dataset):
                if not(box[1] < 480 < box[1] + box[3] and box[0] < 512 < box[0] + box[2]) and (box[4] > 0): 
                    drawPred(result_imgL, classes[classIDs[detected_object]], confidences[detected_object], box, (255, 178, 50))
                    # update minimum distance
                    if(box[4] < min_distance):
                        min_distance = box[4]

        #################################################################################
        # Output of image
        #################################################################################
        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000
        
        #Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        label = 'Inference time: %.2f ms' % (stop_t)
        cv2.putText(result_imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # print file names and minimum distance to standard output
        print(filename_left)
        print(filename_right , " : nearest detected scene object (%.2fm)" % (min_distance))

        # display image showing object detection w/distances
        cv2.imshow("Object Detection v2",result_imgL)

        # wait for user input till next image
        cv2.waitKey()
