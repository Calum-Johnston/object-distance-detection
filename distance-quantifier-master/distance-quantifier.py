################################################################################

# Example : calculates distances to nearby objects and displays this
#  - performs YOLO (v3) object detection from a dataset of image files
#  - calculates the disparity from the dataset of image files 
#  - uses this information to display the required data

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
import yolo_detection as yolo
import dense_disparity_detection as dis

################################################################################
# === YOLO Object Detection Functions === #
################################################################################

################################################################################
# dummy on trackbar callback function
def on_trackbar(val):
    return

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
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

#####################################################################
# Gets the distance of an object  in an image based on the area around it
# disparity_scaled: scaled version of the disparity map
# box: image parameters for object detection
def getBoxDistance(disparity_scaled, box):
    # Get information about box
    left = box[0]; top = box[1]
    width = box[2]; height = box[3]
    right = left + width
    bottom = top + height

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    totalDisparity = 0
    totalCount = 0

    # Trim the box to hopefully isolate object and reduce background noise
    right -= int((right - left) * 0.4)
    left += int((right - left) * 0.4)
    bottom -= int((bottom - top) * 0.4)
    top += int((bottom - top) * 0.4)

    # Loops through all box pixels to produce an average disparity
    for x in range(left, right):
        for y in range(top, bottom):
            if(y < imgL.shape[0] and x < imgL.shape[1]):
                if(disparity_scaled[y, x] > 0):
                    currentDisparity = disparity_scaled[y, x]
                    totalDisparity = totalDisparity + currentDisparity
                    totalCount += 1
    if(totalCount > 0):
        averageDisparity = totalDisparity / totalCount
        averageDistance = (f * B) / averageDisparity
        return averageDistance 
    return 0       






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




################################################################################
# Initialisation of image parameters 

# fixed camera parameters for this stereo setup (from calibration)
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_cent4re_w = 474.5;



################################################################################
# Define display window name + trackbar

windowName = 'Object distance detection'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName , 0, 100, on_trackbar)






################################################################################
# === Main Program - Calculations === #
################################################################################

for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames
    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a correspondoning right image actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        ################################################################################
        # Setup of windows, images, timers, etc
        ################################################################################
        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # create window by name (as resizable)
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

        # read left and right images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)


        ################################################################################
        # YOLO Object Detection 
        ################################################################################
        # Calculate threshold to remove the bounding boxes with low confidence (based on trackbar)
        confThreshold = cv2.getTrackbarPos(trackbarName,windowName) / 100

        # Gets the information about objects
        classIDs, classes, confidences, boxes, t = yolo.yolo(imgL, confThreshold)


        ################################################################################
        # Stereo Disparity Calculations + Function Calls
        ################################################################################
        # Gets the disparity map for the left and right image
        disparity = dis.disparity(imgL, imgR)


        ################################################################################
        # Resulting Distance Calculations + Drawing 
        ################################################################################
        # for each box (representing one object) get it's distance
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            box.append(getBoxDistance(disparity, box))
        
        # sorts the boxes as to draw the closest box first
        boxes.sort(key = lambda box: box[4], reverse = True)

        # draw the boxes around the objects (label with data)
        for box in boxes:
            drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], box, (255, 178, 50))


        #################################################################################
        # Output of image
        #################################################################################
        #Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image
        cv2.imshow(windowName,imgL)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # wait for user input till next image
        cv2.waitKey()