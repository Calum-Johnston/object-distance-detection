#####################################################################

# Example : SURF / SIFT or ORB feature point detection and matching
# from a video file specified on the command line (e.g. python FILE.py
# video_file) or from an attached web camera (default to SURF or ORB)

# N.B. use mouse to select region

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2016/17 Toby Breckon
#                       Computer Science, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based in part on tutorial at:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

#####################################################################

import cv2
import argparse
import sys
import math
import numpy as np
import os

#####################################################################
4
# check if the OpenCV we are using has the extra modules available

def extraOpenCVModulesPresent():

    # we only need to check this once and remember the result
    # so we can do this via a stored function attribute (static variable)
    # which is preserved across calls

    if not hasattr(extraOpenCVModulesPresent, "already_checked"):
        (is_built, not_built) = cv2.getBuildInformation().split("Disabled:")
        extraOpenCVModulesPresent.already_checked = ('xfeatures2d' in is_built)

    return extraOpenCVModulesPresent.already_checked

#####################################################################

keep_processing = True

# where is the data?

master_path_to_dataset = os.path.dirname(os.path.realpath(__file__)) + "\TTBB-durham-02-10-17-sub10";
master_path_to_yolo_resources = os.path.dirname(os.path.realpath(__file__)) + "\yolo resources"; 
directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images";  


#####################################################################

selection_in_progress = False # support interactive region selection

compute_object_position_via_homography = False  # compute homography H ?
transform_image_via_homography = False  # transform whole image via H
show_ellipse_fit = False # show ellipse fitted to matched points
show_detection_only = False # show detction of points only
MIN_MATCH_COUNT = 10 # number of matches to compute homography

#####################################################################

# select a region using the mouse

boxes = []
current_mouse_position = np.ones(2, dtype=np.int32)

def on_mouse(event, x, y, flags, params):

    global boxes
    global selection_in_progress

    current_mouse_position[0] = x
    current_mouse_position[1] = y

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = []
        # print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = [x, y]
        selection_in_progress = True
        boxes.append(sbox)

    elif event == cv2.EVENT_LBUTTONUP:
        # print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = [x, y]
        selection_in_progress = False
        boxes.append(ebox)

#####################################################################

# controls
print("** click and drag to select region")
print("")
print("x - exit")
print("d - display detected feature points on live image")
print("e - fit ellipse to matched points")
print("s - switch to SIFT features (default: SURF or ORB)")
print("h - compute homography H (bounding box shown)")
print("t - transform cropped image region into live image via H")

#####################################################################

# define display window name

windowName = "Photo input" # window name
windowName2 = "Feature Matches" # window name
windowNameSelection = "Selected Features"

# resolve full directory location of data set for left / right images
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left));

for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # create window by name (note flags for resizable or not)

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
        cv2.namedWindow(windowNameSelection, cv2.WINDOW_NORMAL)

        # set a mouse callback

        cv2.setMouseCallback(windowName, on_mouse, 0)
        cropped = False

        # create feature point objects

        if (extraOpenCVModulesPresent()):

            # if we have SURF available then use it (with Hessian Threshold = 400)
            # SURF features - [Bay et al, 2006 - https://en.wikipedia.org/wiki/Speeded_up_robust_features]
            feature_object = cv2.xfeatures2d.SURF_create(400)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)

        else:

            print("Sorry - SURF unavailable: falling back to ORB")

            # otherwise fall back to ORB (with Max Features = 800)
            #  ORB features - [Rublee et al., 2011 - https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF]

            feature_object = cv2.ORB_create(800)
            # if using ORB points use FLANN object that can handle binary descriptors
            # taken from: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
            # N.B. "commented values are recommended as per the docs,
            # but it didn't provide required results in some cases"

            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2

        # create a Fast Linear Approx. Nearest Neightbours (Kd-tree) object for
        # fast feature matching
        # ^ ^ ^ ^ yes - in an ideal world, but in the world where this issue
        # still remains open in OpenCV 3.1 (https://github.com/opencv/opencv/issues/5667)
        # just use the slower Brute Force matcher and go to bed
        # summary: python OpenCV bindings issue, ok to use in C++ or OpenCV > 3.1

        (major, minor, _) = cv2.__version__.split(".")
        if ((int(major) >= 3) and (int(minor) >= 1)):
            search_params = dict(checks=50)   # or pass empty dictionary
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            matcher = cv2.BFMatcher()

        while (keep_processing):

            # read left image
            frame = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)


            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount()

            # convert to grayscale

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #####################################################################

            # select region using the mouse and display it

            if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (boxes[0][0] < boxes[1][0]):

                # obtain cropped region as an image

                crop = frame[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0]].copy()
                h, w, c = crop.shape   # size of cropped region

                # if region is valid

                if (h > 0) and (w > 0):
                    cropped = True

                    # detect features and compute associated descriptor vectors

                    keypoints_cropped_region, descriptors_cropped_region = feature_object.detectAndCompute(crop,None)

                    # display keypoints on the image

                    cropped_region_with_features = cv2.drawKeypoints(crop,keypoints_cropped_region,None,(255,0,0),4)

                    # display features on cropped region

                    cv2.imshow(windowNameSelection,cropped_region_with_features)

                # reset list of boxes so we do this part only once

                boxes = []

            # interactive display of selection box

            if (selection_in_progress):
                top_left = (boxes[0][0], boxes[0][1])
                bottom_right = (current_mouse_position[0], current_mouse_position[1])
                cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 2)

            #####################################################################

            # if we have a selected region

            if (cropped):

                # detect and match features from current image

                keypoints, descriptors = feature_object.detectAndCompute(gray_frame,None)

                #print(len(descriptors))
                #print(len(descriptors_cropped_region))

                # get best matches (and second best matches) between current and cropped region features
                # using a k Nearst Neighboour (kNN) radial matcher with k=2

                matches = []
                if (len(descriptors) > 0):
                        #flann.clear()
                        matches = matcher.knnMatch(descriptors_cropped_region, trainDescriptors = descriptors, k = 2)

                # Need to isolate only good matches, so create a mask

                # matchesMask = [[0,0] for i in range(len(matches))]

                # perform a first match to second match ratio test as original SIFT paper (known as Lowe's ration)
                # using the matching distances of the first and second matches

                good_matches = []
                try:
                    for (m,n) in matches:
                        if m.distance < 0.7*n.distance:
                            good_matches.append(m)
                except ValueError:
                    print("caught error - no matches from current frame")

                # fit an ellipse to the detection

                if (show_ellipse_fit):
                    destination_pts = np.float32([ keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

                    # least squares ellipse fitting requires at least 5 points

                    if (len(destination_pts) > 5):
                        ellipseFit = cv2.fitEllipse(destination_pts)
                        cv2.ellipse(frame, ellipseFit, (0, 0, 255), 2, 8)

                # if set to compute homography also

                if (compute_object_position_via_homography):

                    # check we have enough good matches

                    if len(good_matches)>MIN_MATCH_COUNT:

                        # construct two sets of points - source (the selected object/region points), destination (the current frame points)

                        source_pts = np.float32([ keypoints_cropped_region[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                        destination_pts = np.float32([ keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

                        # compute the homography (matrix transform) from one set to the other using RANSAC

                        H, mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC, 5.0)

                        if (transform_image_via_homography):

                            # if we are transforming the whole image

                            h,w,c = frame.shape

                            # create empty image and transform cropped area into it

                            transformOverlay = np.zeros((w,h,1), np.uint8)
                            transformOverlay = cv2.warpPerspective(crop, H, (w,h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                            # add both together to get output where re-inserted cropped
                            # region is brighter than the surronding area so we can see
                            # visualize the warped image insertion

                            frame = cv2.addWeighted(frame, 0.5, transformOverlay, 0.5, 0)

                        else:

                            # extract the bounding co-ordinates of the cropped/selected region

                            h,w,c = crop.shape
                            boundingbox_points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                            # transform the bounding co-ordinates by homography H

                            dst = cv2.perspectiveTransform(boundingbox_points,H)

                            # draw the corresponding

                            frame = cv2.polylines(frame,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)

                    else:
                        print("Not enough matches for found for homography - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))

                # draw the matches

                draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), flags = 0)
                display_matches = cv2.drawMatches(crop,keypoints_cropped_region,frame,keypoints,good_matches,None,**draw_params)
                cv2.imshow(windowName2,display_matches)

            # if running in detection only then draw detections

            if (show_detection_only):
                keypoints, descriptors = feature_object.detectAndCompute(gray_frame,None)
                frame = cv2.drawKeypoints(frame,keypoints,None,(255,0,0),4)

            #####################################################################

            # display live image

            cv2.imshow(windowName,frame)

            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

            # start the event loop - essential

            # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
            # It waits for specified milliseconds for any keyboard event.
            # If you press any key in that time, the program continues.
            # If 0 is passed, it waits indefinitely for a key stroke.
            # (bitwise and with 0xFF to extract least significant byte of multi-byte response)
            # here we use a wait time in ms. that takes account of processing time already used in the loop

            # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

            # It can also be set to detect specific key strokes by recording which key is pressed

            # e.g. if user presses "x" then exit

            if (key == ord('x')):
                keep_processing = False

            # compute homography position for object

            elif (key == ord('h')):
                compute_object_position_via_homography = not(compute_object_position_via_homography)

            # compute transform of whole image via homography

            elif (key == ord('t')):
                transform_image_via_homography = not(transform_image_via_homography)

            # compute ellipse fit to matched points

            elif (key == ord('e')):
                show_ellipse_fit = not(show_ellipse_fit)

            # just shown feature points

            elif (key == ord('d')):
                show_detection_only = not(show_detection_only)

            # use SIFT points

            elif (key == ord('s')):

                if (extraOpenCVModulesPresent()):
                    # Create a SIFT feature object with a Hessian Threshold set to 400
                    # if this fails see (http://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/)

                    feature_object = cv2.xfeatures2d.SIFT_create(400)
                    keypoints_cropped_region = []
                    keypoints = []
                    matches = []
                    cropped = False
                else:
                    print("sorry - SIFT xfeatures2d module not available")

        # close all windows

        cv2.destroyAllWindows()

    else:
        print("No image file specified.")

    #####################################################################
