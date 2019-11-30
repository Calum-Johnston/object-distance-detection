####################################################################

# Example : ORB feature point detection and matching between two
# images from a set of image files

# Author : Calum Johnston, toby.breckon@durham.ac.uk

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based in part on tutorial at:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

#####################################################################

import cv2
import os

# where is the data?

master_path_to_dataset = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\z-resources\TTBB-durham-02-10-17-sub10";
directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images";

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

    feature_object = cv2.ORB_create(800, scoreType=cv2.ORB_FAST_SCORE)
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
        table_number = 6, # 12
        key_size = 12,     # 20
        multi_probe_level = 1) #2
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params,search_params)

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :
        
        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # read in images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # detect the keypoints using ORB Detector, compute the descriptors
        kpL, desL = feature_object.detectAndCompute(imgL,None)
        kpR, desR = feature_object.detectAndCompute(imgR,None)

        # Matching descriptor vectors with a FLANN based matcher
        matches = []
        if(len(desR > 0)):
            matches = matcher.knnMatch(desL, desR, k = 2)
        
        # Need to draw only good matches, so create a mask
        good_matches = []

        # filter matches using the Lowe's ratio test
        try:
            for (m,n) in matches:
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)
                    pt1 = kpL[m.queryIdx].pt  #coordinates of left image feature
                    pt2 = kpR[m.trainIdx].pt  #coordinates of corresponding right image feature
        except ValueError:
            print("caught error - no matches from current frame")

        # draw matches onto images and display
        draw_params = dict(matchColor = (0,255,0), 
                           singlePointColor = (255,0,0), 
                           flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        display_matches = cv2.drawMatches(imgL,kpL,imgR,kpR,good_matches,None,**draw_params)
        cv2.imshow("Matches", display_matches)

        #draw keypoints onto images and display
        #keypoints_imgL = cv2.drawKeypoints(imgL, kpL, None, (0, 255, 0))
        #cv2.imshow("hi",keypoints_imgL)

        cv2.waitKey()
