####################################################################

# Example : ORB feature point detection and matching between two images
# - images are passed in as parameters (assumed setup in another file)

# Author : Calum Johnston, calum.p.johnston@durham.ac.uk

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based in part on tutorial at:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

#####################################################################

import cv2

# Initiate ORB detector (to detect feature points within images)
feature_object = cv2.ORB_create(800, scoreType=cv2.ORB_FAST_SCORE)

# setup the FLANN parameters and initialise the matcher
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
        table_number = 6, # 12
        key_size = 12,     # 20
        multi_probe_level = 1) #2
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)

def disparity(imgL, imgR):
    # detect the keypoints using ORB Detector, compute the descriptors
    kpL, desL = feature_object.detectAndCompute(imgL,None)
    kpR, desR = feature_object.detectAndCompute(imgR,None)

    # Matching descriptor vectors with a FLANN based matcher
    matches = []
    if(len(desR > 0)):
        matches = matcher.knnMatch(desL, desR, k = 2)
        
    # Need to draw only good matches, so create a mask
    good_matches = []

    # filter matches so some matches aren't included
    # - using Lowe's ratio test
    # - by determining whether they lie on a similar y axis 
    try:
        for (m,n) in matches:
            if m.distance < 0.7*n.distance:
                pt1 = kpL[m.queryIdx].pt  #coordinates of left image feature
                pt2 = kpR[m.trainIdx].pt  #coordinates of corresponding right image feature
                if not(pt1[1] > pt2[1] + 10 or pt1[1] + 10 < pt2[1]):
                        print(pt1, pt2)
                        good_matches.append(m)
    except ValueError:
        print("caught error - no matches from current frame")

    # draw matches onto images and display
    draw_params = dict(matchColor = (0,255,0), 
                        singlePointColor = (255,0,0), 
                        flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    display_matches = cv2.drawMatches(imgL,kpL,imgR,kpR,good_matches,None,**draw_params)
    cv2.imshow("Matches", display_matches)

    return display_matches
