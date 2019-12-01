####################################################################

# Example : ORB feature point detection and matching between two images
# - images are passed in as parameters (assumed setup in another file)
# - imgL should be simply a singular object (calculated by some method)

# Author : Calum Johnston, calum.p.johnston@durham.ac.uk

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based in part on tutorial at:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

#####################################################################

import cv2

# Initiate ORB detector (to detect feature points within images)
feature_object = cv2.ORB_create(50000, scoreType=cv2.ORB_FAST_SCORE)

# setup the FLANN parameters and initialise the matcher
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
        table_number = 6, # 12
        key_size = 12,     # 20
        multi_probe_level = 1) #2
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)

#####################################################################
# Performs feature mapping between two seperate images
# imgL: image that contains the object to have features mapped against
# imgR: image that contains scene to have object features (imgL) matched to
# f: focal length in pixels (camera parameter)
# B: camera baseline in metres (camera parameter)
# startX: corresponding starting x coordinate from imgL (cropped object image) to original image
def disparity(imgL, imgR, f, B, startX):

    # Pad the image with bits if too small
    # (done due to the default scale settings on ORB)
    if(imgL.shape[0] < 100):
        padHeight = int((100 - imgL.shape[0]) / 2)
        imgL = cv2.copyMakeBorder(imgL, padHeight, padHeight, 0, 0, cv2.BORDER_CONSTANT)
        imgR = cv2.copyMakeBorder(imgR, padHeight, padHeight, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if(imgL.shape[1] < 100):
        padWidth = int((100 - imgL.shape[1]) / 2)
        imgL = cv2.copyMakeBorder(imgL, 0, 0, padWidth, padWidth, cv2.BORDER_CONSTANT)
        imgR = cv2.copyMakeBorder(imgR, 0, 0, padWidth, padWidth, cv2.BORDER_CONSTANT, value=(255, 255, 255))

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
                if (pt1[1] == pt2[1]):
                    good_matches.append(m)
    except ValueError:
        print("caught error - no matches from current frame")

    # draw matches onto images and display
    draw_params = dict(matchColor = (0,255,0), 
                        singlePointColor = (255,0,0), 
                        flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    display_matches = cv2.drawMatches(imgL,kpL,imgR,kpR,good_matches,None,**draw_params)

    # Gets the average distance based on the best features mapped
    average_distance = getAverageDistances(good_matches, kpL, kpR, f, B, startX)

    return average_distance


#####################################################################
# Gets the distance of an object based on the best features matched between two images
# good_matches: list of matches between the two images
# kpL: list of keypoints found in the first image (imgL)
# kpR: list of keypoints found in the second image (imgR)
# f: focal length in pixels (camera parameter)
# B: camera baseline in metres (camera parameter)
# startX: corresponding starting x coordinate from imgL (cropped object image) to original image
def getAverageDistances(good_matches, kpL, kpR, f, B, startX):
    totalDisparity = 0
    count = 0
    for match in good_matches:
        ptL = kpL[match.queryIdx].pt  #coordinates of left image feature
        ptR = kpR[match.trainIdx].pt  # coordinates of right image features
        disparity = abs((ptL[0] + startX) - ptR[0])
        if(disparity > 0):
            totalDisparity += disparity
            count += 1
    if(count > 0): 
        averageDisparity = totalDisparity / count
        averageDistance = (f * B) / averageDisparity
        return averageDistance
    return 0