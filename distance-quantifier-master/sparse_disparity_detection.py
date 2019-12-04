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

# setup ORB feature detection parameters
nfeatures = 5000                   # Maximum number of features to retain
scaleFactor = 1.2                   # Pyramid decimation ratio
nlevels = 8                         # The number of pyramid levels
edgeThreshold = 15                   # Size of the border where the features are not detected (should match PatchSize)
firstLevel = 0                      # The level of pyramid to put source image to
WTA_K = 2                           # The number of points that produce each element of the oriented BRIEF descriptor
scoreType = cv2.ORB_HARRIS_SCORE    # Harris algorithm used to rank features. FAST_SCORE could also be used, faster to compute but produces less stable keypoints
patchSize = 15                      # Size of the patch used by the oriented BRIEF descriptor
fastThreshold = 20                  # The fast threshold

# Initiate ORB detector (to detect feature points within images)
feature_object = cv2.ORB_create(
    nfeatures=nfeatures,
    scaleFactor=scaleFactor,                   
    nlevels=nlevels,             
    edgeThreshold=edgeThreshold,
    firstLevel=firstLevel,
    WTA_K=WTA_K,
    scoreType=scoreType,    
    patchSize=patchSize,                      
    fastThreshold=fastThreshold
    )

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
# increaseSize: size the area of the object has been increased (if image was too small)
# left: corresponding x value in original image where object area starts
def disparity(imgL, imgR, f, B, increaseSize, left):
    
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
    # - using Lowe's ratio test (rejects poor matches by computing ratio
    # between best and the second-best match)
    # - by determining whether they lie on a similar y axis
    # - by determining whether they were outside the box containing the object
    try:
        for (m,n) in matches:
            if m.distance < 0.7*n.distance:
                pt1 = kpL[m.queryIdx].pt  #coordinates of left image feature
                pt2 = kpR[m.trainIdx].pt  #coordinates of corresponding right image feature
                if (pt1[1] == pt2[1]):

                    # check the match is for the object in question
                    if(pt1[0] > 20 and pt1[0] < imgL.shape[0] - 20 and
                       pt1[1] > 20 and pt1[1] < imgL.shape[1] - 20):
                        good_matches.append(m)
                    
    except ValueError:
        print("caught error - no matches from current frame")

    # draw matches onto images and display
    draw_params = dict(matchColor = (0,255,0), 
                        singlePointColor = (255,0,0), 
                        flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    display_matches = cv2.drawMatches(imgL,kpL,imgR,kpR,good_matches,None,**draw_params)

    # Gets the average distance based on the best features mapped
    average_distance = getAverageDistances(good_matches, kpL, kpR, f, B, left)
    
    return average_distance


#####################################################################
# Gets the distance of an object based on the best features matched between two images
# good_matches: list of matches between the two images
# kpL: list of keypoints found in the first image (imgL)
# kpR: list of keypoints found in the second image (imgR)
# f: focal length in pixels (camera parameter)
# B: camera baseline in metres (camera parameter)
# left: corresponding x value in original image where object area starts
def getAverageDistances(good_matches, kpL, kpR, f, B, left):
    # variables definitions
    disparity_total = 0         # counts sum of total disparity
    disparity_count = 0         # counts number of times disparity is calculated at > 0

    # determines the disparity for each match 
    for match in good_matches:
        ptL = kpL[match.queryIdx].pt  #coordinates of left image feature
        ptR = kpR[match.trainIdx].pt  # coordinates of right image features
        disparity = int(abs((ptL[0] + left) - ptR[0]))
        if(disparity > 0):
            disparity_total += disparity
            disparity_count += 1

    # calculates average distances based on disparitys calculated
    if(disparity_count > 0): 
        averageDisparity = disparity_total / disparity_count
        averageDistance = (f * B) / averageDisparity
        return averageDistance
    return 0
