#####################################################################

# Example : compute SGBM disparity of two images

# Author : Calum Johnston, calum.p.johnston@durham.ac.uk

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
  
#####################################################################

# https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html

#setup disparity stereo processor parameters
SADWindowSize = 2
minDisparity = 0                   # Minimum possible disparity value (default 0)
numDisparities = 128               # Maximum disparity minus minimum disparity (default 16)
blockSize = 6                      # Matched block size - must be odd (default 3)
P1 = 8 * 1 * SADWindowSize ** 2    # Controls disparity smoothness (default 0)
P2 = 32 * 1 * SADWindowSize ** 2   # Controls disparity smoothness (default 0)
disp12MaxDiff = 0                  # Maximum allowed difference in the left-right disparity check
preFilterCap = 0                   # Truncation value for the prefiltered image pixels
uniquenessRatio = 15               # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
speckleWindowSize = 200            # Maximum size of smooth disparity regions to consider their noise and speckles invalidate
speckleRange = 2                   # Maximum disparity variation within each connected component
mode = cv2.STEREO_SGBM_MODE_SGBM  

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
max_disparity = 128

# setup the StereoSGBM object
stereoProcessor = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=P1,
    P2=P2,
    disp12MaxDiff=disp12MaxDiff,
    preFilterCap=preFilterCap,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    mode=mode
    )



def disparity(imgL, imgR):
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation
    grayL = np.power(grayL, 0.75).astype('uint8');
    grayR = np.power(grayR, 0.75).astype('uint8');
        
    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)
    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)
    dispNoiseFilter = 5; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available
    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)
    disparity_scaled = (disparity_scaled * (256.0 / max_disparity)).astype(np.uint8)

    # return the image
    return disparity_scaled

#####################################################################