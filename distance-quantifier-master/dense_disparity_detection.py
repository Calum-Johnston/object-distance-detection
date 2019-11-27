#####################################################################

# Example : compute SGBM disparity of two image

# Author : Calum Johnston, calum.p.johnston@durham.ac.uk

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
  
#####################################################################

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

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