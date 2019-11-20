#####################################################################

# Example : load, display and compute SGBM disparity
# for a set of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic python script which combines stereo disparity and object recognition
# to determine distance to objects

# Credit to : Toby Breckon, toby.breckon@durham.ac.uk
#
# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys
import math
import numpy as np
import os

# where is the data ? - set this to where you have it

master_path_to_dataset = os.path.dirname(os.path.realpath(__file__)) + "\TTBB-durham-02-10-17-sub10"; 
directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images";   