####################################################################

# Example : ORB feature point detection and matching between two
# images from a set of image files

# Author : TCalum Johnston, toby.breckon@durham.ac.uk

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

# where is the data?

master_path_to_dataset = os.path.dirname(os.path.realpath(__file__)) + "\TTBB-durham-02-10-17-sub10";
master_path_to_yolo_resources = os.path.dirname(os.path.realpath(__file__)) + "\yolo resources"; 
directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images";  