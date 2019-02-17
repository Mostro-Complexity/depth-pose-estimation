import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeRegressor

# Train-test ratio
TRAIN_RATIO = 0.8
SMALL_DATA_SIZE = 5000

# Dimension of each feature vector
NUM_FEATS = 500
MAX_FEAT_OFFSET = 150

# Number of samples for each joint for each example
NUM_SAMPLES = 300

# Set maximum XYZ offset from each joint
MAX_XY_OFFSET = 10  # image xy coordinates (pixels)
MAX_Z_OFFSET = 0.5  # z-depth coordinates (meters)

# Number of clusters for K-Means regression
K = 20

# Depth image dimension
IMAGE_HEIGHT, IMAGE_WITCH = 240, 320
# H, W = 424, 512

# See https://help.autodesk.com/view/MOBPRO/2018/ENU/?guid=__cpp_ref__nui_image_camera_8h_source_html
C = 3.8605e-3  # NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

###############################################################################
# RTW Constants
###############################################################################

# Number of joints in a skeleton
NUM_JOINTS = 15

# List of joint names
JOINT_NAMES = ['NECK (0)', 'HEAD (1)',
               'LEFT SHOULDER (2)', 'LEFT ELBOW (3)', 'LEFT HAND (4)',
               'RIGHT SHOULDER (5)', 'RIGHT ELBOW (6)', 'RIGHT HAND (7)',
               'LEFT KNEE (8)', 'LEFT FOOT (9)',
               'RIGHT KNEE (10)', 'RIGHT FOOT (11)',
               'LEFT HIP (12)',
               'RIGHT HIP (13)',
               'TORSO (14)']

# Map from joint names to index
JOINT_IDX = {
    'NECK': 0,
    'HEAD': 1,
    'LEFT SHOULDER': 2,
    'LEFT ELBOW': 3,
    'LEFT HAND': 4,
    'RIGHT SHOULDER': 5,
    'RIGHT ELBOW': 6,
    'RIGHT HAND': 7,
    'LEFT KNEE': 8,
    'LEFT FOOT': 9,
    'RIGHT KNEE': 10,
    'RIGHT FOOT': 11,
    'LEFT HIP': 12,
    'RIGHT HIP': 13,
    'TORSO': 14,
}

# Set the kinematic tree (starting from torso body center)
kinem_order = [14,  0, 13, 12, 1, 2, 5, 3, 6, 4, 7,  8, 10, 9, 11]
kinem_parent = [-1, 14, 14, 14, 0, 0, 0, 2, 5, 3, 6, 12, 13, 8, 10]

# Number of steps during evaluation
NUM_STEPS = 300

# Step size (in cm) during evaluation
STEP_SIZE = 2

np.set_printoptions(threshold=np.nan)


palette = [(34, 88, 226), (34, 69, 101), (0, 195, 243), (146, 86, 135),
           (38, 61, 43), (241, 202, 161), (50, 0, 190), (128, 178, 194),
           (23, 45, 136), (0, 211, 220), (172, 143, 230), (108, 68, 179),
           (121, 147, 249), (151, 78, 96), (0, 166, 246), (165, 103, 0),
           (86, 136, 0), (130, 132, 132), (0, 182, 141), (0, 132, 243)]  # BGR

jointNameEVAL = ['NECK', 'HEAD', 'LEFT SHOULDER', 'LEFT ELBOW',
                 'LEFT HAND', 'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT HAND',
                 'LEFT KNEE', 'LEFT FOOT', 'RIGHT KNEE', 'RIGHT FOOT',
                 'LEFT HIP', 'RIGHT HIP', 'TORSO']
jointNameITOP = ['HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_HAND', 'RIGHT_HAND',
                 'TORSO', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE',
                 'RIGHT_KNEE', 'LEFT_FOOT', 'RIGHT_FOOT']

trainTestITOP = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # train = 0, test = 1
kinemOrderEVAL = [0, 1, 2, 5, 3, 6, 4, 7, 8, 10, 9, 11]
kinemParentEVAL = [-1, 0, 0, 0, 2, 5, 3, 6, -1, -1, 8, 10]
kinemOrderITOP = [8, 1, 0, 9, 10, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14]
kinemOrderITOPUpper = [8, 1, 0, 2, 3, 4, 5, 6, 7]
kinemParentITOP = [-1, 8, 1, 8, 8,  1, 1, 2, 3, 4, 5, 9,  10, 11, 12]

from . import modeling
from . import visualizing
from . import preprocessing