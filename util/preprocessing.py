import argparse
import os

import numpy as np

from util import *


def load_dataset(processed_dir, is_mask=False, small_data=False):
    """Loads the depth images and joints from the processed dataset.

    Note that each joint is a coordinate of the form (im_x, im_y, depth_z).
    Each depth image is an H x W image containing depth_z values.

    depth_z values are in meters.

    @return:
        depth_images : depth images (N x H x W)
        joints : joint positions (N x NUM_JOINTS x 3)
    """
    print('Loading data from directory %s' % processed_dir)

    # Load input and labels from numpy files
    # N x H x W depth images
    depth_images = np.load(os.path.join(processed_dir, 'depth_images.npy'))
    joints = np.load(os.path.join(processed_dir, 'joints.npy')
                     )  # N x NUM_JOINTS x 3 joint locations

    assert depth_images.shape[1] == IMAGE_HEIGHT and depth_images.shape[2] == IMAGE_WITCH, "Invalid dimensions for depth image"

    # Load and apply mask to the depth images
    if is_mask:
        # N x H x W depth mask
        depth_mask = np.load(os.path.join(processed_dir, 'depth_mask.npy'))
        depth_images = depth_images * depth_mask

    # Run experiments on random subset of data
    if small_data:
        random_idx = np.random.choice(
            depth_images.shape[0], SMALL_DATA_SIZE, replace=False)
        depth_images, joints = depth_images[random_idx], joints[random_idx]

    print('Data loaded: # data: %d' % depth_images.shape[0])
    return depth_images, joints


def split_dataset(X, y, train_ratio):
    """Splits the dataset according to the train-test ratio.

    @params:
        X : depth images (N x H x W)
        y : joint positions (N x NUM_JOINTS x 3)
        train_ratio : ratio of training to test
    """
    test_ratio = 1.0 - train_ratio
    num_test = int(X.shape[0] * test_ratio)

    X_train, y_train = X[num_test:], y[num_test:]
    X_test, y_test = X[:num_test], y[:num_test]

    print('Data split: # training data: %d, # test data: %d' %
          (X_train.shape[0], X_test.shape[0]))
    return X_train, y_train, X_test, y_test


def init_workspace(dir_list):
    try:
        for path in dir_list:
            if not os.path.exists(path):
                os.makedirs(path)
    except OSError as e:
        if e.errno != 17:
            raise
        pass
