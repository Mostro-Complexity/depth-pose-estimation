import argparse
import os

import numpy as np

from util import *
from util.modeling import DecisionTreeModel
from util.preprocessing import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Random Tree Walks algorithm.')
    parser.add_argument('--load-model', action='store_true',
                        help='Load a pretrained model')
    parser.add_argument('--load-test', action='store_true',
                        help='Run trained model on test set')
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='Directory of the processed input')
    parser.add_argument('--dataset', type=str, default='NTU-RGBD',  # NTU-RGBD, CAD-60
                        help='Name of the dataset to load')

    # Training options
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed')
    parser.add_argument('--shuffle', type=int, default=1,
                        help='Shuffle the data')

    # Evaluation hyperparameters
    # parser.add_argument('--num-steps', type=int, default=300,
    #                     help='Number of steps during evaluation')
    # parser.add_argument('--step-size', type=int, default=2,
    #                     help='Step size (in cm) during evaluation')

    # Output options
    parser.add_argument('--make-png', action='store_true',
                        help='Draw predictions on top of inputs')

    args = parser.parse_args()

    # Set location of output saved files
    args.model_dir = 'model'
    args.preds_dir = 'data/output/preds'
    args.png_dir = 'data/output/png'

    init_workspace([args.preds_dir, args.png_dir])

    depth_images, joints = load_dataset(args.input_dir)
    imgs_train, joints_train, imgs_test, joints_test = split_dataset(
        depth_images, joints, TRAIN_RATIO)

    num_train = imgs_train.shape[0]
    num_test = imgs_test.shape[0]

    model = DecisionTreeModel()

    if args.load_model:
        print('\n------- Loading models -------')
        model.load(args.model_dir)
    else:
        print('\n------- Training models -------')
        model.train_regressors(args.model_dir, imgs_train, joints_train)

    print('\n------- Testing models -------')
    model.predict_regressors(num_test, imgs_test, joints_test)

    print('\n------- Computing evaluation metrics -------')

    model.evaluate_regressors(args.preds_dir, joints_test)

    print('\n------- Saving prediction visualizations -------')
    model.visualize(args.png_dir, imgs_test, joints_test)
