import argparse
import os
import pickle
import sys
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool

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

# entire offset
entire_offset = np.empty((NUM_SAMPLES, 1287, 3))
# Number of steps during evaluation
NUM_STEPS = 300

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


def compute_theta(num_feats=NUM_FEATS, max_feat_offset=MAX_FEAT_OFFSET):
    """Computes the theta for each skeleton.

    @params:
        max_feat_offset : the maximum offset for features (before divided by d)
        num_feats : the number of features of each offset point
    """
    print('Computing theta...')

    # Compute the theta = (-max_feat_offset, max_feat_offset) for 4 coordinates (x1, x2, y1, y2)
    # (4, num_feats)
    theta = np.random.randint(-max_feat_offset,
                              max_feat_offset + 1, (4, num_feats))

    return theta


def get_features_by_time(img_seq, joint_time_seq, z, theta):
    """Gets the feature vector for a single example.

    @params:
        img : depth image sequence. shape=(T x H x W)
        joint_time_seq : 关节点的时间序列 shape=(T x 3)
        z : z-value of body center shape=(T)
        theta : (-max_feat_offset, max_feat_offset) = (T, 4, num_feats)
    """

    joint_time_seq[:, 1] = np.clip(  # limits y between 0 and H
        joint_time_seq[:, 1], 0, IMAGE_HEIGHT - 1)

    joint_time_seq[:, 0] = np.clip(  # limits x between 0 and W
        joint_time_seq[:, 0], 0, IMAGE_WITCH - 1)
    coor = np.rint(joint_time_seq[:, :2])  # rounds to nearest integer
    coor = coor[:, ::-1].astype(int)  # 按列逆序, x, y -> y, x

    # Find z-value of joint offset by indexing into depth imag
    LARGE_NUM = 100  # initialize to LARGE_NUM
    img_seq[img_seq == 0] = LARGE_NUM  # no division by zero

    # Extracted depth time sequence from images
    depth_seq = img_seq[np.arange(img_seq.shape[0]), coor[:, 0], coor[:, 1]]
    flag = depth_seq == LARGE_NUM  # If depth equals to LARGE_NUM
    depth_seq[flag] = z[flag]  # depth is z

    # Normalize x theta by z-value
    x1 = np.clip(coor[:, 1, None] + theta[0] /
                 depth_seq[:, None], 0, IMAGE_WITCH - 1).astype(int)
    x2 = np.clip(coor[:, 1, None] + theta[2] /
                 depth_seq[:, None], 0, IMAGE_WITCH - 1).astype(int)

    # Normalize y theta by z-value
    y1 = np.clip(coor[:, 0, None] + theta[1] /
                 depth_seq[:, None], 0, IMAGE_HEIGHT - 1).astype(int)
    y2 = np.clip(coor[:, 0, None] + theta[3] /
                 depth_seq[:, None], 0, IMAGE_HEIGHT - 1).astype(int)

    # Feature matrix. shape=(1608, 500) 列向量是每张图片的特征矩阵
    feature = np.array([img_seq[t, y1[t, :], x1[t, :]] -
                        img_seq[t, y2[t, :], x2[t, :]] for t in range(img_seq.shape[0])])
    return feature


def get_features(img, q, z, theta):
    """Gets the feature vector for a single example.

    @params:
        img : depth image = (H x W)
        q : joint xyz position with some random offset vector
        z : z-value of body center
        theta : (-max_feat_offset, max_feat_offset) = (4, num_feats)
    """
    # Retrieve the (y, x) of the joint offset coordinates
    coor = q[:2][::-1]  # coor: flip x, y -> y, x
    coor[0] = np.clip(coor[0], 0, IMAGE_HEIGHT - 1)  # limits y between 0 and H
    coor[1] = np.clip(coor[1], 0, IMAGE_WITCH - 1)  # limits x between 0 and W
    coor = np.rint(coor).astype(int)  # rounds to nearest integer

    # Find z-value of joint offset by indexing into depth imag
    LARGE_NUM = 100
    img[img == 0] = LARGE_NUM  # no division by zero
    # initialize to LARGE_NUM
    dq = z if (img[tuple(coor)] == LARGE_NUM) else img[tuple(coor)]

    # Normalize x theta by z-value
    x1 = np.clip(coor[1] + theta[0] / dq, 0, IMAGE_WITCH - 1).astype(int)
    x2 = np.clip(coor[1] + theta[2] / dq, 0, IMAGE_WITCH - 1).astype(int)

    # Normalize y theta by z-value
    y1 = np.clip(coor[0] + theta[1] / dq, 0, IMAGE_HEIGHT - 1).astype(int)
    y2 = np.clip(coor[0] + theta[3] / dq, 0, IMAGE_HEIGHT - 1).astype(int)

    # Get the feature vector as difference of depth-values
    feature = img[y1, x1] - img[y2, x2]
    return feature


def get_random_offset(max_offset_xy=MAX_XY_OFFSET, max_offset_z=MAX_Z_OFFSET):
    """Gets xyz vector with uniformly random xy and z offsets.
    """
    offset_xy = np.random.randint(-max_offset_xy, max_offset_xy + 1, 2)
    offset_z = np.random.uniform(-max_offset_z, max_offset_z, 1)
    offset = np.concatenate((offset_xy, offset_z))  # xyz offset
    return offset


def extract_feat_by_time(joint_id, imgs, joints, theta, num_feats=NUM_FEATS):
    """Generates training samples for each joint by time sequence.

    Each sample is (i, q, u, f) where:
         i is the index of the depth image,
         q is the random offset point from the joint,
         u is the unit direction vector toward the joint location,
         f is the feature array

    @params:
        imgs : depth images (T x H x W)
        joints : joint position = (T x NUM_JOINTS x 3) = (im_x, im_y, depth_z)
        joint_id : current joint id
        num_samples : number of samples of each joint
        max_offset_xy : maximum offset for samples in (x, y) axes
        max_offset_z : maximum offset for samples in z axis

    @return:
        feature : samples feature array (T x num_samples x num_feats)
        unit : samples unit direction vectors (T x num_samples x 3)
    """
    feature = np.zeros((imgs.shape[0], NUM_SAMPLES, num_feats))
    unit = np.zeros((imgs.shape[0], NUM_SAMPLES, 3))

    for sample_id in range(NUM_SAMPLES):  # 生成300个采样点
        print('Start getting %s sample point %d...' %
              (JOINT_NAMES[joint_id], sample_id))
        offset = np.array([np.random.randint(-10, 11, imgs.shape[0]),
                           np.random.randint(-10, 11, imgs.shape[0]),
                           np.random.uniform(-0.5, 0.5, imgs.shape[0])]).T
        # if np.linalg.norm(offset[i, :]) != 0,
        # unit[i, sample_id, :] = np.array([-offset[i, :] / np.linalg.norm(offset[i, :])
        flag = np.linalg.norm(offset, axis=1) != 0
        unit[flag, sample_id, :] = -offset[flag, :] / \
            np.linalg.norm(offset[flag], axis=1)[:, None]

        body_center_z = joints[:, JOINT_IDX['TORSO'], 2]

        feature[:, sample_id, :] = get_features_by_time(  # 一次采样的特征(T*500)
            imgs, joints[:, joint_id] + offset, body_center_z, theta)

    return feature, unit


def stochastic(regressor, features, unit_directions):
    """Applies stochastic relaxation when choosing the unit direction. Training
    samples at the leaf nodes are further clustered using K-means.
    """
    L = {}

    indices = regressor.apply(features)  # leaf id of each sample
    leaf_ids = np.unique(indices)  # array of unique leaf ids

    print('Running stochastic (minibatch) K-Means...')
    for leaf_id in leaf_ids:
        kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000)
        labels = kmeans.fit_predict(unit_directions[indices == leaf_id])
        weights = np.bincount(labels).astype(float) / labels.shape[0]

        # Normalize the centers
        centers = kmeans.cluster_centers_
        centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
        # checkUnitVectors(centers)

        L[leaf_id] = (weights, centers)
    return L


def train(joint_id, X, y, model_dir, load_models, min_samples_leaf=400):
    """Trains a regressor tree on the unit directions towards the joint.

    @params:
        joint_id : current joint id
        X : samples feature array (N x num_samples x num_feats)
        y : samples unit direction vectors (N x num_samples x 3)
        min_samples_split : minimum number of samples required to split an internal node
        load_models : load trained models from disk (if exist)
    """
    print('Start training %s model...' % JOINT_NAMES[joint_id])

    regressor_path = os.path.join(
        model_dir, 'regressor' + str(joint_id) + '.pkl')
    L_path = os.path.join(model_dir, 'L' + str(joint_id) + '.pkl')

    # Load saved model from disk
    if load_models and (os.path.isfile(regressor_path) and os.path.isfile(L_path)):
        print('Loading model %s from files...' % JOINT_NAMES[joint_id])

        regressor = pickle.load(open(regressor_path, 'rb'))
        L = pickle.load(open(L_path, 'rb'))
        return regressor, L

    # (N x num_samples, num_feats)
    X_reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_reshape = y.reshape(y.shape[0] * y.shape[1],
                          y.shape[2])  # (N x num_samples, 3)

    # Count the number of valid (non-zero) samples
    # inverse of invalid samples
    valid_rows = np.logical_not(np.all(X_reshape == 0, axis=1))
    print('Model %s - Valid samples: %d / %d' %
          (JOINT_NAMES[joint_id], X_reshape[valid_rows].shape[0], X_reshape.shape[0]))

    # Fit decision tree to samples
    regressor = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
    regressor.fit(X_reshape[valid_rows], y_reshape[valid_rows])

    L = stochastic(regressor, X_reshape, y_reshape)

    # Print statistics on leafs
    leaf_ids = regressor.apply(X_reshape)
    bin = np.bincount(leaf_ids)
    unique_ids = np.unique(leaf_ids)
    biggest = np.argmax(bin)
    smallest = np.argmin(bin[bin != 0])

    print('Model %s - # Leaves: %d' %
          (JOINT_NAMES[joint_id], unique_ids.shape[0]))
    print('Model %s - Smallest Leaf ID: %d, # Samples: %d/%d' %
          (JOINT_NAMES[joint_id], smallest, bin[bin != 0][smallest], np.sum(bin)))
    print('Model %s - Biggest Leaf ID: %d, # Samples: %d/%d' %
          (JOINT_NAMES[joint_id], biggest, bin[biggest], np.sum(bin)))
    print('Model %s - Average Leaf Size: %d' %
          (JOINT_NAMES[joint_id], np.sum(bin) / unique_ids.shape[0]))

    # Save models to disk
    pickle.dump(regressor, open(regressor_path, 'wb'))
    pickle.dump(L, open(L_path, 'wb'))

    return regressor, L


def train_series(joint_id, X, y, theta, model_dir, load_model_flag):
    """Train each joint sequentially.
    """
    feature, unit = extract_feat_by_time(joint_id, X, y, theta)
    # feature, unit = extract_feat_by_frame(joint_id, X, y, theta)
    regressor, L = train(joint_id, feature, unit, model_dir, load_model_flag)
    return regressor, L


def predict(regressor, L, theta, qm0, img_seq, body_center, num_steps=300, step_size=2):
    """Test the model on a single example.
    """
    num_test_img = img_seq.shape[0]
    qm = np.zeros((num_test_img, num_steps + 1, 3))
    qm[:, 0] = qm0
    joint_pred = np.zeros((num_test_img, 3))

    for i in range(num_steps):
        body_center_z = body_center[:, 2]
        f = get_features_by_time(img_seq, qm[:, i], body_center_z, theta)
        # f = f.reshape(1, -1)  # flatten feature vector
        leaf_id = np.array([regressor.apply(f[t].reshape(1, -1))[0]
                            for t in range(num_test_img)])

        idx = np.array([np.random.choice(K, p=L[leaf_id[t]][0])
                        for t in range(num_test_img)])  # L[leaf_id][0] = weights
        u = np.array([L[leaf_id[t]][1][idx[t]]
                      for t in range(num_test_img)])  # L[leaf_id][1] = centers

        qm[:, i + 1] = qm[:, i] + u * step_size
        # limit x between 0 and W
        qm[:, i + 1, 0] = np.clip(qm[:, i + 1, 0], 0, IMAGE_WITCH - 1)
        # limit y between 0 and H
        qm[:, i + 1, 1] = np.clip(qm[:, i + 1, 1], 0, IMAGE_HEIGHT - 1)
        # index (y, x) into image for z position
        qm[:, i + 1, 2] = np.array([img_seq[t, int(qm[t, i + 1, 1]), int(qm[t, i + 1, 0])]
                                    for t in range(num_test_img)])
        joint_pred += qm[:, i + 1]

    joint_pred = joint_pred / num_steps
    return qm, joint_pred


def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != 17:
            raise
        pass


def drawPred(img, joints, paths, center, filename, nJoints, jointName):
    H = img.shape[0]
    W = img.shape[1]

    img = (img - np.amin(img)) * 255.0 / (np.amax(img) - np.amin(img))
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    img = np.hstack((img, np.zeros((H, 100, 3)))).astype(np.uint8)

    if paths is not None:
        paths_copy = paths.copy()
        for i, path in enumerate(paths_copy):
            nPts = path.shape[0]
            for j, pt in enumerate(path):
                color = tuple(c * (2 * j + nPts) / (3 * nPts)
                              for c in palette[i])
                cv2.circle(img, tuple(pt[:2].astype(np.uint16)), 1, color, -1)

    if joints is not None:
        joints_copy = joints.copy()
        for i, joint in enumerate(joints_copy):
            cv2.circle(img, tuple(joint[:2].astype(
                np.uint16)), 4, palette[i], -1)

        for i, joint in enumerate(joints):
            cv2.rectangle(img, (W, int(H * i / nJoints)), (W + 100,
                                                           int(H * (i + 1) / nJoints - 1)), palette[i], -1)
            cv2.putText(img, jointName[i], (W, int(
                H * (i + 1) / nJoints - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

    cv2.rectangle(img, tuple([int(center[0] - 2), int(center[1] - 2)]),
                  tuple([int(center[0] + 2), int(center[1] + 2)]),
                  palette[nJoints], -1)

    cv2.imwrite(filename, img)


def checkUnitVectors(unitVectors):
    s1 = np.sum(unitVectors.astype(np.float32)**2)
    s2 = unitVectors.shape[0]
    print('error: %0.3f' % (abs(s1 - s2) / s2))


def pixel2world(pixel, C):
    world = np.empty(pixel.shape)
    world[:, 2] = pixel[:, 2]
    world[:, 0] = (pixel[:, 0] - IMAGE_WITCH / 2.0) * C * pixel[:, 2]
    world[:, 1] = -(pixel[:, 1] - IMAGE_HEIGHT / 2.0) * C * pixel[:, 2]
    return world


def world2pixel(world, C):
    pixel = np.empty(world.shape)
    pixel[:, 2] = world[:, 2]
    pixel[:, 0] = (world[:, 0] / world[:, 2] / C +
                   IMAGE_WITCH / 2.0).astype(int)
    pixel[:, 1] = (-world[:, 1] / world[:, 2] / C +
                   IMAGE_HEIGHT / 2.0).astype(int)
    return pixel


def get_distances(y_test, y_pred):
    """Compute the raw world distances between the prediction and actual joint
    locations.
    """
    assert y_test.shape == y_pred.shape, "Mismatch of y_test and y_pred"

    distances = np.zeros((y_test.shape[:2]))
    for i in range(y_test.shape[0]):
        p1 = pixel2world(y_test[i], C)
        p2 = pixel2world(y_pred[i], C)
        distances[i] = np.sqrt(np.sum((p1 - p2)**2, axis=1))
    return distances


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

    depth_images, joints = load_dataset(args.input_dir)
    imgs_train, joints_train, imgs_test, joints_test = split_dataset(
        depth_images, joints, TRAIN_RATIO)

    num_train = imgs_train.shape[0]
    num_test = imgs_test.shape[0]

    # theta = compute_theta()
    theta = None
    regressors, Ls = {}, {}

    if args.load_model:
        print('\n------- Testing models -------')
        theta = pickle.load(
            open(os.path.join(args.model_dir, 'theta.pkl'), 'rb'))

        for joint_id in range(NUM_JOINTS):
            # Load saved model from disk
            print('Loading model %s from files...' % JOINT_NAMES[joint_id])

            regressors[joint_id] = pickle.load(
                open(os.path.join(args.model_dir, 'regressor' + str(joint_id) + '.pkl'), 'rb'))
            Ls[joint_id] = pickle.load(open(os.path.join(args.model_dir,
                                                         'L' + str(joint_id) + '.pkl'), 'rb'))
    else:
        print('\n------- Training models -------')

        theta = np.random.randint(low=-MAX_FEAT_OFFSET,
                                  high=MAX_FEAT_OFFSET + 1,
                                  size=(4, NUM_FEATS))  # (4, num_feats)
        pickle.dump(theta, open(os.path.join(
            args.model_dir, 'theta.pkl'), 'wb'))

        for joint_id in range(NUM_JOINTS):
            regressors[joint_id], Ls[joint_id] = train_series(
                joint_id, imgs_train, joints_train, theta, args.model_dir, args.load_model)

    print('\n------- Testing models -------')

    qms = np.zeros((num_test, NUM_JOINTS, NUM_STEPS + 1, 3))
    joints_pred = np.zeros((num_test, NUM_JOINTS, 3))
    local_error = np.zeros((num_test, NUM_STEPS + 1, NUM_JOINTS, 3))

    for i, joint_id in enumerate(kinem_order):
        print('Testing %s model', JOINT_NAMES[joint_id])
        parent_joint_id = kinem_parent[i]

        parent_joint = joints_test[:, JOINT_IDX['TORSO']] if parent_joint_id == - \
            1 else joints_pred[:, parent_joint_id]  # 父关节节点的位置
        qms[:, joint_id], joints_pred[:, joint_id] = predict(  # 当前的关节位置簇和关节位置
            regressors[joint_id], Ls[joint_id], theta, parent_joint, imgs_test, joints_test[:, JOINT_IDX['TORSO']])
        local_error[:, :, joint_id, :] = np.tile(
            joints_test[:, np.newaxis, joint_id], (1, NUM_STEPS + 1, 1)) - qms[:, joint_id]

    joints_pred[:, :, 2] = joints_test[:, :, 2]

    print('\n------- Computing evaluation metrics -------')

    distances = get_distances(joints_test, joints_pred) * \
        100.0  # convert from m to cm

    distances_path = os.path.join(args.preds_dir, 'distances.txt')
    np.savetxt(distances_path, distances, fmt='%.3f')

    distances_pixel = np.zeros((joints_test.shape[:2]))
    for i in range(joints_test.shape[0]):
        p1 = joints_test[i]
        p2 = joints_pred[i]
        distances_pixel[i] = np.sqrt(np.sum((p1 - p2)**2, axis=1))

    mAP = 0
    for i in range(NUM_JOINTS):
        print('\nJoint %s:', JOINT_NAMES[i])
        print('Average distance: %f cm' % np.mean(distances[:, i]))
        print('Average pixel distance: %f' %
              np.mean(distances_pixel[:, i]))
        print('5cm accuracy: %f' % (np.sum(
            distances[:, i] < 5) / float(distances.shape[0])))
        print('10cm accuracy: %f' % (np.sum(
            distances[:, i] < 10) / float(distances.shape[0])))
        print('15cm accuracy: %f' % (np.sum(
            distances[:, i] < 15) / float(distances.shape[0])))
        mAP += np.sum(distances[:, i] < 10) / float(distances.shape[0])

    print('mAP (10cm): %f' % (mAP / NUM_JOINTS))

    print('\n------- Saving prediction visualizations -------')

    for test_idx in range(num_test):
        png_path = os.path.join(args.png_dir, str(test_idx) + '.png')
        drawPred(imgs_test[test_idx], joints_pred[test_idx], qms[test_idx],
                 joints_test[test_idx][JOINT_IDX['TORSO']], png_path, NUM_JOINTS, JOINT_NAMES)
