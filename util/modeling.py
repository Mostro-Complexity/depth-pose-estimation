from util import *
from util.visualizing import *


class DecisionTreeModel(object):
    def __init__(self, regressors=None, Ls=None, theta=None):
        if regressors is None:
            self.regressors = {}
        if Ls is None:
            self.Ls = {}
        if theta is None:
            self.theta = np.random.randint(low=-MAX_FEAT_OFFSET,
                                           high=MAX_FEAT_OFFSET + 1,
                                           size=(4, NUM_FEATS))  # (4, num_feats)

    def train_regressors(self, model_dir, imgs_train, joints_train):
        pickle.dump(self.theta, open(os.path.join(
            model_dir, 'theta.pkl'), 'wb'))

        for joint_id in range(NUM_JOINTS):
            feature, unit = self.get_samples_feature(
                joint_id, imgs_train, joints_train, self.theta)

            self.regressors[joint_id], self.Ls[joint_id] = self.train(
                joint_id, feature, unit, model_dir)

    def load(self, model_dir):
        self.theta = pickle.load(
            open(os.path.join(model_dir, 'theta.pkl'), 'rb'))

        for joint_id in range(NUM_JOINTS):
            # Load saved model from disk
            print('Loading model %s from files...' % JOINT_NAMES[joint_id])

            self.regressors[joint_id] = pickle.load(
                open(os.path.join(model_dir, 'regressor' + str(joint_id) + '.pkl'), 'rb'))
            self.Ls[joint_id] = pickle.load(
                open(os.path.join(model_dir, 'L' + str(joint_id) + '.pkl'), 'rb'))

    def predict_regressors(self, num_test, imgs_test, joints_test):
        self.num_test = num_test
        self.qms = np.zeros((num_test, NUM_JOINTS, NUM_STEPS + 1, 3))
        self.joints_pred = np.zeros((num_test, NUM_JOINTS, 3))
        self.local_error = np.zeros((num_test, NUM_STEPS + 1, NUM_JOINTS, 3))

        for i, joint_id in enumerate(kinem_order):
            print('Testing %s model', JOINT_NAMES[joint_id])
            parent_joint_id = kinem_parent[i]

            parent_joint = joints_test[:, JOINT_IDX['TORSO']] if parent_joint_id == - \
                1 else self.joints_pred[:, parent_joint_id]  # 父关节节点的位置

            self.qms[:, joint_id], self.joints_pred[:, joint_id] = self.predict(  # 当前的关节位置簇和关节位置
                self.regressors[joint_id], self.Ls[joint_id], self.theta, parent_joint, imgs_test,
                joints_test[:, JOINT_IDX['TORSO']])

            self.local_error[:, :, joint_id, :] = np.tile(
                joints_test[:, np.newaxis, joint_id], (1, NUM_STEPS + 1, 1)) - self.qms[:, joint_id]

        self.joints_pred[:, :, 2] = joints_test[:, :, 2]  # 可以不要

    def evaluate_regressors(self, preds_dir, joints_test):
        distances = get_distances(joints_test, self.joints_pred) * \
            100.0  # convert from m to cm

        distances_path = os.path.join(preds_dir, 'distances.txt')
        np.savetxt(distances_path, distances, fmt='%.3f')

        distances_pixel = np.zeros((joints_test.shape[:2]))
        for i in range(joints_test.shape[0]):
            p1 = joints_test[i]
            p2 = self.joints_pred[i]
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

    def visualize(self, png_dir, imgs_test, joints_test):
        for test_idx in range(self.num_test):
            png_path = os.path.join(png_dir, str(test_idx) + '.png')
            draw_prediction(imgs_test[test_idx], self.joints_pred[test_idx], self.qms[test_idx],
                     joints_test[test_idx][JOINT_IDX['TORSO']], png_path, NUM_JOINTS, JOINT_NAMES)

    def get_samples_feature(self, joint_id, imgs, joints, theta, num_feats=NUM_FEATS):
        """Generates training samples feature for each joint by time sequence.

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

            feature[:, sample_id, :] = self.get_single_sample_feature(  # 一次采样的特征(T*500)
                imgs, joints[:, joint_id] + offset, body_center_z, theta)

        return feature, unit

    def stochastic(self, regressor, features, unit_directions):
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

    def train(self, joint_id, features, units, model_dir,  min_samples_leaf=400):
        """Trains a regressor tree on the unit directions towards the joint.

        @params:
            joint_id : current joint id
            features : samples feature array (N x num_samples x num_feats)
            units : samples unit direction vectors (N x num_samples x 3)
            model_dir : path of trained models 
            min_samples_leaf : minimum number of samples required to split an internal node
        """
        print('Start training %s model...' % JOINT_NAMES[joint_id])

        regressor_path = os.path.join(
            model_dir, 'regressor' + str(joint_id) + '.pkl')
        L_path = os.path.join(model_dir, 'L' + str(joint_id) + '.pkl')

        # (N x num_samples, num_feats)
        features_rs = features.reshape(
            features.shape[0] * features.shape[1], features.shape[2])
        units_rs = units.reshape(units.shape[0] * units.shape[1],
                                 units.shape[2])  # (N x num_samples, 3)

        # Count the number of valid (non-zero) samples
        # inverse of invalid samples
        valid_rows = np.logical_not(np.all(features_rs == 0, axis=1))
        print('Model %s - Valid samples: %d / %d' %
              (JOINT_NAMES[joint_id], features_rs[valid_rows].shape[0], features_rs.shape[0]))

        # Fit decision tree to samples
        regressor = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
        regressor.fit(features_rs[valid_rows], units_rs[valid_rows])

        L = self.stochastic(regressor, features_rs, units_rs)

        # Print statistics on leafs
        leaf_ids = regressor.apply(features_rs)
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

    def predict(self, regressor, L, theta, qm0, img_seq, body_center, num_steps=NUM_STEPS, step_size=2):
        """Test the model on a single example.
        """
        num_test_img = img_seq.shape[0]
        qm = np.zeros((num_test_img, num_steps + 1, 3))
        qm[:, 0] = qm0
        joint_pred = np.zeros((num_test_img, 3))

        for i in range(num_steps):
            body_center_z = body_center[:, 2]
            f = self.get_single_sample_feature(
                img_seq, qm[:, i], body_center_z, theta)
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

    def get_single_sample_feature(self, img_seq, joint_time_seq, z, theta):
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
        depth_seq = img_seq[np.arange(
            img_seq.shape[0]), coor[:, 0], coor[:, 1]]
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
