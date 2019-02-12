from util import *

def draw_prediction(img, joints, paths, center, filename, nJoints, jointName):
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
