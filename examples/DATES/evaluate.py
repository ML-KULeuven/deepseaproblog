import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def simple_evaluate_OD_classification(classifier, regressor, digits, dataset):
    acc = [0 for i in range(digits)]
    count = 0
    for I, _ in dataset:
        regions = regressor(I[0])
        for i in range(digits):
            C = tf.cast(tf.argmax(classifier(I[0], regions[i]), axis=-1), dtype=tf.int32)
            acc[i] += tf.reduce_sum(tf.where((C == I[1 + i]), 1, 0))
        count += I[0].shape[0]
    return [float(a / count) for a in acc]


def bbox_iou(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=0)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=0)

    xI1 = tf.maximum(x11, x21)
    xI2 = tf.minimum(x12, x22)
    
    yI1 = tf.minimum(y11, y21)
    yI2 = tf.maximum(y12, y22)

    x_corrector = tf.where(xI2 > xI1, 1., 0.)
    y_corrector = tf.where(yI1 > yI2, 1., 0.)

    inter_area = (xI2 - xI1) * (yI1 - yI2) * x_corrector * y_corrector

    bboxes1_area = (x12 - x11) * (y11 - y12)
    bboxes2_area = (x22 - x21) * (y21 - y22)

    union = (bboxes1_area + bboxes2_area) - inter_area

    return tf.maximum(inter_area / union, 0)

def simple_evaluate_regression_IoU(regressor, dataset, digits, mean=True, meanmean=False):
    iou = [[] for i in range(digits)]

    for I, _ in dataset:
        regions = regressor(I[0])
        for i in range(digits):
            mu, sigma = regions[i]
            mu_x, sigma_x = mu[:, 0], sigma[:, 0]
            mu_y, sigma_y = mu[:, 1], sigma[:, 1]

            target_x1, target_x2 = I[2 + i][:, 0] - 14 / 200, I[2 + i][:, 0] + 14 / 200
            target_y1, target_y2 = I[2 + i][:, 1] + 14 / 120, I[2 + i][:, 1] - 14 / 120
            target_box = tf.stack([target_x1, target_y1, target_x2, target_y2])

            pred_x1, pred_x2 = mu_x - sigma_x, mu_x + sigma_x
            pred_y1, pred_y2 = mu_y + sigma_y, mu_y - sigma_y
            pred_x1, pred_x2 = tf.maximum(pred_x1, 0.), tf.minimum(pred_x2, 1.)
            pred_y1, pred_y2 = tf.minimum(pred_y1, 1.), tf.maximum(pred_y2, 0.)
            pred_box = tf.stack([pred_x1, pred_y1, pred_x2, pred_y2])

            if mean:
                iou[i].append(tf.reduce_mean(bbox_iou(target_box, pred_box)))
            else:
                iou[i].append(tf.squeeze(bbox_iou(target_box, pred_box)))
    if mean and not meanmean:
        return [np.mean(i) for i in iou]
    elif mean and meanmean:
        return np.mean([np.mean(i) for i in iou])
    else:
        return tf.stack(iou, axis=-1)