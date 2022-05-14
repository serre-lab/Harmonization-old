import time
from scipy.stats import spearmanr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def spearmanr_sim(explanations1, explanations2, reducers=[1, 4, 16]):
    sims = {k: [] for k in reducers}

    if explanations1.shape[-1] != 1:
        explanations1 = explanations1[:, :, :, None]
    if explanations2.shape[-1] != 1:
        explanations2 = explanations2[:, :, :, None]

    explanations1 = tf.cast(explanations1, tf.float32).numpy()
    explanations2 = tf.cast(explanations2, tf.float32).numpy()

    for reducer in reducers:
        sz = int(explanations1.shape[1] / reducer)
        explanations1_resize = tf.image.resize(explanations1, (sz, sz)).numpy()
        explanations2_resize = tf.image.resize(explanations2, (sz, sz)).numpy()

        for x1, x2 in zip(explanations1_resize, explanations2_resize):
            rho, _ = spearmanr(x1.flatten(), x2.flatten())
            sims[reducer].append(rho)

    return sims


def dice(explanations1, explanations2, reducers=[1, 4, 16], p=10):
    dice_scores = {k: [] for k in reducers}

    if explanations1.shape[-1] != 1:
        explanations1 = explanations1[:, :, :, None]
    if explanations2.shape[-1] != 1:
        explanations2 = explanations2[:, :, :, None]

    explanations1 = tf.cast(explanations1, tf.float32).numpy()
    explanations2 = tf.cast(explanations2, tf.float32).numpy()

    for reducer in reducers:
        sz = int(explanations1.shape[1] / reducer)
        explanations1_resize = tf.image.resize(explanations1, (sz, sz)).numpy()
        explanations2_resize = tf.image.resize(explanations2, (sz, sz)).numpy()

        for j, (x1, x2) in enumerate(zip(explanations1_resize, explanations2_resize)):
            x1 = x1 > np.percentile(x1, 100-p, (0, 1))
            x2 = x2 > np.percentile(x2, 100-p, (0, 1))

            dice_score = 2.0 * np.sum(x1 * x2) / np.sum(x1 + x2)
            dice_scores[reducer].append(dice_score)

    return dice_scores


def iou(explanations1, explanations2, reducers=[1, 4, 16], p=10):
    iou_scores = {k: [] for k in reducers}

    if explanations1.shape[-1] != 1:
        explanations1 = explanations1[:, :, :, None]
    if explanations2.shape[-1] != 1:
        explanations2 = explanations2[:, :, :, None]

    explanations1 = tf.cast(explanations1, tf.float32).numpy()
    explanations2 = tf.cast(explanations2, tf.float32).numpy()

    for reducer in reducers:
        sz = int(explanations1.shape[1] / reducer)
        explanations1_resize = tf.image.resize(explanations1, (sz, sz)).numpy()
        explanations2_resize = tf.image.resize(explanations2, (sz, sz)).numpy()

        for j, (x1, x2) in enumerate(zip(explanations1_resize, explanations2_resize)):
            x1 = x1 > np.percentile(x1, 100-p, (0, 1))
            x2 = x2 > np.percentile(x2, 100-p, (0, 1))

            iou_inter = np.sum(np.logical_and(x1, x2))
            iou_union = np.sum(np.logical_or(x1, x2))

            iou_score = iou_inter / iou_union

            iou_scores[reducer].append(iou_score)

    return iou_scores
