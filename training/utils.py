
import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr


def pm(metric):
    """Print metric results"""
    return float(metric.result())


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

    scores = {k: np.mean(sims[k]) for k in reducers}

    return scores


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

    scores = {k: np.mean(dice_scores[k]) for k in reducers}

    return scores


def _norm_numpy(h):
    h = np.array(tf.cast(h, tf.float32).numpy(), np.float32)
    h -= h.min()
    h /= h.max() + 1e-3
    return h


def _quick_saliency(strategy, model, x, y):
    with strategy.scope():
        with tf.GradientTape() as tape:
            tape.watch(x)

            y_pred = model(x, training=False)
            loss = tf.reduce_sum(y_pred * y, -1)

        sa_maps = tape.gradient(loss, x)

    sa_maps = tf.cast(sa_maps, tf.float32).numpy()
    sa_maps = np.mean(sa_maps, -1)
    sa_maps = np.abs(sa_maps)
    sa_maps = np.clip(sa_maps, np.percentile(sa_maps, 1,  axis=(1, 2), keepdims=True),
                      np.percentile(sa_maps, 99, axis=(1, 2), keepdims=True))
    sa_maps = np.abs(sa_maps)

    return sa_maps


def batch_saliency(strategy, model, x, y):
    sa_maps = None
    for batchx, batchy in tf.data.Dataset.from_tensor_slices((x, y)).batch(128):
        heatmaps = _quick_saliency(strategy, model, x, y)
        sa_maps = heatmaps if sa_maps is None else tf.concat([sa_maps, heatmaps], 0)
    return sa_maps
