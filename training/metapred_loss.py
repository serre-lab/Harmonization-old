
import tensorflow as tf
import numpy as np
import tensorflow_graphics.image.pyramid as pyramid
import time
from preprocess import normalize

cross_entropy = tf.keras.losses.categorical_crossentropy


def get_train_val_step(strategy, model, optimizer, batch_size,
                       training_loss, val_loss,
                       metric_cce_loss, metric_weight_loss, metric_phi_loss,
                       mean_gradinput, std_gradinput, norm_grad,
                       training_accuracy, val_accuracy,
                       lambda_metapred=1.0, label_smoothing=0.1,
                       ):

    lambda_metapred = tf.cast(lambda_metapred, tf.float32)

    def standardize_cut(heatmaps, axes=(1, 2), epsilon=1e-5):
        means = tf.reduce_mean(heatmaps, axes, keepdims=True)
        stds = tf.math.reduce_std(heatmaps, axes, keepdims=True)

        heatmaps = heatmaps - means
        heatmaps = heatmaps / (stds + epsilon)

        heatmaps = tf.nn.relu(heatmaps)

        return heatmaps

    def _mse(a, b, t):
        return tf.reduce_mean(tf.square(a - b) * t[:, None, None, None])

    def metapred_loss(clickme_maps, predicted_maps, tokens):
        a = pyramid.downsample(clickme_maps[:, :, :, None], 5)
        h = pyramid.downsample(predicted_maps[:, :, :, None], 5)
        loss = tf.reduce_mean([_mse(a[0], h[0], tokens),
                               _mse(a[1], h[1], tokens),
                               _mse(a[2], h[2], tokens),
                               _mse(a[3], h[3], tokens),
                               _mse(a[4], h[4], tokens),
                               _mse(a[5], h[5], tokens)])
        return loss

    @tf.function
    def train_step(iterator):
        def step_fn(inputs):

            images, heatmaps, labels, tokens = inputs
            images = normalize(images)

            labels = tf.cast(labels, tf.float32)
            tokens = tf.cast(tokens, tf.float32)
            heatmaps = tf.cast(heatmaps, tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(images)
                tape.watch(model.trainable_variables)

                with tf.GradientTape() as tape_metapred:
                    tape_metapred.watch(images)
                    tape_metapred.watch(model.trainable_variables)

                    y_pred = model(images, training=True)
                    loss_metapred = tf.reduce_sum(y_pred * labels, -1)

                sa_maps = tf.cast(tape_metapred.gradient(loss_metapred, images), tf.float32)
                sa_maps = tf.reduce_mean(sa_maps, -1)

                sa_maps_preprocess = standardize_cut(sa_maps)
                heatmaps_preprocess = standardize_cut(heatmaps)

                sa_maps_preprocess = sa_maps_preprocess
                heatmaps_preprocess = heatmaps_preprocess[:, :, :, 0]

                # re-normalize before pyramidal
                _hm_max = tf.nn.relu(tf.math.reduce_max(
                    heatmaps_preprocess, (1, 2), keepdims=True)) + 1e-4
                _sa_max = tf.stop_gradient(tf.math.reduce_max(
                    sa_maps_preprocess, (1, 2), keepdims=True))

                heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max
                # done

                phi_loss = metapred_loss(sa_maps_preprocess, heatmaps_preprocess, tokens)

                pred_loss = cross_entropy(labels, y_pred,
                                          from_logits=True, label_smoothing=label_smoothing)
                pred_loss = tf.reduce_mean(pred_loss)

                weight_loss = tf.add_n([tf.nn.l2_loss(v)
                                       for v in model.trainable_variables if 'bn/' not in v.name])
                weight_loss = tf.cast(weight_loss, tf.float32) * 1e-5

                loss = pred_loss + weight_loss + phi_loss * lambda_metapred

                scaled_loss = loss / strategy.num_replicas_in_sync
                scaled_loss = tf.cast(scaled_loss, tf.float32)

            optimizer.minimize(scaled_loss, model.trainable_variables, tape=tape)

            del tape
            del tape_metapred

            training_loss.update_state(scaled_loss)
            training_accuracy.update_state(labels, y_pred)

            mean_gradinput.update_state(tf.reduce_mean(sa_maps))
            std_gradinput.update_state(tf.math.reduce_std(sa_maps))
            norm_grad.update_state(tf.reduce_mean(tf.abs(sa_maps)))

            metric_cce_loss.update_state(pred_loss)
            metric_weight_loss.update_state(weight_loss)
            metric_phi_loss.update_state(phi_loss)

        strategy.run(step_fn, args=(next(iterator),))

    @tf.function
    def test_step(iterator):
        def step_fn(inputs):

            images, _, labels = inputs
            images = normalize(images)

            logits = model(images, training=False)

            val_accuracy.update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))

    return train_step, test_step
