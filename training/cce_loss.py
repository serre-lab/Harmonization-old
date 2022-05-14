
import tensorflow as tf
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

    @tf.function
    def train_step(iterator):
        def step_fn(inputs):

            images, heatmaps, labels, tokens = inputs
            images = normalize(images)

            labels = tf.cast(labels, tf.float32)
            tokens = tf.cast(tokens, tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(images)
                tape.watch(model.trainable_variables)

                y_pred = model(images, training=True)
                pred_loss = cross_entropy(labels, y_pred,
                                          from_logits=True, label_smoothing=label_smoothing)
                pred_loss = tf.reduce_mean(pred_loss)

                weight_loss = tf.add_n([tf.nn.l2_loss(v)
                                       for v in model.trainable_variables if 'bn/' not in v.name])
                weight_loss = tf.cast(weight_loss, tf.float32) * 1e-5

                loss = tf.cast(pred_loss, tf.float32) + tf.cast(weight_loss, tf.float32)

                scaled_loss = loss / strategy.num_replicas_in_sync

            optimizer.minimize(scaled_loss, model.trainable_variables, tape=tape)

            del tape

            training_loss.update_state(scaled_loss)
            training_accuracy.update_state(labels, y_pred)

            metric_cce_loss.update_state(pred_loss)
            metric_weight_loss.update_state(weight_loss)

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
