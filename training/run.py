import math
import sys
import time

import tensorflow as tf
from utils import pm

from models import get_resnet50
from dataset import get_imagenet_val_dataset, get_clickme_val_dataset, get_train_dataset, get_imagenet_val_dataset
from scheduler import LearningRateBatchScheduler, learning_rate_schedule_wrapper

import cce_loss
import metapred_loss


SIZE = 224


def run_epochs(get_step_function, epochs, base_learning_rate, scheduler, batch_size,
               steps_per_epoch, steps_per_eval, lambda_metapred, label_smoothing,
               step_multiplier,
               model_save_file='efficientnet.h5',
               mixup=False
               ):
    # create the val dataset
    best_val_accuracy = 0.0

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
        model = get_resnet50()

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=base_learning_rate, momentum=0.9, nesterov=True)

        model.optimizer = optimizer

        training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        metric_cce_loss = tf.keras.metrics.Mean('cce_loss', dtype=tf.float32)
        metric_weight_loss = tf.keras.metrics.Mean('weight_loss', dtype=tf.float32)
        metric_phi_loss = tf.keras.metrics.Mean('phi_loss', dtype=tf.float32)

        mean_gradinput = tf.keras.metrics.Mean('mu_gradinput', dtype=tf.float32)
        std_gradinput = tf.keras.metrics.Mean('std_gradinput', dtype=tf.float32)
        norm_grad = tf.keras.metrics.Mean('norm_grad', dtype=tf.float32)

        training_accuracy = tf.keras.metrics.CategoricalAccuracy(
            'train_sparse_categorical_accuracy', dtype=tf.float32)
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(
            'val_sparse_categorical_accuracy', dtype=tf.float32)

    train_ds = strategy.experimental_distribute_dataset(get_train_dataset(batch_size, mixup=mixup))
    test_ds = strategy.experimental_distribute_dataset(get_imagenet_val_dataset(batch_size*2))

    train_iterator = iter(train_ds)
    test_iterator = iter(test_ds)

    lr_schedule_cb = LearningRateBatchScheduler(
        schedule=learning_rate_schedule_wrapper(
            steps_per_epoch, lr_schedule=scheduler, base_learning_rate=base_learning_rate),
        model=model)

    train_step, test_step = get_step_function(strategy=strategy, model=model, optimizer=optimizer, batch_size=batch_size,
                                              lambda_metapred=lambda_metapred, label_smoothing=label_smoothing,
                                              training_loss=training_loss, val_loss=val_loss,
                                              metric_cce_loss=metric_cce_loss, metric_weight_loss=metric_weight_loss, metric_phi_loss=metric_phi_loss,
                                              mean_gradinput=mean_gradinput, std_gradinput=std_gradinput, norm_grad=norm_grad,
                                              training_accuracy=training_accuracy, val_accuracy=val_accuracy)

    for epoch_i in range(epochs):
        lr_schedule_cb.on_epoch_begin(epoch_i)

        lt = time.time()

        for step_i in range(steps_per_epoch):
            lr_schedule_cb.on_batch_begin(step_i)
            train_step(train_iterator)

            if step_i % 50 == 0:
                print(f"  [step] {step_i} || loss={pm(training_loss)} :: cce={pm(metric_cce_loss)} :: phi={pm(metric_phi_loss)} :: cce={pm(metric_weight_loss)} :: grad=({pm(mean_gradinput)}, {pm(std_gradinput)})")
                print(
                    f"        time to get 50 step {time.time() - lt} (train={round(pm(training_accuracy)*100,2)}%)")
                lt = time.time()

        for step_i in range(steps_per_eval):
            test_step(test_iterator)

        if val_accuracy.result().numpy() > best_val_accuracy and epoch_i > 20:
            model.save(model_save_file)
            best_val_accuracy = val_accuracy.result().numpy()

        print(f"\n[VAL] loss={pm(val_loss)} val_accuracy={pm(val_accuracy)} \n")


def get_arg(index, cls, default):
    try:
        val = cls(sys.argv[index])
        return val
    except:
        return default


if __name__ == "__main__":

    DEFAULT_LEARNING_RATE = get_arg(1, float, 0.7)
    LAMBDA_METAPRED = get_arg(2, float, 2.0)

    LR_SCHEDULE = [
        (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
    ]

    APPROX_IMAGENET_TRAINING_IMAGES = 1_281_167 + 327_680
    IMAGENET_VALIDATION_IMAGES = 50_000

    NUM_CLASSES = 1000
    EPOCHS = 90
    WARMUP_EPOCHS = 5

    PER_CORE_BATCH_SIZE = 128
    STEP_MULTIPLIER = 1
    MIXUP = False

    NB_EXPLANATIONS_TO_SAVE = 30

    NUM_CORES = 8
    BATCH_SIZE = PER_CORE_BATCH_SIZE * NUM_CORES
    STEPS_PER_EPOCHS = int(APPROX_IMAGENET_TRAINING_IMAGES // BATCH_SIZE * STEP_MULTIPLIER)
    STEPS_PER_EVAL = int(1.0 * math.ceil(IMAGENET_VALIDATION_IMAGES / BATCH_SIZE))

    val_clickme = get_clickme_val_dataset(100)
    x_val, h_val, y_val = next(val_clickme.take(1).as_numpy_iterator())

    train_clickme = get_clickme_val_dataset(100)
    x_train, h_train, y_train = next(train_clickme.take(1).as_numpy_iterator())

    run_epochs(
        get_step_function=metapred_loss.get_train_val_step,
        epochs=EPOCHS,
        base_learning_rate=DEFAULT_LEARNING_RATE,
        scheduler=LR_SCHEDULE,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCHS,
        steps_per_eval=STEPS_PER_EVAL,
        lambda_metapred=LAMBDA_METAPRED,
        label_smoothing=0.1,
        model_save_file='efficientnet.h5',
        min_lr=DEFAULT_LEARNING_RATE * 1e-3,
        step_multiplier=STEP_MULTIPLIER,
        mixup=MIXUP,
    )
