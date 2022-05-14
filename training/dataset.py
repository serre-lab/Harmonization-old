import os
import tensorflow as tf
import numpy as np

from augmentations import mixup, flip_left_right
from preprocess import decode_and_center_crop, decode_and_random_crop, heatmap_diffusion

shards_folder = "/mnt/disks/metapredictor-shards"

imagenet_train_shards = [
    f"/mnt/disks/imagenet/train/{f}" for f in os.listdir(f"/mnt/disks/imagenet/train") if 'train' in f]
imagenet_validation_shards = [f"/mnt/disks/imagenet/validation/{f}" for f in os.listdir(
    f"/mnt/disks/imagenet/validation/") if 'validation' in f]
train_clickme_shards = [
    f"{shards_folder}/{f}" for f in os.listdir(f"{shards_folder}") if 'train' in f]
val_clickme_shards = [f"{shards_folder}/{f}" for f in os.listdir(f"{shards_folder}") if 'test' in f]

AUTO = tf.data.AUTOTUNE
SIZE = 224

CLICKME_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "heatmap": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

IMAGENET_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
}

MEAN_IMAGENET = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=tf.bfloat16)
STD_IMAGENET = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=tf.bfloat16)


def _apply_mixup(dataset):
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(2)
    dataset = dataset.map(mixup, num_parallel_calls=AUTO)
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(1024)
    return dataset


def _preprocess_for_train(image_bytes, heatmap_bytes, image_size=SIZE):
    image, heatmap = decode_and_random_crop(image_bytes, heatmap_bytes, image_size)

    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.cast(image, tf.bfloat16)

    heatmap = tf.reshape(heatmap, [image_size, image_size, 1])
    heatmap = tf.cast(heatmap, tf.bfloat16)

    return image, heatmap


def _preprocess_for_eval(image_bytes, image_size=SIZE):
    image, _ = decode_and_center_crop(image_bytes, image_bytes, image_size)

    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.cast(image, tf.bfloat16)

    return image


def _init_shards(shards, training=False):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(shards, num_parallel_reads=AUTO, buffer_size=8 * 1024 * 1024)
    dataset = dataset.with_options(ignore_order)

    dataset = dataset.repeat()

    if training:
        dataset = dataset.shuffle(100)

    return dataset


def _parse_clickme_prototype(prototype, training=False):
    data = tf.io.parse_single_example(prototype, CLICKME_FEATURE_DESCRIPTION)

    if training:
        # distorsion crop / resize for image and heatmap
        image, heatmap = _preprocess_for_train(data['image'], data['heatmap'], image_size=SIZE)
    else:
        # load and resize (click-me image are already 256)
        image = tf.io.decode_jpeg(data['image'])
        image = tf.reshape(image, (256, 256, 3))
        image = tf.image.resize(image, (SIZE, SIZE), method='bicubic')
        image = tf.cast(image, tf.bfloat16)

        heatmap = tf.io.decode_jpeg(data['heatmap'])
        heatmap = tf.reshape(heatmap, (256, 256, 1))
        heatmap = tf.image.resize(heatmap, (SIZE, SIZE), method='bicubic')
        heatmap = tf.cast(heatmap, tf.bfloat16)

    heatmap = heatmap_diffusion(heatmap)

    # ensure all labels are one_hot for mixup
    label = tf.cast(data['label'], tf.int32)
    label = tf.one_hot(label, 1_000, dtype=tf.bfloat16)

    return image, heatmap, label


def _parse_imagenet_prototype(prototype, training=False):
    data = tf.io.parse_single_example(prototype, IMAGENET_FEATURE_DESCRIPTION)

    # imagenet label are [1-1000] -> [0, 999]
    # also, ensure all labels are one-hot for mixup
    label = tf.cast(data['image/class/label'], tf.int32) - 1
    label = tf.one_hot(label, 1_000, dtype=tf.bfloat16)

    if training:
        image, _ = _preprocess_for_train(data['image/encoded'], data['image/encoded'])
    else:
        image = _preprocess_for_eval(data['image/encoded'])

    heatmap = tf.zeros((SIZE, SIZE, 1), dtype=tf.bfloat16)

    return image, heatmap, label


def get_imagenet_val_dataset(batch_size):
    val_imagenet = _init_shards(imagenet_validation_shards, training=False).map(
        lambda proto: _parse_imagenet_prototype(proto, training=False), num_parallel_calls=AUTO)

    val_imagenet = val_imagenet.batch(batch_size, drop_remainder=True)
    val_imagenet = val_imagenet.prefetch(AUTO)

    return val_imagenet


def get_clickme_val_dataset(batch_size):
    val_clickme = _init_shards(val_clickme_shards, training=False).map(
        lambda proto: _parse_clickme_prototype(proto, training=False), num_parallel_calls=AUTO)

    val_clickme = val_clickme.batch(batch_size, drop_remainder=True)
    val_clickme = val_clickme.prefetch(AUTO)

    return val_clickme


def get_train_dataset(batch_size, mixup=True):
    clickme_dataset = _init_shards(train_clickme_shards, training=True).map(
        lambda proto: _parse_clickme_prototype(proto, training=True), num_parallel_calls=AUTO)
    imagenet_dataset = _init_shards(imagenet_train_shards, training=True).map(
        lambda proto: _parse_imagenet_prototype(proto, training=True), num_parallel_calls=AUTO)

    imagenet_dataset = imagenet_dataset.apply(tf.data.experimental.ignore_errors())

    if mixup:
        clickme_dataset = _apply_mixup(clickme_dataset)
        imagenet_dataset = _apply_mixup(imagenet_dataset)

    clickme_dataset = clickme_dataset.map(lambda x, h, y: (x, h, y, True), num_parallel_calls=AUTO)
    imagenet_dataset = imagenet_dataset.map(
        lambda x, h, y: (x, h, y, False), num_parallel_calls=AUTO)

    # 80% imagenet, 20% clickme
    train_dataset = tf.data.experimental.sample_from_datasets(
        [clickme_dataset, imagenet_dataset], weights=[0.2, 0.8])  # as click-me is ~25% of imagenet

    train_dataset = train_dataset.map(flip_left_right, num_parallel_calls=AUTO)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    train_dataset = train_dataset.prefetch(AUTO)

    return train_dataset
