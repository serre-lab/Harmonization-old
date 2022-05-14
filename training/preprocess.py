import tensorflow as tf
import numpy as np

BLUR_KERNEL_SIZE = 10
BLUR_SIGMA = 10
SIZE = 224
CROP_FRACTION = 0.875

MEAN_IMAGENET = tf.constant([0.485, 0.456, 0.406], shape=[3], dtype=tf.bfloat16)
STD_IMAGENET  =  tf.constant([0.229, 0.224, 0.225], shape=[3], dtype=tf.bfloat16)


def _gaussian_kernel(size, sigma):
  x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
  y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

  xs, ys = tf.meshgrid(x_range, y_range)
  kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))

  kernel = tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

  return tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

GAUSSIAN_KERNEL = tf.cast(_gaussian_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA), tf.float32)

def _gaussian_blur(heatmap):
    heatmap = tf.nn.conv2d(heatmap[None, :, :, :], GAUSSIAN_KERNEL, [1, 1, 1, 1], 'SAME')
    return heatmap[0]

def heatmap_diffusion(heatmap):
  heatmap = tf.cast(heatmap, tf.float32)

  heatmap = tf.image.resize(heatmap, (64, 64), method='bicubic')
  heatmap = _gaussian_blur(heatmap)
  heatmap = tf.image.resize(heatmap, (SIZE, SIZE))
  
  heatmap = tf.cast(heatmap, tf.bfloat16)
  
  heatmap = heatmap - tf.math.reduce_min(heatmap, keepdims=True)
  heatmap = heatmap / (tf.math.reduce_max(heatmap, keepdims=True) + 1e-5)

  return heatmap

def _distorted_bounding_box_crop(image_bytes,
                                heatmap_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  shape = tf.image.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  heatmap = tf.image.decode_and_crop_jpeg(heatmap_bytes, crop_window, channels=1)

  return image, heatmap


def decode_and_center_crop(image_bytes, heatmap_bytes, image_size):
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  # crop_fraction = image_size / (image_size + crop_padding)
  crop_padding = round(image_size * (1/CROP_FRACTION - 1))
  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize([image], [image_size, image_size], method='bicubic')[0]

  heatmap = tf.image.decode_and_crop_jpeg(heatmap_bytes, crop_window, channels=1)
  heatmap = tf.image.resize([heatmap], [image_size, image_size], method='bicubic')[0]

  return image, heatmap

def _at_least_x_are_equal(a, b, x):
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)

def decode_and_random_crop(image_bytes, heatmap_bytes, image_size):
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image, heatmap = _distorted_bounding_box_crop(
      image_bytes,
      heatmap_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)

  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image, heatmap = tf.cond(
      bad,
      lambda: decode_and_center_crop(image_bytes, heatmap_bytes, image_size),
      lambda: (tf.image.resize([image], [image_size, image_size], method='bicubic')[0], tf.image.resize([heatmap], [image_size, image_size], method='bicubic')[0]) 
  )

  return image, heatmap

DIVISOR = tf.cast(1.0 / 255.0, tf.bfloat16)
STD_DIVISOR = tf.cast(1.0 / STD_IMAGENET, tf.bfloat16)

def normalize(image):
  image = tf.cast(image, tf.bfloat16)

  image = image * DIVISOR
  image = image - MEAN_IMAGENET
  image = image * STD_DIVISOR

  return image

