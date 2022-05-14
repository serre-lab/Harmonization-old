import cv2
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
AUTO = tf.data.AUTOTUNE
BLUR_KERNEL_SIZE = 10
BLUR_SIGMA = 10

_feature_description = {
      "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def _tf_to_torch(t):
  t = tf.cast(t, tf.float32).numpy()
  if t.shape[-1] in [1, 3]:
    t = np.moveaxis(t, -1, 1)
  
  t = torch.tensor(t, requires_grad = True).cuda()

  return t


def set_size(w,h):
  """Set matplot figure size"""
  plt.rcParams["figure.figsize"] = [w,h]

def show(img, p=False, smooth=False, **kwargs):
  """ Display torch/tf tensor """ 
  try:
    img = img.detach().cpu()
  except:
    img = np.array(img)
  
  img = np.array(img, dtype=np.float32)

  # check if channel first
  if img.shape[0] == 1:
    img = img[0]
  elif img.shape[0] == 3:
    img = np.moveaxis(img, 0, -1)
  # check if cmap
  if img.shape[-1] == 1:
    img = img[:,:,0]
  # normalize
  if img.max() > 1 or img.min() < 0:
    img -= img.min(); img/=img.max()
  # check if clip percentile
  if p is not False:
    img = np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))
  
  if smooth and len(img.shape) == 2:
    img = gaussian_filter(img, smooth)

  plt.imshow(img, **kwargs)
  plt.axis('off')
  plt.grid(None)

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
    #heatmap = tf.nn.conv2d(heatmap, GAUSSIAN_KERNEL, [1, 1, 1, 1], 'SAME')
    #heatmap = tf.nn.conv2d(heatmap, GAUSSIAN_KERNEL, [1, 1, 1, 1], 'SAME')
    return heatmap[0]

def _random_crop(image, heatmap):
  seed = tf.random.uniform([2], maxval=10_000, dtype=tf.int32)
  crop_size = tf.random.uniform([], minval=224, maxval=256, dtype=tf.int32)
  
  cropped_image   = tf.image.stateless_random_crop(image, (crop_size, crop_size, 3), seed=seed)
  cropped_heatmap = tf.image.stateless_random_crop(heatmap, (crop_size, crop_size, 1), seed=seed)

  return cropped_image, cropped_heatmap


def parse_prototype(prototype, training=False):
  data    = tf.io.parse_single_example(prototype, _feature_description)

  image   = tf.io.decode_raw(data['image'], tf.float32)
  image   = tf.reshape(image, (256, 256, 3))
  image   = tf.cast(image, tf.float16)
  
  #heatmap = tf.io.decode_jpeg(data['heatmap'])
  heatmap = tf.io.decode_raw(data['heatmap'], tf.float32)
  heatmap = tf.reshape(heatmap, (256, 256, 1))

  image   = tf.image.resize(image, (224, 224), method='bilinear')
  image   = tf.cast(image, tf.float16)

  heatmap = tf.cast(heatmap, tf.float32)
  heatmap = tf.image.resize(heatmap, (64, 64), method="bilinear")
  heatmap = _gaussian_blur(heatmap)
  heatmap = tf.image.resize(heatmap, (224, 224), method="bilinear")
  heatmap = tf.cast(heatmap, tf.float16)

  label   = tf.cast(data['label'], tf.int32)
  label   = tf.one_hot(label, 1_000)

  return image, heatmap, label

def get_dataset(batch_size, training=False,path='data/clickme_val.tfrecords'):
    deterministic_order = tf.data.Options()
    deterministic_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset([path], num_parallel_reads=AUTO)
    dataset = dataset.with_options(deterministic_order) 
      
    dataset = dataset.map(parse_prototype, num_parallel_calls=AUTO)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset



def _torch_to_tf(t):
  try:
    t = t.detach()
  except:
    pass
  try:
    t = t.cpu()
  except:
    pass
  try:
    t = t.numpy()
  except:
    pass
  t = np.array(t)
  if t.shape[1] in [1, 3]:
    t = np.moveaxis(t, 1, -1)
  
  t = np.array(t, np.float32)

  return t

def find_last_conv_layer(model):
    last_conv = None

    modules_to_look = []
    for module_key in list(model._modules.keys())[::-1]:
      modules_to_look.append((model, module_key, []))

    while len(modules_to_look):
      base_module, module_key, path = modules_to_look.pop(0)
      module = base_module._modules[module_key]

      sub_modules = module._modules
      
      if not len(sub_modules):
        # a layer
        if module.__class__.__name__ == 'Conv2d' and module.out_channels > 1_000:
          print('found', path, module_key)
          return module
      else:
        for next_key in list(module._modules.keys()):
          modules_to_look = [(module, next_key, path + [module_key])] + modules_to_look
    
    # no conv found
    print('[ERROR] not found!')
    return False
