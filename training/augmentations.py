import tensorflow as tf

SIZE = 224

def mixup(images, heatmaps, labels, proba = 0.5):
  
  img1, img2 = images[0], images[1]
  hea1, hea2 = heatmaps[0], heatmaps[1]
  lab1, lab2 = labels[0], labels[1]

  p1 = tf.cast( tf.random.uniform([],0,1) <= proba, tf.bfloat16) 
  p2 = tf.cast( tf.random.uniform([],0,1) <= proba, tf.bfloat16) 
  
  alpha_1 = tf.random.uniform([], 0, 1, dtype=tf.bfloat16) * p1
  alpha_2 = tf.random.uniform([], 0, 1, dtype=tf.bfloat16) * p2

  img_mixup_1 = (1. - alpha_1) * img1 + alpha_1 * img2
  img_mixup_2 = (1. - alpha_2) * img2 + alpha_2 * img1

  hea_mixup_1 = (1. - alpha_1) * hea1 + alpha_1 * hea2
  hea_mixup_2 = (1. - alpha_2) * hea2 + alpha_2 * hea1
  
  label_mixup_1 = (1. - alpha_1) * lab1 + alpha_1 * lab2
  label_mixup_2 = (1. - alpha_2) * lab2 + alpha_2 * lab1

  images = tf.stack([img_mixup_1, img_mixup_2])
  heatmaps = tf.stack([hea_mixup_1, hea_mixup_2])
  labels = tf.stack([label_mixup_1, label_mixup_2])

  images = tf.reshape(images, (2, SIZE, SIZE, 3))
  heatmaps = tf.reshape(heatmaps, (2, SIZE, SIZE, 1))
  labels = tf.reshape(labels, (2, 1000))

  return images, heatmaps, labels

def flip_left_right(image, heatmap, label, token):

  seed = tf.random.uniform([2], maxval=1_000, dtype=tf.int32)

  image = tf.image.stateless_random_flip_left_right(image, seed)
  heatmap = tf.image.stateless_random_flip_left_right(heatmap, seed)
  
  return image, heatmap, label, token
