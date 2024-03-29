import tensorflow as tf

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Convolution2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '2a')(
          input_tensor)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters2,
      kernel_size,
      use_bias=False,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '2b')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '2c')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(
          x)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
  """A block that has a conv layer at shortcut.

  Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.

  Returns:
      Output tensor for the block.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Convolution2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '2a')(
          input_tensor)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '2b')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '2c')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(
          x)

  shortcut = tf.keras.layers.Convolution2D(
      filters3, (1, 1),
      use_bias=False,
      strides=strides,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name=conv_name_base + '1')(
          input_tensor)
  shortcut = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '1')(
          shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def ResNet50(num_classes):
  """Instantiates the ResNet50 architecture.

  Args:
    num_classes: `int` number of classes for image classification.

  Returns:
      A Keras model instance.
  """
  # Determine proper input shape
  if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, SIZE, SIZE)
    bn_axis = 1
  else:
    input_shape = (SIZE, SIZE, 3)
    bn_axis = 3

  img_input = tf.keras.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = tf.keras.layers.Convolution2D(
      64, (7, 7),
      use_bias=False,
      strides=(2, 2),
      padding='valid',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name='conv1')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(
          x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  features = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(features)
  x = tf.keras.layers.Dense(
      num_classes,
      activation='linear',
      kernel_initializer=tf.keras.initializers.RandomNormal(
          stddev=0.01),
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name='fc1000')(
          x)

  # Create model.
  return tf.keras.Model(img_input, x, name='resnet50')

def get_resnet50():
    return ResNet50(1000)