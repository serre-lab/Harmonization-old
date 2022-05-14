import tensorflow as tf

def get_vgg():
    return tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling='avg',
            classes=1000,
            classifier_activation="linear",
    )