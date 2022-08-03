import tensorflow as tf

def get_mobilenet():
    return tf.keras.applications.mobilenet.MobileNet(weights=None)